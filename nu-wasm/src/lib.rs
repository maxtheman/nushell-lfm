//! Nushell in WebAssembly over an in-memory workspace.
//!
//! The filesystem commands of Nushell need an operating system, so they are left out of a wasm build.
//! This crate supplies `ls`, `glob`, `open`, `save`, `cd`, `pwd`, `path exists` and `print` over a map of
//! path -> bytes. `set_files` loads a workspace. `run` evaluates source in a fresh engine, the way
//! `nu -c` does; `session_run` keeps one engine alive so variables, files and the current directory
//! carry over from one line to the next, the way a REPL does.

use std::collections::BTreeMap;
use std::sync::Mutex;

use nu_engine::{eval_block, CallExt};
use nu_protocol::debugger::WithoutDebug;
use nu_protocol::engine::{Call, Command, EngineState, Stack, StateWorkingSet};
use nu_protocol::{
    Category, IntoPipelineData, IntoValue, PipelineData, ShellError, Signature, Span, SyntaxShape, Type, Value,
};
use wasm_bindgen::prelude::*;

const ROOT: &str = "/workspace";

static FILES: Mutex<BTreeMap<String, Vec<u8>>> = Mutex::new(BTreeMap::new());
static PRINTED: Mutex<String> = Mutex::new(String::new());
static CHANGED: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// The current directory, relative to the workspace root ("" is the root).
static CWD: Mutex<String> = Mutex::new(String::new());
static SESSION: Mutex<Option<(EngineState, Stack)>> = Mutex::new(None);

fn fail(msg: impl Into<String>, span: Span) -> ShellError {
    ShellError::Generic(nu_protocol::shell_error::generic::GenericError::new(msg.into(), "here", span))
}

/// A path as the workspace stores it: relative to the root, resolved against the current directory.
fn relative(path: &str) -> String {
    let joined = match path.strip_prefix(ROOT) {
        Some(rest) => rest.to_string(),
        None if path.starts_with('/') => path.to_string(),
        None => format!("{}/{}", CWD.lock().unwrap(), path),
    };
    let mut parts: Vec<&str> = Vec::new();
    for part in joined.split('/') {
        match part {
            "" | "." => {}
            ".." => { parts.pop(); }
            other => parts.push(other),
        }
    }
    parts.join("/")
}

/// A stored path as the user should see it from the current directory.
fn shown(path: &str) -> String {
    let cwd = CWD.lock().unwrap().clone();
    if cwd.is_empty() {
        return path.to_string();
    }
    path.strip_prefix(&format!("{cwd}/")).unwrap_or(path).to_string()
}

fn cwd_path() -> String {
    let cwd = CWD.lock().unwrap();
    if cwd.is_empty() { ROOT.to_string() } else { format!("{ROOT}/{cwd}") }
}

/// Every directory implied by the stored files.
fn directories(files: &BTreeMap<String, Vec<u8>>) -> Vec<String> {
    let mut dirs = std::collections::BTreeSet::new();
    for path in files.keys() {
        let mut parts: Vec<&str> = path.split('/').collect();
        parts.pop();
        while !parts.is_empty() {
            dirs.insert(parts.join("/"));
            parts.pop();
        }
    }
    dirs.into_iter().collect()
}

fn expand_braces(pattern: &str) -> Vec<String> {
    if let (Some(open), Some(close)) = (pattern.find('{'), pattern.find('}')) {
        if open < close {
            let (head, tail) = (&pattern[..open], &pattern[close + 1..]);
            return pattern[open + 1..close]
                .split(',')
                .flat_map(|option| expand_braces(&format!("{head}{option}{tail}")))
                .collect();
        }
    }
    vec![pattern.to_string()]
}

fn segment_matches(pattern: &[u8], name: &[u8]) -> bool {
    match (pattern.first(), name.first()) {
        (None, None) => true,
        (Some(b'*'), _) => {
            segment_matches(&pattern[1..], name) || (!name.is_empty() && segment_matches(pattern, &name[1..]))
        }
        (Some(b'?'), Some(_)) => segment_matches(&pattern[1..], &name[1..]),
        (Some(p), Some(n)) if p == n => segment_matches(&pattern[1..], &name[1..]),
        _ => false,
    }
}

fn path_matches(pattern: &[&str], path: &[&str]) -> bool {
    match pattern.first() {
        None => path.is_empty(),
        Some(&"**") => {
            // like native glob, `dir/**/*` also matches `dir` itself
            (path.is_empty() && pattern.len() == 2 && pattern[1] == "*")
                || path_matches(&pattern[1..], path)
                || (!path.is_empty() && path_matches(pattern, &path[1..]))
        }
        Some(segment) => {
            !path.is_empty()
                && segment_matches(segment.as_bytes(), path[0].as_bytes())
                && path_matches(&pattern[1..], &path[1..])
        }
    }
}

/// Files and directories matching a glob, as (relative path, is_dir), sorted.
fn glob_entries(pattern: &str) -> Vec<(String, bool)> {
    let files = FILES.lock().unwrap();
    let mut entries: Vec<(String, bool)> = files.keys().map(|p| (p.clone(), false)).collect();
    entries.extend(directories(&files).into_iter().map(|d| (d, true)));
    entries.sort();
    let patterns = expand_braces(&relative(pattern));
    // Like native glob, `**/*` also matches the directory it starts from.
    if patterns.iter().any(|p| p == "**/*" || p == "**") {
        entries.insert(0, (String::new(), true));
    }
    entries
        .into_iter()
        .filter(|(path, _)| {
            if path.is_empty() {
                return true;
            }
            let parts: Vec<&str> = path.split('/').collect();
            patterns.iter().any(|p| path_matches(&p.split('/').collect::<Vec<_>>(), &parts))
        })
        .collect()
}

fn text_arg(value: Value, span: Span) -> Result<String, ShellError> {
    match value {
        Value::String { val, .. } | Value::Glob { val, .. } => Ok(val),
        other => Err(fail(format!("expected a path, got {}", other.get_type()), span)),
    }
}

#[derive(Clone)]
struct Pwd;
impl Command for Pwd {
    fn name(&self) -> &str { "pwd" }
    fn signature(&self) -> Signature {
        Signature::build("pwd").input_output_type(Type::Nothing, Type::String).category(Category::FileSystem)
    }
    fn description(&self) -> &str { "Return the current directory." }
    fn run(&self, _: &EngineState, _: &mut Stack, call: &Call, _: PipelineData) -> Result<PipelineData, ShellError> {
        Ok(Value::string(cwd_path(), call.head).into_pipeline_data())
    }
}

#[derive(Clone)]
struct Cd;
impl Command for Cd {
    fn name(&self) -> &str { "cd" }
    fn signature(&self) -> Signature {
        Signature::build("cd")
            .optional("path", SyntaxShape::String, "The directory to change to; the workspace root when omitted.")
            .input_output_type(Type::Nothing, Type::Nothing)
            .category(Category::FileSystem)
    }
    fn description(&self) -> &str { "Change the current directory inside the workspace." }
    fn run(&self, engine: &EngineState, stack: &mut Stack, call: &Call, _: PipelineData) -> Result<PipelineData, ShellError> {
        let given: Option<Value> = call.opt(engine, stack, 0)?;
        let target = match given {
            Some(value) => relative(&text_arg(value, call.head)?),
            None => String::new(),
        };
        if !target.is_empty() && !directories(&FILES.lock().unwrap()).contains(&target) {
            return Err(fail(format!("Directory not found: {target}"), call.head));
        }
        *CWD.lock().unwrap() = target;
        stack.add_env_var("PWD".into(), Value::string(cwd_path(), call.head));
        Ok(PipelineData::empty())
    }
}

#[derive(Clone)]
struct Glob;
impl Command for Glob {
    fn name(&self) -> &str { "glob" }
    fn signature(&self) -> Signature {
        Signature::build("glob")
            .required("glob", SyntaxShape::OneOf(vec![SyntaxShape::String, SyntaxShape::GlobPattern]), "The glob expression.")
            .switch("no-dir", "Leave directories out of the results.", Some('D'))
            .switch("no-file", "Leave files out of the results.", Some('F'))
            .input_output_type(Type::Nothing, Type::List(Box::new(Type::String)))
            .category(Category::FileSystem)
    }
    fn description(&self) -> &str { "List the workspace paths matching a glob pattern." }
    fn run(&self, engine: &EngineState, stack: &mut Stack, call: &Call, _: PipelineData) -> Result<PipelineData, ShellError> {
        let pattern = text_arg(call.req(engine, stack, 0)?, call.head)?;
        let no_dir = call.has_flag(engine, stack, "no-dir")?;
        let no_file = call.has_flag(engine, stack, "no-file")?;
        let found = glob_entries(&pattern)
            .into_iter()
            .filter(|(_, is_dir)| if *is_dir { !no_dir } else { !no_file })
            .map(|(path, _)| Value::string(if path.is_empty() { ROOT.to_string() } else { format!("{ROOT}/{path}") }, call.head))
            .collect();
        Ok(Value::list(found, call.head).into_pipeline_data())
    }
}

#[derive(Clone)]
struct Ls;
impl Command for Ls {
    fn name(&self) -> &str { "ls" }
    fn signature(&self) -> Signature {
        Signature::build("ls")
            .optional("pattern", SyntaxShape::OneOf(vec![SyntaxShape::String, SyntaxShape::GlobPattern]), "The path or glob to list.")
            .switch("all", "Accepted for compatibility; the workspace has no hidden files.", Some('a'))
            .input_output_type(Type::Nothing, Type::table())
            .category(Category::FileSystem)
    }
    fn description(&self) -> &str { "List workspace files as a table of name, type and size." }
    fn run(&self, engine: &EngineState, stack: &mut Stack, call: &Call, _: PipelineData) -> Result<PipelineData, ShellError> {
        let span = call.head;
        let given: Option<Value> = call.opt(engine, stack, 0)?;
        let pattern = match given {
            Some(value) => text_arg(value, span)?,
            None => "*".to_string(),
        };
        let is_dir = directories(&FILES.lock().unwrap()).contains(&relative(&pattern));
        let pattern = if relative(&pattern).is_empty() {
            "*".to_string()
        } else if is_dir {
            format!("{}/*", relative(&pattern))
        } else {
            pattern
        };
        // unlike glob, `ls dir/**/*` does not list `dir` itself
        let own = relative(pattern.strip_suffix("/**/*").unwrap_or(""));
        let entries: Vec<(String, bool)> = glob_entries(&pattern)
            .into_iter()
            .filter(|(path, _)| !path.is_empty() && !(pattern.ends_with("/**/*") && *path == own))
            .collect();
        if entries.is_empty() {
            return Err(fail(format!("No matches found for {pattern}"), span));
        }
        let files = FILES.lock().unwrap();
        let rows = entries
            .into_iter()
            .map(|(path, is_dir)| {
                // A directory's size is what the disk the data was generated on (APFS) reports: 32 bytes for
                // each entry, counting `.` and `..`. Reference values such as `ls | get size | math sum` include it.
                let size = if is_dir {
                    let prefix = format!("{path}/");
                    let children: std::collections::BTreeSet<&str> = files
                        .keys()
                        .filter_map(|k| k.strip_prefix(&prefix))
                        .map(|rest| rest.split('/').next().unwrap_or(rest))
                        .collect();
                    32 * (children.len() as i64 + 2)
                } else {
                    files.get(&path).map(|bytes| bytes.len() as i64).unwrap_or(0)
                };
                let mut record = nu_protocol::Record::new();
                record.push("name", Value::string(shown(&path), span));
                record.push("type", Value::string(if is_dir { "dir" } else { "file" }, span));
                record.push("size", Value::filesize(size, span));
                // The workspace has no clock; every file carries the same fixed time.
                record.push("modified", Value::date(chrono::DateTime::parse_from_rfc3339("2026-03-02T08:00:00+00:00").unwrap(), span));
                Value::record(record, span)
            })
            .collect();
        Ok(Value::list(rows, span).into_pipeline_data())
    }
}

#[derive(Clone)]
struct Open;
impl Command for Open {
    fn name(&self) -> &str { "open" }
    fn signature(&self) -> Signature {
        Signature::build("open")
            .required("path", SyntaxShape::OneOf(vec![SyntaxShape::String, SyntaxShape::GlobPattern]), "The file to open.")
            .switch("raw", "Return the text without parsing it.", Some('r'))
            .input_output_type(Type::Nothing, Type::Any)
            .category(Category::FileSystem)
    }
    fn description(&self) -> &str { "Read a workspace file; JSON, TOML and YAML are parsed unless --raw is given." }
    fn run(&self, engine: &EngineState, stack: &mut Stack, call: &Call, _: PipelineData) -> Result<PipelineData, ShellError> {
        let span = call.head;
        let path = relative(&text_arg(call.req(engine, stack, 0)?, span)?);
        let raw = call.has_flag(engine, stack, "raw")?;
        let bytes = FILES.lock().unwrap().get(&path).cloned();
        let Some(bytes) = bytes else {
            return Err(fail(format!("File not found: {path}"), span));
        };
        // A byte stream, as native `open` returns, so text commands treat it the same way.
        let text = || {
            PipelineData::byte_stream(
                nu_protocol::ByteStream::read(std::io::Cursor::new(bytes.clone()), span, engine.signals().clone(), nu_protocol::ByteStreamType::Unknown),
                None,
            )
        };
        let parser = match path.rsplit('.').next() {
            Some("json") if !raw => Some("from json"),
            Some("toml") if !raw => Some("from toml"),
            Some("yaml") | Some("yml") if !raw => Some("from yaml"),
            _ => None,
        };
        match parser.and_then(|name| engine.find_decl(name.as_bytes(), &[])) {
            Some(decl) => engine.get_decl(decl).run(engine, stack, call, text()),
            None => Ok(text()),
        }
    }
}

#[derive(Clone)]
struct Save;
impl Command for Save {
    fn name(&self) -> &str { "save" }
    fn signature(&self) -> Signature {
        Signature::build("save")
            .required("path", SyntaxShape::String, "The file to write.")
            .switch("force", "Overwrite an existing file.", Some('f'))
            .input_output_type(Type::Any, Type::Nothing)
            .category(Category::FileSystem)
    }
    fn description(&self) -> &str { "Write the input to a workspace file." }
    fn run(&self, engine: &EngineState, stack: &mut Stack, call: &Call, input: PipelineData) -> Result<PipelineData, ShellError> {
        let span = call.head;
        let path = relative(&text_arg(call.req(engine, stack, 0)?, span)?);
        let force = call.has_flag(engine, stack, "force")?;
        if !force && FILES.lock().unwrap().contains_key(&path) {
            return Err(fail(format!("Destination file already exists: {path}"), span));
        }
        let value = input.into_value(span)?;
        let text = match value {
            Value::String { val, .. } => val,
            other => other.to_expanded_string("\n", &engine.get_config()),
        };
        FILES.lock().unwrap().insert(path.clone(), text.into_bytes());
        CHANGED.lock().unwrap().push(path);
        Ok(PipelineData::empty())
    }
}

#[derive(Clone)]
struct PathExists;
impl Command for PathExists {
    fn name(&self) -> &str { "path exists" }
    fn signature(&self) -> Signature {
        Signature::build("path exists").input_output_type(Type::String, Type::Bool).category(Category::Path)
    }
    fn description(&self) -> &str { "Check whether a path exists in the workspace." }
    fn run(&self, _: &EngineState, _: &mut Stack, call: &Call, input: PipelineData) -> Result<PipelineData, ShellError> {
        let span = call.head;
        let path = relative(&text_arg(input.into_value(span)?, span)?);
        let files = FILES.lock().unwrap();
        let exists = path.is_empty() || files.contains_key(&path) || directories(&files).contains(&path);
        Ok(Value::bool(exists, span).into_pipeline_data())
    }
}

#[derive(Clone)]
struct PathType;
impl Command for PathType {
    fn name(&self) -> &str { "path type" }
    fn signature(&self) -> Signature {
        Signature::build("path type")
            .input_output_types(vec![(Type::String, Type::String), (Type::List(Box::new(Type::String)), Type::List(Box::new(Type::String)))])
            .category(Category::Path)
    }
    fn description(&self) -> &str { "Say whether a workspace path is a file or a dir; nothing when it does not exist." }
    fn run(&self, _: &EngineState, _: &mut Stack, call: &Call, input: PipelineData) -> Result<PipelineData, ShellError> {
        let span = call.head;
        let kind = |value: Value| -> Result<Value, ShellError> {
            let path = relative(&text_arg(value, span)?);
            let files = FILES.lock().unwrap();
            Ok(if files.contains_key(&path) {
                Value::string("file", span)
            } else if path.is_empty() || directories(&files).contains(&path) {
                Value::string("dir", span)
            } else {
                Value::nothing(span)
            })
        };
        match input.into_value(span)? {
            Value::List { vals, .. } => Ok(Value::list(vals.into_iter().map(kind).collect::<Result<Vec<_>, _>>()?, span).into_pipeline_data()),
            one => Ok(kind(one)?.into_pipeline_data()),
        }
    }
}

#[derive(Clone)]
struct Print;
impl Command for Print {
    fn name(&self) -> &str { "print" }
    fn signature(&self) -> Signature {
        Signature::build("print")
            .rest("rest", SyntaxShape::Any, "The values to print.")
            .input_output_type(Type::Any, Type::Nothing)
            .category(Category::Strings)
    }
    fn description(&self) -> &str { "Print the given values." }
    fn run(&self, engine: &EngineState, stack: &mut Stack, call: &Call, input: PipelineData) -> Result<PipelineData, ShellError> {
        let config = engine.get_config();
        let mut values: Vec<Value> = call.rest(engine, stack, 0)?;
        if values.is_empty() {
            values.push(input.into_value(call.head)?);
        }
        let mut out = PRINTED.lock().unwrap();
        for value in values {
            out.push_str(&value.to_expanded_string("\n", &config));
            out.push('\n');
        }
        Ok(PipelineData::empty())
    }
}

fn engine() -> EngineState {
    let mut engine = nu_command::add_shell_command_context(nu_cmd_lang::create_default_context());
    let delta = {
        let mut working_set = StateWorkingSet::new(&engine);
        let commands: Vec<Box<dyn Command>> = vec![
            Box::new(Pwd), Box::new(Cd), Box::new(Glob), Box::new(Ls), Box::new(Open), Box::new(Save), Box::new(PathExists), Box::new(PathType), Box::new(Print),
        ];
        for command in commands {
            working_set.add_decl(command);
        }
        working_set.render()
    };
    let _ = engine.merge_delta(delta);
    engine.add_env_var("PWD".into(), ROOT.into_value(Span::unknown()));
    // There are no threads in a browser tab, so par-each runs its closure one item at a time.
    let delta = {
        let mut working_set = StateWorkingSet::new(&engine);
        nu_parser::parse(&mut working_set, None, b"alias par-each = each", false);
        working_set.render()
    };
    let _ = engine.merge_delta(delta);
    engine
}

/// Evaluate source in the given engine. Ok is what Nu would print; Err is (message, was it a parse error).
fn evaluate_in(engine: &mut EngineState, stack: &mut Stack, code: &str) -> Result<String, (String, bool)> {
    let (block, delta) = {
        let mut working_set = StateWorkingSet::new(engine);
        let block = nu_parser::parse(&mut working_set, None, code.as_bytes(), false);
        if let Some(err) = working_set.parse_errors.first() {
            return Err((format!("nu::parser: {err}"), true));
        }
        if let Some(err) = working_set.compile_errors.first() {
            return Err((format!("nu::compile: {err}"), true));
        }
        (block, working_set.render())
    };
    engine.merge_delta(delta).map_err(|e| (e.to_string(), false))?;
    let config = engine.get_config().clone();
    let result = eval_block::<WithoutDebug>(engine, stack, &block, PipelineData::empty())
        .map_err(|e| (e.to_string(), false))?;
    let value = result.body.into_value(Span::unknown()).map_err(|e| (e.to_string(), false))?;
    let shown = match value {
        Value::Nothing { .. } => String::new(),
        Value::Error { error, .. } => return Err((error.to_string(), false)),
        Value::String { val, .. } => val,
        other => {
            // Anything else goes through `table`, as it does at a terminal.
            let rendered = {
                let mut working_set = StateWorkingSet::new(engine);
                let table = nu_parser::parse(&mut working_set, None, b"table", false);
                let delta = working_set.render();
                engine.merge_delta(delta).ok().and_then(|_| {
                    eval_block::<WithoutDebug>(engine, stack, &table, other.clone().into_pipeline_data()).ok()
                })
            };
            match rendered.and_then(|data| data.body.into_value(Span::unknown()).ok()) {
                Some(Value::String { val, .. }) => val,
                _ => other.to_expanded_string("\n", &config),
            }
        }
    };
    Ok(shown)
}

fn report(outcome: Result<String, (String, bool)>) -> String {
    let mut stdout = std::mem::take(&mut *PRINTED.lock().unwrap());
    let (ok, stderr, parse_error) = match outcome {
        Ok(text) => {
            stdout.push_str(&text);
            (true, String::new(), false)
        }
        Err((message, parse_error)) => (false, message, parse_error),
    };
    serde_json::json!({
        "ok": ok,
        "stdout": stdout.trim_end(),
        "stderr": stderr,
        "parse_error": parse_error,
        "changed": *CHANGED.lock().unwrap(),
        "cwd": cwd_path(),
    })
    .to_string()
}

/// Replace the workspace with `{path: text}` given as JSON.
#[wasm_bindgen]
pub fn set_files(files_json: &str) -> Result<(), JsValue> {
    let parsed: BTreeMap<String, String> =
        serde_json::from_str(files_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    CWD.lock().unwrap().clear();
    *FILES.lock().unwrap() = parsed.into_iter().map(|(k, v)| (relative(&k), v.into_bytes())).collect();
    Ok(())
}

/// Run Nushell source the way `nu -c` does: a fresh engine, starting at the workspace root.
/// Returns JSON: ok, stdout, stderr, parse_error, changed, cwd.
#[wasm_bindgen]
pub fn run(code: &str) -> String {
    PRINTED.lock().unwrap().clear();
    CHANGED.lock().unwrap().clear();
    CWD.lock().unwrap().clear();
    let mut engine = engine();
    let mut stack = Stack::new();
    report(evaluate_in(&mut engine, &mut stack, code))
}

/// Run one line in the long-lived session: variables, saved files and the current directory persist.
#[wasm_bindgen]
pub fn session_run(code: &str) -> String {
    PRINTED.lock().unwrap().clear();
    CHANGED.lock().unwrap().clear();
    let mut session = SESSION.lock().unwrap();
    let (engine, stack) = session.get_or_insert_with(|| (engine(), Stack::new()));
    report(evaluate_in(engine, stack, code))
}

/// Forget the session's variables and return to the workspace root. Files are left as they are.
#[wasm_bindgen]
pub fn session_reset() {
    *SESSION.lock().unwrap() = None;
    CWD.lock().unwrap().clear();
}

/// The files of the workspace as JSON `{path: text}`, for showing what a session changed.
#[wasm_bindgen]
pub fn get_files() -> String {
    let files = FILES.lock().unwrap();
    let map: BTreeMap<&String, String> = files.iter().map(|(k, v)| (k, String::from_utf8_lossy(v).into_owned())).collect();
    serde_json::to_string(&map).unwrap_or_default()
}

/// The Nushell release this build links.
#[wasm_bindgen]
pub fn nu_version() -> String {
    "0.117.0".to_string()
}
