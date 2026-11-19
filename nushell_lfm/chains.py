"""Command chains: a walk on a graph of data kinds, kept only if the chain runs and repeats."""

import json
import random

from .core import ROOT
from .catalog import NU_COMMANDS
from .environments import OPTIONAL_KEYS, run_nu_many, same_output, save_nu_cache

_LOG_RX = r"'(?<ts>\S+)\s+(?<level>\S+)\s+(?<msg>.*)'"
_REL = "each {|p| $p | path relative-to (pwd)}"

def _priors() -> tuple[dict, dict, dict]:
    """Two sets of command weights, and where they came from.

    NATIVE: how often each command heads a pipeline stage in the example code of the Nushell book and
    cookbook (data/nushell_examples_usage.json). No translation is involved; this is the main prior.

    PORTED: Unix command frequencies from public corpora (data/command_usage.json), carried over through
    data/unix_to_nushell.json: the Nushell book's own Bash table plus a short supplementary list kept apart
    in that file. It decides how the dataset is divided among the categories, and steers a capped minority
    of chains; it never decides what a Nushell pipeline looks like."""
    known = {c["name"] for c in NU_COMMANDS}
    native, ported, info = {}, {}, {"native": None, "ported": None, "lengths": {}}
    examples_file = ROOT / "data" / "nushell_examples_usage.json"
    if examples_file.exists():
        examples = json.loads(examples_file.read_text(encoding="utf-8"))
        native = {k: v for k, v in examples["share"].items() if k in known}
        info["native"] = {"pages": examples["source"]["pages"], "command_lines": examples["source"]["command_lines"], "commands": len(native)}
        info["lengths"]["native"] = {int(k): v for k, v in examples["pipeline_lengths"].items()}
    usage_file, map_file = ROOT / "data" / "command_usage.json", ROOT / "data" / "unix_to_nushell.json"
    if usage_file.exists() and map_file.exists():
        usage, mapping = json.loads(usage_file.read_text(encoding="utf-8")), json.loads(map_file.read_text(encoding="utf-8"))
        book = mapping["map"]
        extra = {unix: entry["nu"] for unix, entry in mapping.get("suggested_not_in_source", {}).items() if unix not in book}
        mass = {"book": 0.0, "supplementary": 0.0}
        for origin, table in (("book", book), ("supplementary", extra)):
            for unix, targets in table.items():
                share = usage["combined"].get(unix, 0.0)
                targets = [t for t in targets if t in known]
                for target in targets:  # a Unix command's usage is shared among its Nushell equivalents
                    ported[target] = ported.get(target, 0.0) + share / len(targets)
                mass[origin] += share if targets else 0.0
        info["ported"] = {"sources": [s["id"] for s in usage["sources"]], "commands": len(ported), "usage_covered": {k: round(v, 3) for k, v in mass.items()}}
        info["lengths"]["ported"] = {int(k): v for k, v in usage.get("pipeline_lengths_combined", {}).items()}
    return native, ported, info

NATIVE, PORTED, PRIOR_INFO = _priors()
PORTED_LIMIT = 0.3  # at most this share of chains may be steered by the translated Unix weights

def _op(name, src, dst, commands, make, code, say, alt=None, wrong=None, fields=None, binds=False, auto=False):
    return {"name": name, "src": src, "dst": dst, "commands": commands, "make": make, "code": code, "say": say,
            "alt": alt, "wrong": wrong or (lambda p: []), "fields": fields, "binds": binds, "auto": auto}

def _pick(rng, items):
    return rng.choice(list(items)) if items else None

def _files_of(env, kind):
    layout, dirs = env["layout"], env["layout"]["dirs"]
    if kind == "log":
        return [f"{dirs['logs']}/{n}" for n in layout["log_names"]]
    if kind == "text":
        return sorted(p for p in env["files"] if not p.endswith((".log", ".json")))
    return sorted(env["files"])

def _folders(env):
    dirs = env["layout"]["dirs"]
    return [(dirs[r], f"{dirs[r]}/**/*") for r in ("src", "cfg", "logs", "docs", "scripts")] + [("the whole project", "**/*")]

def _terms(env):
    return env["layout"]["components"] + env["layout"]["topics"]

def _n(rng):
    return rng.choice([1, 2, 3, 5])

def _c(env, role):
    return env["layout"]["columns"][role]

# ---- sources: (make returns the parameters, or None when the environment offers nothing suitable) ----
SOURCES = [
    _op("where-am-i", None, "scalar", ["pwd"], lambda e, r: {}, lambda p: "pwd",
        lambda p: ("show the current directory", "cwd", "where are we"), alt=lambda p: "$env.PWD"),
    _op("list-folder", None, "files", ["ls"],
        lambda e, r: _pick(r, [{"dir": "", "label": "this folder"}] + [{"dir": d, "label": d} for d, _ in _folders(e)[:5]]),
        lambda p: f"ls {p['dir']}".strip(),
        lambda p: (f"list what is in {p['label']}", f"ls {p['label']}", f"what's in {p['label']}")),
    _op("list-tree", None, "files", ["ls"],
        lambda e, r: (lambda f: {"dir": f[0], "g": f[1]})(_pick(r, _folders(e))),
        lambda p: f"ls {p['g']}",
        lambda p: (f"list everything under {p['dir']}, at any depth", f"all of {p['dir']}, recursive", f"everything inside {p['dir']}, subfolders too")),
    _op("find-paths", None, "paths", ["glob", "each", "path relative-to"],
        lambda e, r: _pick(r, [{"g": f"**/*.{x}", "label": f".{x} files anywhere"} for x in (e["layout"]["ext"], "log", "md", "toml")]
                           + [{"g": f"{d}/**/*", "label": f"paths under {d}"} for d, _ in _folders(e)[:5]]),
        lambda p: f"glob {p['g']} | {_REL}",
        lambda p: (f"find the {p['label']}", f"glob {p['label']}", f"track down the {p['label']}"),
        alt=lambda p: f"ls {p['g']} | get name"),
    _op("read-file", None, "text", ["open"],
        lambda e, r: {"path": _pick(r, _files_of(e, "text"))},
        lambda p: f"open --raw {p['path']}",
        lambda p: (f"show the contents of {p['path']}", f"cat {p['path']}", f"what does {p['path']} say")),
    _op("read-log", None, "loglines", ["open", "lines"],
        lambda e, r: {"path": _pick(r, _files_of(e, "log"))},
        lambda p: f"open --raw {p['path']} | lines",
        lambda p: (f"read the lines of {p['path']}", f"lines of {p['path']}", f"go through {p['path']} line by line")),
    _op("read-table", None, "table", ["open"],
        lambda e, r: {"path": f"{e['layout']['dirs']['cfg']}/{e['layout']['table']}"},
        lambda p: f"open {p['path']}",
        lambda p: (f"load {p['path']}", f"open {p['path']}", f"take the rows in {p['path']}")),
    _op("read-config", None, "config", ["open"],
        lambda e, r: {"path": f"{e['layout']['dirs']['cfg']}/{e['layout']['config']}"},
        lambda p: f"open {p['path']}",
        lambda p: (f"load {p['path']}", f"open {p['path']}", f"look in {p['path']}")),
    _op("try-open", None, "result", ["try", "open", "glob"],
        lambda e, r: (lambda d, x: _pick(r, [
            {"path": f"{d['cfg']}/defaults.toml", "g": f"{d['cfg']}/*"}, {"path": f"{d['logs']}/server.log", "g": f"{d['logs']}/*"},
            {"path": f"{d['docs']}/setup.md", "g": f"{d['docs']}/*"}, {"path": f"{d['src']}/main.{x}", "g": f"{d['src']}/**/*.{x}"}]))(e["layout"]["dirs"], e["layout"]["ext"]),
        lambda p: f"try {{ open {p['path']} }} catch {{ {{status: 'needs_user_choice', requested: '{p['path']}', candidates: (glob {p['g']} | {_REL} | sort)}} }}",
        lambda p: (f"open {p['path']}; if it is missing, return status needs_user_choice with the requested path and the candidates matching {p['g']}",
                   f"open {p['path']} or list {p['g']} as candidates (status, requested, candidates)",
                   f"read {p['path']} — not sure that's the right name, so if it's missing show me what's in {p['g']} to pick from (status, requested, candidates)"),
        alt=lambda p: f"if ('{p['path']}' | path exists) {{ open {p['path']} }} else {{ {{status: 'needs_user_choice', requested: '{p['path']}', candidates: (glob {p['g']} | {_REL} | sort)}} }}"),
]

# ---- stages ----
def _first_last(src):
    return [
        _op(f"first-{src}", src, src, ["first"], lambda e, r: {"n": _n(r)}, lambda p: f"first {p['n']}",
            lambda p: (f"keep the first {p['n']}", f"first {p['n']}", f"just the first {p['n']}"),
            alt=lambda p: f"take {p['n']}", wrong=lambda p: [(f"last {p['n']}", "Takes the last rows instead of the first.")]),
        _op(f"last-{src}", src, src, ["last"], lambda e, r: {"n": _n(r)}, lambda p: f"last {p['n']}",
            lambda p: (f"keep the last {p['n']}", f"last {p['n']}", f"only the final {p['n']}"),
            wrong=lambda p: [(f"first {p['n']}", "Takes the first rows instead of the last.")]),
        _op(f"count-{src}", src, "scalar", ["length"], lambda e, r: {}, lambda p: "length",
            lambda p: ("count them", "count", "how many is that"),
            wrong=lambda p: [("first", "Returns the first item instead of the count.")]),
    ]

STAGES = [
    # files (rows of name, type, size)
    _op("only-files", "files", "files", ["where"], lambda e, r: {}, lambda p: "where type == file",
        lambda p: ("keep only the files", "files only", "skip the folders"),
        wrong=lambda p: [("where type == dir", "Keeps the folders instead of the files.")]),
    _op("largest-first", "files", "files", ["sort-by"], lambda e, r: {}, lambda p: "sort-by name | sort-by size --reverse",
        lambda p: ("sort by size, largest first", "by size desc", "biggest on top"),
        wrong=lambda p: [("sort-by name | sort-by size", "Sorts ascending; the request wanted the largest first.")]),
    _op("by-name", "files", "files", ["sort-by"], lambda e, r: {}, lambda p: "sort-by name",
        lambda p: ("sort by name", "by name", "in alphabetical order"),
        wrong=lambda p: [("sort-by name --reverse", "Sorts in reverse order.")]),
    _op("name-has", "files", "files", ["where"], lambda e, r: {"term": _pick(r, _terms(e) + [e["layout"]["ext"], "log"])},
        lambda p: f"where name =~ '{p['term']}'",
        lambda p: (f"keep those whose name contains '{p['term']}'", f"name ~ {p['term']}", f"only the ones with '{p['term']}' in the name"),
        wrong=lambda p: [(f"where name !~ '{p['term']}'", "Keeps the names that do not match.")]),
    _op("name-size", "files", "rows", ["select"], lambda e, r: {}, lambda p: "select name size",
        lambda p: ("return name and size", "name+size", "just the name and size"), fields=lambda p: ["name", "size"],
        alt=lambda p: "reject type modified", wrong=lambda p: [("select name type", "Returns the type column instead of the size.")]),
    _op("names", "files", "paths", ["get"], lambda e, r: {}, lambda p: "get name",
        lambda p: ("return their names", "names", "just the names"),
        wrong=lambda p: [("get size", "Returns sizes instead of names.")]),
    _op("total-size", "files", "scalar", ["get", "math"], lambda e, r: {}, lambda p: "get size | math sum",
        lambda p: ("add up their sizes", "total size", "how big is that altogether"),
        wrong=lambda p: [("get size | math max", "Returns the largest size instead of the total."), ("length", "Counts the entries instead of adding their sizes.")]),
    _op("extension-counts", "files", "rows", ["get", "path parse", "uniq", "sort-by"], lambda e, r: {},
        lambda p: "get name | path parse | get extension | uniq -c | sort-by value | sort-by count --reverse",
        lambda p: ("count them by file extension, most common first, as value and count", "ext counts (value, count)", "tally them by extension, biggest group first (value, count)"),
        fields=lambda p: ["value", "count"],
        wrong=lambda p: [("get name | path parse | get extension | uniq | sort", "Lists the extensions without counting them.")]),
    # paths (list of relative paths)
    _op("base-names", "paths", "list", ["path basename"], lambda e, r: {}, lambda p: "path basename",
        lambda p: ("reduce each to its file name", "basenames", "drop the folder part"),
        wrong=lambda p: [("path dirname", "Returns directories instead of file names.")]),
    _op("parent-folders", "paths", "list", ["path dirname", "uniq"], lambda e, r: {}, lambda p: "path dirname | uniq | sort",
        lambda p: ("list the distinct folders they are in, sorted", "unique dirs, sorted", "which folders are those in"),
        wrong=lambda p: [("path basename | uniq | sort", "Returns file names instead of folders."), ("path dirname | sort", "Keeps duplicates.")]),
    _op("path-has", "paths", "paths", ["where"], lambda e, r: {"term": _pick(r, _terms(e))},
        lambda p: f"where ($it =~ '(?i){p['term']}')",
        lambda p: (f"keep the paths mentioning '{p['term']}', any case", f"path ~ {p['term']} (any case)", f"only the ones with {p['term']} somewhere in the path"),
        wrong=lambda p: [(f"where ($it !~ '(?i){p['term']}')", "Keeps the paths that do not match.")]),
    _op("sorted", "paths", "paths", ["sort"], lambda e, r: {}, lambda p: "sort",
        lambda p: ("sort them", "sorted", "in order"), wrong=lambda p: [("sort --reverse", "Sorts in reverse order.")]),
    _op("hit-counts", "paths", "rows", ["par-each", "open", "lines", "where", "str contains", "sort-by"], lambda e, r: {"term": _pick(r, _terms(e))},
        lambda p: f"where ($it | path type) == file | par-each {{|f| {{file: $f, hits: (open --raw $f | lines | where ($it | str contains -i '{p['term']}') | length)}} }} | where hits > 0 | sort-by file | sort-by hits --reverse",
        lambda p: (f"count the lines mentioning '{p['term']}' in each file and rank the files with any, most hits first, as file and hits", f"grep -c {p['term']} per file, >0, by hits desc (file, hits)", f"which of them talk about {p['term']} the most? file + number of matching lines"),
        fields=lambda p: ["file", "hits"],
        alt=lambda p: f"where ($it | path type) == file | each {{|f| {{file: $f, hits: (open --raw $f | lines | find -n -i '{p['term']}' | length)}} }} | where hits > 0 | sort-by file | sort-by hits --reverse",
        wrong=lambda p: [(f"where ($it | path type) == file | par-each {{|f| {{file: $f, hits: (open --raw $f | lines | where ($it | str contains -i '{p['term']}') | length)}} }} | where hits == 0 | sort-by file", "Keeps the files without a match.")]),
    _op("matching-lines", "paths", "rows", ["each", "open", "lines", "where", "str contains", "str trim", "flatten"], lambda e, r: {"term": _pick(r, _terms(e))},
        lambda p: f"where ($it | path type) == file | sort | each {{|f| open --raw $f | lines | where ($it | str contains -i '{p['term']}') | each {{|l| {{file: $f, line: ($l | str trim)}} }} }} | flatten",
        lambda p: (f"show every line mentioning '{p['term']}' with its file, as file and line", f"grep {p['term']} (file, line)", f"pull out the lines that mention {p['term']}, with the file each is from"),
        fields=lambda p: ["file", "line"],
        alt=lambda p: f"where ($it | path type) == file | sort | each {{|f| open --raw $f | lines | find -n -i '{p['term']}' | each {{|l| {{file: $f, line: ($l | str trim)}} }} }} | flatten",
        wrong=lambda p: [(f"where ($it | path type) == file | sort | each {{|f| open --raw $f | lines | where ($it | str starts-with -i '{p['term']}') | each {{|l| {{file: $f, line: ($l | str trim)}} }} }} | flatten", "Matches only at the start of a line.")]),
    _op("resolve", "paths", "result", ["is-empty", "length"], lambda e, r: {}, 
        lambda p: "if ($matches | is-empty) { {status: 'not_found'} } else if (($matches | length) > 1) { {status: 'choose_one', matches: $matches} } else { {status: 'resolved', file: ($matches | first)} }",
        lambda p: ("report status not_found, resolved (with file) or choose_one (with matches)", "→ not_found / resolved+file / choose_one+matches", "if there's exactly one use it, if several let me choose, if none say so (status, file or matches)"),
        alt=lambda p: "match ($matches | length) { 0 => {status: 'not_found'}, 1 => {status: 'resolved', file: $matches.0}, _ => {status: 'choose_one', matches: $matches} }",
        wrong=lambda p: [("if ($matches | is-not-empty) { {status: 'not_found'} } else if (($matches | length) > 1) { {status: 'choose_one', matches: $matches} } else { {status: 'resolved', file: ($matches | first)} }", "The emptiness test is inverted.")],
        binds="matches"),
    # text and lines
    _op("split-lines", "text", "lines", ["lines"], lambda e, r: {}, lambda p: "lines",
        lambda p: ("split it into lines", "lines", "line by line")),
    *[_op(f"containing-{t}", t, t, ["where", "str contains"], lambda e, r, t=t: {"term": _pick(r, (["ERROR", "WARN", "INFO"] if t == "loglines" else []) + _terms(e))},
          lambda p: f"where ($it | str contains '{p['term']}')",
          lambda p: (f"keep the lines containing '{p['term']}'", f"grep {p['term']}", f"only the lines that mention {p['term']}"),
          alt=lambda p: f"find -n '{p['term']}'",
          wrong=lambda p: [(f"where not ($it | str contains '{p['term']}')", "Keeps the lines that do not contain it."), (f"where ($it | str ends-with '{p['term']}')", "Matches only at the end of a line.")])
      for t in ("lines", "loglines")],
    *[_op(f"lowercase-{t}", t, "lines", ["str downcase", "str trim"], lambda e, r: {}, lambda p: "str downcase | str trim",
          lambda p: ("lowercase and trim each line", "lower+trim", "all lower case, no stray spaces"),
          wrong=lambda p: [("str upcase | str trim", "Upper-cases instead of lower-casing.")]) for t in ("lines", "loglines")],
    _op("join-lines", "lines", "scalar", ["str join"], lambda e, r: {"sep": _pick(r, [" | ", "; ", ", "])}, lambda p: f"str join '{p['sep']}'",
        lambda p: (f"join them with '{p['sep']}'", f"join '{p['sep']}'", f"glue them together with '{p['sep']}' between"),
        wrong=lambda p: [("str join", "Joins without the requested separator.")]),
    _op("parse-log", "loglines", "logrows", ["parse"], lambda e, r: {}, lambda p: f"parse -r {_LOG_RX}",
        lambda p: ("parse each line into ts, level and msg", "parse → ts, level, msg", "split each entry into time, level and message (ts, level, msg)"), fields=lambda p: ["ts", "level", "msg"]),
    _op("level-is", "logrows", "logrows", ["where"], lambda e, r: {"lvl": _pick(r, ["ERROR", "WARN", "INFO"])}, lambda p: f"where level == '{p['lvl']}'",
        lambda p: (f"keep the {p['lvl']} rows", f"level={p['lvl']}", f"only the {p['lvl']} ones"),
        wrong=lambda p: [(f"where level != '{p['lvl']}'", "Keeps every level except the one asked for.")]),
    _op("when-and-what", "logrows", "rows", ["select"], lambda e, r: {}, lambda p: "select ts msg",
        lambda p: ("return ts and msg", "ts+msg", "when and what (ts, msg)"), fields=lambda p: ["ts", "msg"],
        alt=lambda p: "reject level", wrong=lambda p: [("select ts level", "Returns the level column instead of the message.")]),
    _op("messages", "logrows", "list", ["get"], lambda e, r: {}, lambda p: "get msg",
        lambda p: ("return the message text", "msgs", "just the messages"), wrong=lambda p: [("get ts", "Returns timestamps instead of messages.")]),
    _op("level-counts", "logrows", "rows", ["get", "uniq", "sort-by"], lambda e, r: {}, lambda p: "get level | uniq -c | sort-by value",
        lambda p: ("count the rows per level, as value and count, sorted by level", "level histogram (value, count)", "how many of each level (value, count)"), fields=lambda p: ["value", "count"],
        alt=lambda p: "group-by level | transpose value rows | each {|g| {value: $g.value, count: ($g.rows | length)} } | sort-by value",
        wrong=lambda p: [("get level | uniq | sort", "Lists the levels without counting them.")]),
    # table (rows from the JSON file)
    _op("owned-by", "table", "table", ["where"], lambda e, r: {"col": _c(e, "owner"), "who": _pick(r, e["layout"]["owners"])},
        lambda p: f"where {p['col']} == '{p['who']}'",
        lambda p: (f"keep the rows where {p['col']} is {p['who']}", f"{p['col']}={p['who']}", f"only {p['who']}'s rows"),
        wrong=lambda p: [(f"where {p['col']} != '{p['who']}'", "Keeps everything except what was asked for.")]),
    _op("at-least", "table", "table", ["where"], lambda e, r: {"col": _c(e, "number"), "t": _pick(r, [20, 40, 50, 60, 80])},
        lambda p: f"where {p['col']} >= {p['t']}",
        lambda p: (f"keep the rows with {p['col']} of at least {p['t']}", f"{p['col']}>={p['t']}", f"only where {p['col']} is {p['t']} or more"),
        wrong=lambda p: [(f"where {p['col']} < {p['t']}", "The comparison is inverted."), (f"where {p['col']} > {p['t']}", "Excludes rows exactly at the threshold.")]),
    _op("switched-on", "table", "table", ["where"], lambda e, r: {"col": _c(e, "flag"), "on": _pick(r, [True, True, False])},
        lambda p: f"where {p['col']} == {str(p['on']).lower()}",
        lambda p: (f"keep the rows where {p['col']} is {str(p['on']).lower()}", f"{p['col']}={str(p['on']).lower()}", f"only the ones that {'are' if p['on'] else 'are not'} {p['col']}"),
        alt=lambda p: f"where {'' if p['on'] else 'not '}{p['col']}",
        wrong=lambda p: [(f"where {p['col']} == {str(not p['on']).lower()}", "Keeps the opposite rows.")]),
    _op("highest-first", "table", "table", ["sort-by"], lambda e, r: {"key": _c(e, "key"), "col": _c(e, "number")},
        lambda p: f"sort-by {p['key']} | sort-by {p['col']} --reverse",
        lambda p: (f"sort by {p['col']}, highest first", f"by {p['col']} desc", f"highest {p['col']} on top"),
        wrong=lambda p: [(f"sort-by {p['key']} | sort-by {p['col']}", "Sorts ascending; the request wanted the highest first.")]),
    _op("pick-columns", "table", "rows", ["select"], lambda e, r: {"cols": [_c(e, "key")] + sorted(r.sample([_c(e, "owner"), _c(e, "number"), _c(e, "flag")], 2))},
        lambda p: "select " + " ".join(p["cols"]),
        lambda p: ("return " + ", ".join(p["cols"]), "+".join(p["cols"]), "just " + ", ".join(p["cols"])), fields=lambda p: p["cols"],
        wrong=lambda p: [("select " + " ".join(p["cols"][:-1]), f"Drops the {p['cols'][-1]} column.")]),
    _op("column-values", "table", "numbers", ["get"], lambda e, r: {"col": _c(e, "number")}, lambda p: f"get {p['col']}",
        lambda p: (f"take the {p['col']} values", p["col"], f"the {p['col']} numbers")),
    _op("count-per-owner", "table", "rows", ["get", "uniq", "sort-by"], lambda e, r: {"col": _c(e, "owner")},
        lambda p: f"get {p['col']} | uniq -c | rename {p['col']} count | sort-by {p['col']}",
        lambda p: (f"count the rows per {p['col']}, as {p['col']} and count, sorted by {p['col']}", f"count by {p['col']}", f"how many each {p['col']} has ({p['col']}, count)"), fields=lambda p: [p["col"], "count"],
        alt=lambda p: f"group-by {p['col']} | transpose {p['col']} items | insert count {{|r| $r.items | length}} | select {p['col']} count | sort-by {p['col']}",
        wrong=lambda p: [(f"get {p['col']} | uniq | sort", "Lists the values without counting them.")]),
    _op("sum-per-owner", "table", "rows", ["group-by", "transpose", "each", "math", "sort-by"], lambda e, r: {"col": _c(e, "owner"), "num": _c(e, "number")},
        lambda p: f"group-by {p['col']} | transpose {p['col']} rows | each {{|g| {{{p['col']}: $g.{p['col']}, total: ($g.rows.{p['num']} | math sum)}} }} | sort-by {p['col']}",
        lambda p: (f"sum {p['num']} per {p['col']}, as {p['col']} and total, sorted by {p['col']}", f"sum {p['num']} by {p['col']}", f"add up each {p['col']}'s {p['num']} ({p['col']}, total)"), fields=lambda p: [p["col"], "total"],
        wrong=lambda p: [(f"group-by {p['col']} | transpose {p['col']} rows | each {{|g| {{{p['col']}: $g.{p['col']}, total: ($g.rows | length)}} }} | sort-by {p['col']}", "Counts the rows instead of summing them.")]),
    _op("tag-reviewer", "table", "table", ["upsert"], lambda e, r: {"who": _pick(r, e["layout"]["owners"])}, lambda p: f"upsert reviewer '{p['who']}'",
        lambda p: (f"add a reviewer column set to '{p['who']}'", f"reviewer={p['who']}", f"mark {p['who']} as the reviewer on each"),
        alt=lambda p: f"insert reviewer '{p['who']}'"),
    *[_op(f"math-{f}", "numbers", "scalar", ["math"], lambda e, r: {}, lambda p, f=f: f"math {f}",
          lambda p, w=w: (f"take the {w}", w, f"what's the {w}"),
          wrong=lambda p, g=g, w=w: [(f"math {g}", f"Does not compute the {w}.")])
      for f, w, g in (("sum", "sum", "avg"), ("avg", "average", "sum"), ("max", "maximum", "min"), ("min", "minimum", "max"))],
    # config (record of sections) and record (one section)
    _op("section", "config", "record", ["get"], lambda e, r: {"sec": _pick(r, [e["layout"]["service_section"], e["layout"]["store_section"]])},
        lambda p: f"get {p['sec']}", lambda p: (f"take the {p['sec']} section", p["sec"], f"the {p['sec']} part")),
    _op("setting", "config", "scalar", ["get"],
        lambda e, r: _pick(r, [{"sec": e["layout"]["service_section"], "key": k} for k in ("name", "port", "debug")] + [{"sec": e["layout"]["store_section"], "key": k} for k in ("host", "port", "timeout_ms", "max_connections")]),
        lambda p: f"get {p['sec']}.{p['key']}",
        lambda p: (f"read {p['key']} from the {p['sec']} section", f"{p['sec']}.{p['key']}", f"what's the {p['sec']} {p['key']}"),
        wrong=lambda p: [(f"get {p['sec']}", "Returns the whole section instead of the one setting.")]),
    _op("key", "record", "scalar", ["get"], lambda e, r: {"key": _pick(r, ["host", "port", "name", "timeout_ms", "max_connections"])}, lambda p: f"get -o {p['key']}",
        lambda p: (f"read its {p['key']}, or nothing if it has none", f".{p['key']}?", f"its {p['key']}, if there is one")),
    _op("key-or-default", "record", "scalar", ["get", "default"], lambda e, r: (lambda k: {"key": k, "d": OPTIONAL_KEYS[k]})(_pick(r, sorted(OPTIONAL_KEYS))),
        lambda p: f"get -o {p['key']} | default {p['d']}",
        lambda p: (f"read its {p['key']}, falling back to {p['d']} when it is not set", f"{p['key']} default {p['d']}", f"the {p['key']} — may not be there, use {p['d']} then"),
        alt=lambda p: f"get {p['key']}? | default {p['d']}",
        wrong=lambda p: [(f"get -o {p['key']} | default (-1)", "Falls back to -1 instead of the requested default.")]),
    _op("keys", "record", "list", ["columns", "sort"], lambda e, r: {}, lambda p: "columns | sort",
        lambda p: ("list its keys, sorted", "keys", "which settings does it have"),
        alt=lambda p: "transpose key value | get key | sort", wrong=lambda p: [("values", "Lists the values instead of the keys.")]),
    *[s for t in ("files", "paths", "lines", "loglines", "logrows", "table", "rows", "list") for s in _first_last(t)],
]
_JSON_TYPES = {"files", "paths", "rows", "table", "logrows", "record", "result", "list", "lines", "loglines", "config", "numbers", "value"}

# ---- every other command, from Nu's signatures ----
# Each kind of data the recipes produce is one of Nu's types; a command fits after a step when one of its
# declared input types accepts that type, and its declared output type says what kind comes next.
KIND_TYPE = {
    "files": "table", "rows": "table", "table": "table", "logrows": "table", "paths": "list<string>", "lines": "list<string>",
    "loglines": "list<string>", "list": "list<string>", "numbers": "list<number>", "text": "string", "record": "record", "config": "record",
}
_ACCEPTS = {
    "table": ("table", "list<any>", "any"), "list<string>": ("list<string>", "list<any>", "any"),
    "list<number>": ("list<number>", "list<any>", "any"), "string": ("string", "any"), "record": ("record", "any"),
}
_PRODUCES = {
    "table": "rows", "list<string>": "list", "list<number>": "numbers", "list<int>": "numbers", "list<float>": "numbers",
    "string": "text", "record": "record", "int": "scalar", "float": "scalar", "number": "scalar", "bool": "scalar", "filesize": "scalar",
}
_AUTO_CATEGORIES = {"filters", "strings", "math", "path", "conversions", "hash"}
_SKIPPED = {"print": "writes to the terminal", "shuffle": "random order", "path self": "only valid in scripts", "path expand": "depends on where the copy lives"}
_recipe_commands = {c for o in SOURCES + STAGES for c in o["commands"]}
_names = {c["name"] for c in NU_COMMANDS}

def _automatic_stages() -> list[dict]:
    stages = []
    for command in NU_COMMANDS:
        name = command["name"]
        is_namespace = any(other.startswith(name + " ") for other in _names)  # `str`, `math`, `path`: help text only
        if (command["category"] not in _AUTO_CATEGORIES or name in _recipe_commands or name in _SKIPPED or is_namespace
                or any(not p["name"].startswith("--") for p in command["params"])):  # takes an argument: needs a recipe
            continue
        said = command["description"].rstrip(".")
        said = said[0].lower() + said[1:]
        for nu_type, accepted in _ACCEPTS.items():
            match = next((io for want in accepted for io in command["input_output"] if io["input"] == want), None)
            if match is None:
                continue
            produced = _PRODUCES.get(match["output"], "value")
            if match["output"] in ("any", "list<any>") and match["input"] in ("any", "list<any>"):
                produced = {"table": "rows", "list<string>": "list", "list<number>": "numbers", "string": "text", "record": "record"}[nu_type]
            stages.append(
                _op(f"{name} ({nu_type})", nu_type, produced, [name], lambda e, r: {}, lambda p, n=name: n,
                    lambda p, s=said, n=name: (s, n, s), auto=True)
            )
    return stages

AUTO_STAGES = _automatic_stages()
STAGES = STAGES + AUTO_STAGES

import networkx as nx

KIND_GRAPH = nx.MultiDiGraph()  # nodes: kinds of data; edges: the operators that turn one into another
for _o in STAGES:
    for _kind in ([k for k, t in KIND_TYPE.items() if t == _o["src"]] if _o["auto"] else [_o["src"]]):
        KIND_GRAPH.add_edge(_kind, _o["dst"], key=_o["name"], op=_o)
for _o in SOURCES:
    KIND_GRAPH.add_edge("start", _o["dst"], key=_o["name"], op=_o)

def _weight(op: dict, prior: dict) -> float:
    """An operator's sampling weight under a prior: the mean weight of its commands, relative to the average
    command, with a small floor so that commands the prior never saw can still appear occasionally."""
    if not prior:
        return 1.0
    mean = sum(prior.values()) / len(prior)
    seen = sum(prior.get(c, 0.0) for c in op["commands"]) / len(op["commands"]) / mean
    # a recipe keeps a small floor; an automatically added command appears only if the prior has seen it
    return seen if op["auto"] else max(seen, 0.02)

def _choose(rng, ops, prior):
    weights = [_weight(o, prior) for o in ops]
    return rng.choices(ops, weights=weights)[0] if any(weights) else None

def length_weights(origin: str, bias: float) -> list[float]:
    """How likely a chain of 1..6 steps is. At bias 0 this is the measured distribution: pipeline lengths in
    Nushell's documentation for native chains, in real command lines for ported ones. At bias 1 every length
    is equally likely. In between the two are mixed."""
    measured = PRIOR_INFO["lengths"].get(origin) or {1: 0.4, 2: 0.35, 3: 0.15, 4: 0.07, 5: 0.03}
    total = sum(measured.get(n, 0.0) for n in range(1, 7))
    return [(1 - bias) * measured.get(n, 0.0) / total + bias / 6 for n in range(1, 7)]

def sample_chain(env: dict, rng, prior: dict, lengths: list[float]):
    """One chain: a source, then stages whose input type matches what the previous step produced, up to a
    length drawn from `lengths` (or until nothing fits)."""
    target = rng.choices(range(1, 7), weights=lengths)[0]
    source = _choose(rng, [d["op"] for _, _, d in KIND_GRAPH.out_edges("start", data=True)], prior)
    params = source["make"](env, rng)
    if params is None or None in params.values():
        return None
    steps, kind = [(source, params)], source["dst"]
    while len(steps) < max(target, 2 if kind == "config" else 1):
        used = {s[0]["name"] for s in steps}
        options = [d["op"] for _, _, d in KIND_GRAPH.out_edges(kind, data=True) if d["op"]["name"] not in used] if kind in KIND_GRAPH else []
        if not options:
            break
        op = _choose(rng, options, prior)
        if op is None:
            break
        params = op["make"](env, rng)
        if params is None or None in params.values():
            break
        steps.append((op, params))
        kind = op["dst"]
    as_json = kind in _JSON_TYPES and (kind in ("rows", "result", "record", "config") or rng.random() < 0.6)
    return {"steps": steps, "json": as_json, "kind": kind}

def render(spec: dict, alt: bool = False, replace: dict | None = None) -> tuple[str, list[str]]:
    """Nushell source for a chain and the text of each step. `replace` maps a step index to other text."""
    parts = []
    for i, (op, params) in enumerate(spec["steps"]):
        text = (replace or {}).get(i)
        if text is None:
            text = op["alt"](params) if alt and op["alt"] else op["code"](params)
        parts.append(text)
    tail = " | to json --raw" if spec["json"] else ""
    bound = next((i for i, (op, _) in enumerate(spec["steps"]) if op["binds"]), None)
    if bound is not None:  # a step that reads the pipeline so far through a variable
        name = spec["steps"][bound][0]["binds"]
        rest = "".join(f" | {p}" for p in parts[bound + 1 :])
        return f"let {name} = ({' | '.join(parts[:bound])} | sort); let result = ({parts[bound]}); $result{rest}{tail}", parts
    if len(parts) >= 4:  # longer chains bind the first half, the way people write them
        cut = len(parts) // 2
        return f"let found = ({' | '.join(parts[:cut])}); $found | {' | '.join(parts[cut:])}{tail}", parts
    return " | ".join(parts) + tail, parts

def describe(spec: dict) -> list[str]:
    """The request in three registers, composed from what each step says about itself."""
    said = [op["say"](params) for op, params in spec["steps"]]
    fields = next((op["fields"](params) for op, params in reversed(spec["steps"]) if op["fields"]), None)
    ending = ("as JSON" + (f" with {', '.join(fields)}" if fields and spec["kind"] == "rows" else "")) if spec["json"] else ""
    plain = "; then ".join(s[0] for s in said)
    plain = plain[0].upper() + plain[1:] + (f". Return it {ending}." if ending else ".")
    short = ", ".join(s[1] for s in said) + (", json" if ending else "")
    casual = " and ".join(s[2] for s in said) + (" — json please" if ending else "") + "?"
    return [plain, short, casual]

def categorize(spec: dict) -> str:
    """Which catalog category a chain belongs to, from the operators it uses."""
    names = [op["name"] for op, _ in spec["steps"]]
    first = spec["steps"][0][0]["dst"]
    if names[0] == "try-open" or "resolve" in names or "key-or-default" in names:
        return "pipelines_recovery"
    if any(n in ("hit-counts", "matching-lines", "path-has", "name-has") or n.startswith("containing-") for n in names) or names[0] == "find-paths":
        return "search"
    if first in ("table", "config"):
        return "tables_records"
    if first in ("text", "loglines"):
        return "text_manipulation"
    return "filesystem"

# What each common Unix command is for, in this dataset's categories. Only used to turn measured usage
# into category shares; commands that are not about files or text (git, editors, sudo, package managers)
# are left out.
UNIX_CATEGORY = {
    "search": ["find", "grep", "egrep", "fgrep", "rg", "ag", "ack", "locate", "which", "whereis"],
    "text_manipulation": ["cat", "sed", "awk", "cut", "tr", "sort", "uniq", "wc", "head", "tail", "echo", "less", "more", "tac", "rev", "paste", "printf", "nl", "diff", "strings", "fold"],
    "filesystem": ["ls", "ll", "la", "l", "cd", "pwd", "du", "df", "stat", "file", "tree", "basename", "dirname", "readlink", "realpath", "mkdir", "rm", "mv", "cp", "touch", "ln", "chmod", "chown"],
    "tables_records": ["jq", "column", "join", "bc", "expr", "seq", "yq", "csvcut", "sqlite3"],
    "pipelines_recovery": ["xargs", "test", "[", "read", "tee", "parallel", "true", "false", "timeout", "watch"],
}

def category_shares(categories: list[str], floor: float = 0.12) -> dict:
    """How the dataset is divided among the categories: each category's share of measured Unix usage
    (data/command_usage.json, grouped by UNIX_CATEGORY), so that what people do most in a shell gets the
    most data. A category is never given less than `floor`: structured data and recovery are what Nushell
    adds, and Unix usage barely registers them."""
    usage_file = ROOT / "data" / "command_usage.json"
    if not usage_file.exists():
        return {c: 1 / len(categories) for c in categories}
    combined = json.loads(usage_file.read_text(encoding="utf-8"))["combined"]
    mass = {c: sum(combined.get(u, 0.0) for u in UNIX_CATEGORY[c]) for c in categories}
    total = sum(mass.values()) or 1
    raised = {c: max(mass[c] / total, floor) for c in categories}
    return {c: v / sum(raised.values()) for c, v in raised.items()}

_PROBE = {}

def _chain(env: dict, spec: dict, index: int) -> dict:
    code, parts = render(spec)
    alt = render(spec, alt=True)[0]
    commands = list(dict.fromkeys(c for op, _ in spec["steps"] for c in op["commands"])) + (["to json"] if spec["json"] else [])
    asks = describe(spec)
    return {
        "chain_id": f"{env['id']}/{index:03d}", "env": env["id"], "layout": env["layout"], "files": env["files"], "spec": spec, "parts": parts,
        "family": "+".join(op["name"] for op, _ in spec["steps"]), "category": categorize(spec), "commands": commands,
        "intent": asks[0], "asks": asks, "code": code, "alt_code": alt,
    }

def build_chains(environments: list[dict], categories: list[str], total: int, ported_share: float = 0.25, length_bias: float = 0.5, seed: int = 73) -> tuple[list[dict], list[dict], dict]:
    """Sample chains, keep those that survive execution, and fill each category's quota.

    Quotas come from `category_shares`. Within a category at most `ported_share` (never more than
    PORTED_LIMIT) of the chains are sampled under the translated Unix prior; the rest are sampled under the
    Nushell-native prior. Chains are spread evenly over the environments."""
    ported_share = min(ported_share, PORTED_LIMIT)
    shares = category_shares(categories)
    quota = {c: round(total * shares[c]) for c in categories}
    candidates = []
    for env in environments:
        rng, seen = random.Random(f"chains-{seed}-{env['id']}"), set()
        for origin, prior, share in (("native", NATIVE, 1 - ported_share), ("ported", PORTED, ported_share)):
            wanted = {c: -(-quota[c] * share * 2.5 // len(environments)) for c in categories}  # oversample; some will fail
            got, tries = {c: 0 for c in categories}, 0
            while any(got[c] < wanted[c] for c in categories) and tries < 4000:
                tries += 1
                spec = sample_chain(env, rng, prior, length_weights(origin, length_bias))
                if spec is None:
                    continue
                chain = {**_chain(env, spec, len(seen) + 1), "origin": origin}
                category = chain["category"]
                if chain["code"] in seen or category not in wanted or got[category] >= wanted[category]:
                    continue
                seen.add(chain["code"])
                got[category] += 1
                candidates.append(chain)
    # an automatic stage must change the result, or it is just noise in the chain
    def without_automatic(chain):
        """The chain with each automatically added stage left out in turn."""
        steps = chain["spec"]["steps"]
        return [render({**chain["spec"], "steps": steps[:i] + steps[i + 1 :]})[0] for i, (o, _) in enumerate(steps) if o["auto"]]

    shorter = [without_automatic(c) for c in candidates]
    _jobs = [(code, c["env"]) for codes, c in zip(shorter, candidates) for code in codes]
    shorter_runs = dict(zip([j[0] for j in _jobs], run_nu_many(_jobs)))
    runs = run_nu_many([(c["code"], c["env"]) for c in candidates])
    variants = run_nu_many([(c["code"], c["env"], True) for c in candidates])
    alts = run_nu_many([(c["alt_code"], c["env"]) for c in candidates])
    # par-each returns in any order, so run each chain a second time to catch an unstable result
    repeats = run_nu_many([(c["code"] + " ", c["env"]) for c in candidates])
    verified, dropped = [], []
    for chain, base, variant, alt, again, short in zip(candidates, runs, variants, alts, repeats, shorter):
        reason = None
        if any(shorter_runs[code]["ok"] and shorter_runs[code]["stdout"] == base["stdout"] for code in short):
            reason = "an automatic stage changes nothing"
        elif not base["ok"]:
            reason = "does not run: " + base["stderr"][:120]
        elif base["stdout"] in ("", "[]", "{}", "null", '""'):
            reason = "returns nothing"
        elif again["stdout"] != base["stdout"]:
            reason = "output changes between runs"
        elif len(base["stdout"]) > 4000:
            reason = "output too long to be a useful reference"
        if reason:
            dropped.append({"chain_id": chain["chain_id"], "family": chain["family"], "reason": reason})
            continue
        verified.append(
            {
                **chain,
                "expected": base["stdout"],
                "expected_variant": variant["stdout"] if variant["ok"] else None,
                "alt_ok": chain["alt_code"] != chain["code"] and alt["ok"] and same_output(alt["stdout"], base["stdout"]),
            }
        )
    save_nu_cache()
    # fill each category's quota, going round the environments, native chains first and ported ones up to their cap
    kept = []
    for category in categories:
        pools = {origin: {e["id"]: [c for c in verified if c["category"] == category and c["origin"] == origin and c["env"] == e["id"]] for e in environments} for origin in ("native", "ported")}
        limits = {"ported": int(quota[category] * ported_share), "native": quota[category]}
        chosen = []
        for origin in ("ported", "native"):
            room = limits[origin] if origin == "ported" else quota[category] - len(chosen)
            while room > 0 and any(pools[origin].values()):
                for env_id in list(pools[origin]):
                    if room > 0 and pools[origin][env_id]:
                        chosen.append(pools[origin][env_id].pop(0))
                        room -= 1
        kept.extend(chosen)
    summary = {
        "category_shares": {c: round(v, 3) for c, v in shares.items()},
        "quota": quota,
        "ported_share": round(sum(c["origin"] == "ported" for c in kept) / max(1, len(kept)), 3),
    }
    return kept, dropped, summary

def structural_breaks(chain: dict, failure_type: str, index: int) -> list[tuple]:
    """Broken variants made by changing the sampled structure, as (code, span, note). Whether each one
    deserves its label is decided afterwards by running it."""
    spec, parts, out = chain["spec"], chain["parts"], []
    steps = spec["steps"]
    rng = random.Random(f"{chain['chain_id']}-{failure_type}")
    order = list(range(len(steps)))
    rng.shuffle(order)
    if failure_type == "wrong_operation_or_argument":
        for i in order:
            for text, note in steps[i][0]["wrong"](steps[i][1]):
                out.append((render(spec, replace={i: text})[0], text, note))
    elif failure_type == "request_misunderstood":
        env = {"layout": chain["layout"], "files": chain["files"], "id": chain["env"]}
        for i in order:
            op, params = steps[i]
            if not params:
                continue
            for _ in range(6):  # the same step with another value: the wrong name, number or term
                other = op["make"](env, rng)
                if other and None not in other.values() and op["code"](other) != op["code"](params) and (i > 0 or failure_type != "request_misunderstood" or op["dst"] in ("scalar", "result", "config")):
                    text = op["code"](other)
                    out.append((render(spec, replace={i: text})[0], text, f"Uses `{text}` where the request asked for `{parts[i]}`."))
                    break
    elif failure_type == "wrong_data_source":
        env = {"layout": chain["layout"], "files": chain["files"], "id": chain["env"]}
        op, params = steps[0]
        for _ in range(8):  # the same kind of source, pointed somewhere else
            other = op["make"](env, rng)
            if other and None not in other.values() and op["code"](other) != op["code"](params):
                text = op["code"](other)
                out.append((render(spec, replace={0: text})[0], text, f"Reads from `{text}`; the request named `{parts[0]}`."))
        columns, owners = chain["layout"]["columns"], chain["layout"]["owners"]
        if op["name"] == "read-table":
            text = f"[[{columns['key']} {columns['flag']} {columns['owner']} {columns['number']}]; [demo_a true {owners[0]} 10] [demo_b false {owners[1]} 90] [demo_c true {owners[2]} 55]]"
            out.append((render(spec, replace={0: text})[0], text, f"Uses a made-up example table and never opens {params['path']}."))
        if op["name"] == "read-config":
            text = f"{{{chain['layout']['service_section']}: {{name: 'example', port: 8000, debug: true}}, {chain['layout']['store_section']}: {{host: 'localhost', port: 5432, max_connections: 5, timeout_ms: 100}}}}"
            out.append((render(spec, replace={0: text})[0], text, f"Uses a made-up record and never opens {params['path']}."))
    elif failure_type == "broken_chain_or_function":
        for i in order:
            if 0 < i < len(steps) and not steps[i][0]["binds"]:
                shorter = {**spec, "steps": steps[:i] + steps[i + 1 :]}
                code, new_parts = render(shorter)
                span = new_parts[i - 1] if i - 1 < len(new_parts) else new_parts[-1]
                out.append((code, span, f"The `{parts[i][:60]}` step is missing."))
        for i in order:
            if 0 < i < len(steps) - 1 and not steps[i][0]["binds"] and not steps[i + 1][0]["binds"]:
                swapped = {**spec, "steps": steps[:i] + [steps[i + 1], steps[i]] + steps[i + 2 :]}
                code, new_parts = render(swapped)
                out.append((code, new_parts[i], f"`{parts[i + 1][:40]}` runs before `{parts[i][:40]}`; the order is reversed."))
    return out
