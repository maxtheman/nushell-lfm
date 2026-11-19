"""The ten failure types: broken variants of a chain, each kept only if running it agrees with its label."""

from concurrent.futures import ThreadPoolExecutor
import json
import re

from .chains import structural_breaks
from .environments import OPTIONAL_KEYS, run_nu, same_output, save_nu_cache

FAILURE_TYPES = {
    "valid": "The tool call is well formed and does what was asked.",
    "request_misunderstood": "Request misunderstood: a misheard file name, a missing negation, or a wrong number.",
    "wrong_tool_or_malformed_call": "Wrong tool or malformed call: another tool, prose, or unreadable call syntax.",
    "invalid_nushell": "Invalid Nushell: an unclosed delimiter, an invalid function signature, or syntax from another language.",
    "wrong_data_source": "Wrong data source: the wrong file or column, or example data in place of the real file.",
    "wrong_operation_or_argument": "Wrong operation or argument: count instead of sum, wrong sort column or direction, or wrong filter.",
    "broken_chain_or_function": "Broken chain or function: missing or reordered steps, an uncalled function, or a wrong intermediate value.",
    "wrong_output_or_side_effect": "Wrong output or side effect: wrong output shape, extra text, or a write to the wrong path.",
    "runtime_or_environment_failure": "Runtime or environment failure: a timeout, an unsupported feature, a version difference, or a path the project does not have.",
    "grader_false_pass": "Grader false pass: a hard-coded result that matches this one workspace without computing anything.",
    "grader_false_fail": "Grader false fail: an equivalent answer that differs only in formatting, order, type, or an unspecified tie-break.",
}
FAILURE_LABELS = list(FAILURE_TYPES)
BREAK_LABELS = FAILURE_LABELS[1:]
NOT_BROKEN = {"valid", "grader_false_fail"}  # a false fail is a correct answer that a strict comparison would reject
VERIFIED_BY = {
    "request_misunderstood": "runs; output differs from the reference",
    "wrong_tool_or_malformed_call": "the call does not parse as a run_nushell tool call",
    "invalid_nushell": "Nu reports a parse error",
    "wrong_data_source": "runs; output differs from the reference",
    "wrong_operation_or_argument": "runs; output differs from the reference",
    "broken_chain_or_function": "runs; output differs from the reference",
    "wrong_output_or_side_effect": "runs; output differs or a file was written",
    "runtime_or_environment_failure": "parses; then fails or times out",
    "grader_false_pass": "matches the reference here; the reference changes in a second copy of the environment",
    "grader_false_fail": "output is not identical to the reference but is the same answer once formatting, order and type are set aside",
}

def wrap_tool_call(code: str) -> str:
    return f"<function=run_nushell>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>"

_CALL = re.compile(r"\A<function=run_nushell>\n<parameter=code>\n(.*)\n</parameter>\n</function>\n</tool_call>\s*\Z", re.S)

def parse_tool_call(completion: str):
    """The Nushell source of a well-formed run_nushell call, else None."""
    found = _CALL.match(completion)
    return found.group(1) if found else None

def split_top(code: str, sep: str) -> list[str]:
    """Split at `sep` outside quotes and brackets, so pipes inside closures stay put."""
    parts, depth, quote, start = [], 0, None, 0
    for i, ch in enumerate(code):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"`":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(code[start:i])
            start = i + 1
    parts.append(code[start:])
    return parts

def _pipelines(code: str):
    """Each statement that is a pipeline of three or more stages, longest first, with what is needed to rebuild
    it. A `let name = ( ... )` statement is opened so its inner pipeline can be edited too."""
    statements = split_top(code, ";")
    found = []
    for idx, statement in enumerate(statements):
        lead = statement[: len(statement) - len(statement.lstrip())]
        body, prefix, suffix = statement.strip(), "", ""
        bound = re.match(r"(let \w+ = \()(.*)(\))\Z", body, re.S)
        if bound and len(split_top(bound.group(2), ")")) == 1:
            prefix, body, suffix = bound.groups()
        stages = [s.strip() for s in split_top(body, "|")]
        if len(stages) >= 3:
            found.append((idx, lead, prefix, stages, suffix))

    def rebuild(idx, lead, prefix, stages, suffix):
        out = list(statements)
        out[idx] = f"{lead}{prefix}{' | '.join(stages)}{suffix}"
        return ";".join(out)

    return sorted(found, key=lambda f: -len(f[3])), rebuild

def _sub(code, pattern, repl, span, note, last=False, every=False):
    """One regex edit as a candidate `(code, span, note)`, or nothing when the pattern is absent."""
    matches = list(re.finditer(pattern, code))
    if not matches:
        return []
    if every:
        return [(re.sub(pattern, repl, code), matches[0].expand(span), matches[0].expand(note))]
    m = matches[-1] if last else matches[0]
    return [(code[: m.start()] + m.expand(repl) + code[m.end() :], m.expand(span), m.expand(note))]

_NUMBERS = {0: 50, 50: 5, 80: 8, 1: 3, 2: 5, 3: 1, 5: 2, 10: 100, 30: 3, 100: 7, 1000: 100}

def _vocabulary(chain):
    layout = chain["layout"]
    return layout["components"] + layout["topics"]

def _request_misunderstood(chain):
    code, layout, out = chain["code"], chain["layout"], []
    for level, other in (("ERROR", "WARN"), ("WARN", "INFO"), ("INFO", "ERROR")):
        out += _sub(code, rf"(?<![A-Za-z]){level}(?![A-Za-z])", other, other, f"Uses {other} where the request said {level}.", every=True)
    owners = layout["owners"]
    for i, owner in enumerate(owners):
        other = owners[(i + 1) % len(owners)]
        out += _sub(code, rf"'{owner}'", f"'{other}'", f"'{other}'", f"Uses '{other}' where the request said '{owner}'.", every=True)
    words = _vocabulary(chain)
    for i, term in enumerate(words):
        other = words[(i + 3) % len(words)]
        out += _sub(code, rf"'{term}'", f"'{other}'", f"'{other}'", f"Searches for '{other}' where the request said '{term}'.", every=True)
        out += _sub(code, rf"\*{term}\*", f"*{other}*", f"*{other}*", f"Looks for names with '{other}' where the request said '{term}'.")
    for key in list(OPTIONAL_KEYS) + ["timeout_ms", "max_connections"]:
        other = key.replace("_", "") if "_" in key else key + "s"
        out += _sub(code, rf"\b{key}\b", other, other, f"Reads `{other}` where the request named `{key}`.", every=True)
    for m in re.finditer(r"(first|take|>=|default) (\d+)\b", code):
        n = int(m.group(2))
        wrong = _NUMBERS.get(n, n * 10 + 1)
        out.append((code[: m.start(2)] + str(wrong) + code[m.end(2) :], f"{m.group(1)} {wrong}", f"Uses {wrong} where the request said {n}."))
    out += _sub(code, r"== true", "== false", "== false", "Keeps the rows where the condition is false; the request asked for the true ones.")
    out += _sub(code, r"== false", "== true", "== true", "Keeps the rows where the condition is true; the request asked for the false ones.")
    out += _sub(code, r"'TODO\|FIXME'", "'NOTE|HACK'", "'NOTE|HACK'", "Looks for other markers than the ones requested.")
    out += _sub(code, r"'(TODO|FIXME)'", "'XXX'", "'XXX'", r"Looks for XXX where the request said \1.")
    out += _sub(code, r"'([\w/]+)(\w)\.(toml|json|yaml|py|js|go|rb|rs|log|md)'", r"'\1x.\3'", r"'\1x.\3'", "The file name is not the one requested.")
    out += _sub(code, r"\|(\w+)'\)", r"|\1s_old')", r"|\1s_old'", "One of the two name patterns is not what was requested.")
    return out

def _wrong_data_source(chain):
    code, layout, out = chain["code"], chain["layout"], []
    dirs, columns = layout["dirs"], layout["columns"]
    logs = [f"{dirs['logs']}/{name}" for name in layout["log_names"]]
    for log, other in ((logs[0], logs[1]), (logs[1], logs[0])):
        out += _sub(code, rf"(open --raw) {re.escape(log)}", rf"\1 {other}", f"open --raw {other}", f"Reads {other}; the request was about {log}.")
    table_path = f"{dirs['cfg']}/{layout['table']}"
    table = f"[[{columns['key']} {columns['flag']} {columns['owner']} {columns['number']}]; [demo_a true {layout['owners'][0]} 10] [demo_b false {layout['owners'][1]} 90] [demo_c true {layout['owners'][2]} 55]]"
    out += _sub(code, rf"open {re.escape(table_path)}", table, table, f"Uses a made-up example table and never opens {table_path}.")
    config_path = f"{dirs['cfg']}/{layout['config']}"
    record = f"{{{layout['service_section']}: {{name: 'example', port: 8000}}, {layout['store_section']}: {{host: 'localhost', port: 5432, max_connections: 5}}}}"
    out += _sub(code, rf"open {re.escape(config_path)}", record, record, f"Uses a made-up record and never opens {config_path}.")
    roles = ["src", "cfg", "logs", "docs", "scripts"]
    for i, role in enumerate(roles):  # the same pattern, pointed at another folder
        here, there = re.escape(dirs[role]), dirs[roles[(i + 2) % len(roles)]]
        out += _sub(code, rf"(glob|ls) {here}/(\S*?)(?=[ )])", rf"\1 {there}/**/*", rf"\1 {there}/**/*", f"Looks in {there}; the request named {dirs[role]}.")
    out += _sub(code, r"(glob|ls) \*\*/\*\.\{[\w,]+\}", rf"\1 {dirs['docs']}/**/*", rf"\1 {dirs['docs']}/**/*", "Looks in a different place than the request named.")
    out += _sub(code, r"(glob|ls) \*\*/\*\.(\w+)", rf"\1 {dirs['src']}/**/*", rf"\1 {dirs['src']}/**/*", "Looks in a different place than the request named.")
    out += _sub(code, r"(glob|ls) \*\*/\*(?=[ )])", rf"\1 {dirs['src']}/**/*", rf"\1 {dirs['src']}/**/*", "Looks in a different place than the request named.")
    return out

def _wrong_operation(chain):
    code, columns, out = chain["code"], chain["layout"]["columns"], []
    k, o, n, b = columns["key"], columns["owner"], columns["number"], columns["flag"]
    out += _sub(code, r"sort-by (\w+) --reverse", r"sort-by \1", r"sort-by \1", "Sorts ascending; the request wanted the largest first.", last=True)
    out += _sub(code, r"first (\d+)", r"last \1", r"last \1", "Takes the last rows instead of the first.")
    out += _sub(code, r">= (\d+)", r"< \1", r"< \1", "The comparison is inverted.")
    out += _sub(code, r"hits > 0", "hits == 0", "hits == 0", "Keeps the files without a match.")
    out += _sub(code, r"== '(\w+)'", r"!= '\1'", r"!= '\1'", "Keeps everything except what was asked for.", last=True)
    for a, c, why in (
        ("math avg", "math sum", "Sums where an average was asked for."),
        ("math sum", "length", "Counts the rows instead of summing them."),
        ("math max", "math min", "Takes the minimum instead of the maximum."),
        ("str downcase", "str upcase", "Upper-cases instead of lower-casing."),
        ("path basename", "path dirname", "Returns directories instead of file names."),
        ("path dirname", "path basename", "Returns file names instead of directories."),
        ("is-empty", "is-not-empty", "The emptiness test is inverted."),
        ("select ts msg", "select ts level", "Returns the level column instead of the message."),
        ("get msg", "get ts", "Returns timestamps instead of messages."),
        ("select name size", "select name type", "Returns the type column instead of the size."),
        (f"select {k} {b} {n}", f"select {k} {o}", "Returns other columns than requested."),
        (f"select {k} {n} reviewer", f"select {k} reviewer", f"Drops the {n} column."),
        (f"select {k} {o} {n}", f"select {k} {b}", "Returns other columns than requested."),
        ("$p.stem", "$p.parent", "Reports the parent directory as the stem."),
        ("str contains -i", "str starts-with -i", "Matches only at the start of a line."),
        ("str contains '", "str ends-with '", "Matches only at the end of a line."),
        ("| length)", "| str join | str length)", "Counts characters instead of lines."),
        ("uniq | sort", "sort", "Keeps duplicates."),
        ("| columns", "| values", "Lists the values instead of the keys."),
        ("status: 'choose_one'", "status: 'resolved'", "Reports an ambiguous match as resolved."),
        ("default ", "default (-1) | default ", "Falls back to -1 instead of the requested default."),
        ("candidates: $choices", "candidates: ($choices | length)", "Returns how many candidates there are, not which."),
        ("=~ '(?i)", "!~ '(?i)", "Keeps the paths that do not match."),
        ("=~ '", "!~ '", "Keeps the lines that do not match."),
        ("append $by_b", "append $by_a", "Combines the first search with itself."),
        ("insert count {|r| $r.items | length}", f"insert count {{|r| $r.items | get {n} | math sum}}", f"Sums {n} where a count was asked for."),
    ):
        if a in code:
            at = code.index(a)
            out.append((code[:at] + c + code[at + len(a) :], c, why))
    return out

def _broken_chain(chain):
    found, rebuild = _pipelines(chain["code"])
    drops, swaps = [], []
    for idx, lead, prefix, stages, suffix in found:
        for i in range(1, len(stages) - 1):
            kept = stages[:i] + stages[i + 1 :]
            span = f"{stages[i - 1][-28:]} | {stages[i + 1][:28]}"
            drops.append((rebuild(idx, lead, prefix, kept, suffix), span, f"The `{stages[i][:60]}` step is missing."))
        for i in range(1, len(stages) - 2):
            moved = stages[:i] + [stages[i + 1], stages[i]] + stages[i + 2 :]
            span = f"{stages[i + 1][-28:]} | {stages[i][:28]}"
            swaps.append((rebuild(idx, lead, prefix, moved, suffix), span, f"`{stages[i + 1][:40]}` runs before `{stages[i][:40]}`; the order is reversed."))
    mixed = []
    for pair in zip(drops, swaps):
        mixed.extend(pair)
    return mixed + drops[len(swaps) :] + swaps[len(drops) :]

def _wrong_output(chain):
    code, out, tail = chain["code"], [], " | to json --raw"
    if code.endswith(tail):
        body = code[: -len(tail)]
        out.append((body, body[-40:], "Returns Nu's table rendering; the request needs JSON."))
        out.append((body + " | to yaml", "| to yaml", "Returns YAML where JSON was needed."))
        out.append((code + " | save -f results.json", "| save -f results.json", "Writes results.json, which nobody asked for, and returns nothing."))
    out.append((f"print 'Here are the results:'; {code}", "print 'Here are the results:'", "Prints extra text before the answer."))
    return out

def _runtime_failure(chain):
    code, layout, out = chain["code"], chain["layout"], []
    columns, store, service = layout["columns"], layout["store_section"], layout["service_section"]
    o, n, b = columns["owner"], columns["number"], columns["flag"]
    out += _sub(code, r"(open(?: --raw)?) ([\w/]+)\.(toml|json|log|md)", r"\1 \2_old.\3", r"\1 \2_old.\3", "Opens a path that does not exist.")
    for a, c in (
        (f"get {store}", f"get {store}s_v2"), ("$store.host", "$store.hostname"), (f"$cfg.{service}.name", f"$cfg.{service}_v2.name"),
        ("$r.name", "$r.path"), ("get name", "get path"), ("get level", "get severity"), ("level ==", "severity =="),
        (f"get {n}", f"get {n}_pct"), (f"{o} ==", f"{o}_id =="), (f"{n} >=", f"{n}_pct >="), (f"where {b}", f"where {b}_now"),
        (f"where not {b}", f"where not {b}_now"), (f"group-by {o}", f"group-by {o}_id"), ("get msg", "get message"),
        ("select ts msg", "select time msg"), ("select name size", "select name bytes"), ("where type == file", "where kind == file"),
        ("$g.value", "$g.label"), ("$matches | first", "$matches | get 5"), ("$matches.0", "$matches.9"),
        ("$it | str contains", "$it.text | str contains"), ("path relative-to (pwd)", "path relative-to '/nonexistent'"),
    ):
        if a in code:
            at = code.index(a)
            out.append((code[:at] + c + code[at + len(a) :], c, f"`{c}` does not exist at run time."))
    return out

def _invalid_nushell(chain):
    code, out = chain["code"], []
    for closer, name in ((")", "parenthesis"), ("}", "brace")):
        if closer in code:
            at = code.rindex(closer)
            broken = code[:at] + code[at + 1 :]
            out.append((broken, broken[max(0, at - 24) : at + 6].strip(), f"A closing {name} is missing."))
    out += _sub(code, r"; ", " && ", "&&", "Chains statements with &&, which Nu does not have.")
    out += _sub(code, r"let (\w+) = \((glob|open|ls)", r"\1=$(\2", r"\1=$(\2", "Bash-style assignment and command substitution.")
    out += _sub(code, r"\{\|(\w+)\| ", r"lambda \1: {", r"lambda \1:", "Python lambda syntax in place of a Nu closure.")
    out += _sub(code, r" \| where ", " | grep -v '^$' | awk '{print $1}' | where (", "awk '{print $1}' | where (", "Pipes through grep and awk with an unclosed parenthesis.")
    out += _sub(code, r"if \((.+?)\) \{", r"if [ \1 ]; then {", r"if [ \1 ]; then", "Bash test syntax.")
    return out

def _equivalent_answers(chain):
    """Correct answers a strict comparison would reject: other formatting, column order, type, or tie order."""
    code, out, tail = chain["code"], [], " | to json --raw"
    if code.endswith(tail):
        body = code[: -len(tail)]
        out.append((body + " | to json", "| to json", "The same answer as indented JSON: formatting only."))
        out.append((body + " | to json --raw | str trim", "| str trim", "The same answer through an extra no-op step: formatting only."))
    for m in re.finditer(r"select (\w+) (\w+)(?: (\w+))?", code):
        columns = [c for c in m.groups() if c]
        swapped = "select " + " ".join(reversed(columns))
        out.append((code[: m.start()] + swapped + code[m.end() :], swapped, "The same columns in another order."))
    out += _sub(code, r"sort-by (\w+) \| sort-by (\w+)( --reverse)?", r"sort-by \2\3", r"sort-by \2\3", "The same ranking with ties left in a different order: the request does not say how to break them.")
    if not code.endswith(tail):
        out.append((code + " | into string", "| into string", "The same value as text instead of a number."))
    return out

def _loosely_equal(a: str, b: str) -> bool:
    """The same answer once formatting, key and row order, and number-versus-text are set aside."""
    def canon(value):
        if isinstance(value, dict):
            return {k: canon(v) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return sorted((canon(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
        return str(value).lower() if isinstance(value, (bool, int, float)) else value
    try:
        return canon(json.loads(a)) == canon(json.loads(b))
    except ValueError:
        return " ".join(a.split()).lower() == " ".join(b.split()).lower()

def _from_the_sketch(chain, failure_type):
    """Examples the failure table names that are not edits of a single step."""
    code, layout, out = chain["code"], chain["layout"], []
    columns = layout["columns"]
    if failure_type == "invalid_nushell":
        out.append((f"def run [dir: strng, --limit(-l) int] {{ {code} }}; run .", "def run [dir: strng, --limit(-l) int]", "The function signature is invalid: an unknown type and a flag declared without a colon."))
    if failure_type == "broken_chain_or_function":
        out.append((f"def run [] {{ {code} }}", "def run []", "Defines the work as a function and never calls it."))
        bound = re.match(r"let (\w+) = \((.+?)\); \$\1 \| ", code)
        if bound:  # the bound value is ignored and the second half runs on something else
            first_step = chain["parts"][0]
            out.append((code.replace(f"${bound.group(1)} | ", f"{first_step} | ", 1), first_step, f"Binds `{bound.group(1)}` and then does not use it: the rest runs on the unfiltered input."))
    if failure_type == "wrong_data_source":
        for a, b in ((columns["owner"], columns["key"]), (columns["number"], columns["flag"]), (columns["key"], columns["owner"])):
            out += _sub(code, rf"\b(get|group-by|sort-by) {a}\b", rf"\1 {b}", rf"\1 {b}", f"Reads the `{b}` column where the request meant `{a}`.")
    if failure_type == "wrong_operation_or_argument":
        for a, b in ((columns["number"], columns["key"]), ("size", "name"), ("hits", "file"), ("count", "value")):
            out += _sub(code, rf"sort-by {a}( --reverse)?(?= |$)", rf"sort-by {b}\1", rf"sort-by {b}", f"Sorts by `{b}` where the request said `{a}`.", last=True)
        out += _sub(code, r"math sum", "length", "length", "Counts the values instead of summing them.")
    if failure_type == "runtime_or_environment_failure":
        out.append((f"sleep 30sec; {code}", "sleep 30sec", "Waits longer than the tool allows and times out."))
        for a, b, why in (
            ("to json --raw", "to-json --raw", "`to-json` is how an older release spelled it; this one has no such command."),
            ("| lines", "| split-row (char nl)", "`split-row` is from an older release; this one has no such command."),
            ("str contains", "str find", "`str find` does not exist in this release."),
            ("sort-by", "sort_by", "`sort_by` is not a command in this release."),
            ("ls ", "ls --tree ", "`ls` has no tree view in this release."),
        ):
            if a in code:
                at = code.index(a)
                out.append((code[:at] + b + code[at + len(a) :], b.strip(), why))
    return out

def _hardcoded(chain):
    literal = chain["expected"]
    if len(literal) > 700 or chain["expected_variant"] in (None, chain["expected"]):
        return []
    quoted = f"'{literal}'" if "'" not in literal else f"r#'{literal}'#"
    note = "The answer is pasted in as a literal; nothing in the workspace is read."
    return [(f"{quoted} | from json | to json --raw", quoted, note), (quoted, quoted, note)]

def _malformed_calls(chain):
    code = chain["code"]
    good = wrap_tool_call(code)
    return [
        (good.replace("run_nushell", "run_bash"), "<function=run_bash>", "Calls a tool that was not offered."),
        (f"You can do that with this command:\n```nu\n{code}\n```", "You can do that with this command:", "Answers in prose; there is no tool call."),
        (good.replace("<parameter=code>", "<parameter=command>"), "<parameter=command>", "The parameter is named `command`; the tool takes `code`."),
        ('{"name": "run_nushell", "arguments": {"code": ' + repr(code)[:400] + "}}\n</tool_call>", '{"name": "run_nushell", "arguments":', "A JSON call where the function-tag format is required."),
        (good.replace("\n</parameter>\n</function>", ""), "</tool_call>", "The parameter and function tags are never closed."),
        (good.replace("</tool_call>", "") + "<tool_call>\n" + good, "<tool_call>\n<function=run_nushell>", "Makes the same call twice; one call was asked for."),
    ]

_COMMON_FOLDERS = ["src", "lib", "app", "config", "conf", "logs", "log", "docs", "scripts", "bin", "tests", "data", "build"]

def _assumed_layout(chain):
    """What the target model was seen to do: write a path for a folder or file this project does not have,
    as if every project were laid out the same way. Whether that fails outright or quietly returns the
    wrong thing decides the label."""
    code, dirs, out = chain["code"], chain["layout"]["dirs"], []
    missing = [d for d in _COMMON_FOLDERS if d not in dirs.values()]
    for i, own in enumerate(sorted(set(dirs.values()))):
        other = missing[(len(chain["chain_id"]) + i) % len(missing)]
        out += _sub(code, rf"(?<![\w/]){re.escape(own)}/", f"{other}/", f"{other}/", f"Assumes a `{other}/` folder; this project has `{own}/`.")
    return out

def _shell_habits(chain):
    """Commands written the way another shell would have them."""
    code, parts, out = chain["code"], chain["parts"], []
    ext = chain["layout"]["ext"]
    if parts[0].startswith(("ls", "glob")):
        out.append((code.replace(parts[0], f"find . -type f -name '*.{ext}'", 1), f"find . -type f -name '*.{ext}'", "Uses `find` with POSIX flags; Nushell's `find` is a different command."))
    if parts[0].startswith("open --raw"):
        path = parts[0].split(" ")[2]
        out.append((code.replace(parts[0], f"cat {path} | grep -v '^$'", 1), f"cat {path} | grep -v '^$'", "Pipes through `cat` and `grep` instead of reading the file with Nushell."))
    return out

_RULES = {
    "request_misunderstood": _request_misunderstood,
    "wrong_data_source": _wrong_data_source,
    "wrong_operation_or_argument": _wrong_operation,
    "broken_chain_or_function": _broken_chain,
    "wrong_output_or_side_effect": _wrong_output,
    "runtime_or_environment_failure": _runtime_failure,
    "invalid_nushell": _invalid_nushell,
    "grader_false_pass": _hardcoded,
    "grader_false_fail": _equivalent_answers,
}

def confirms(failure_type: str, code: str, chain: dict) -> bool:
    """Does running `code` agree with the label?"""
    result = run_nu(code, chain["env"])
    differs = result["ok"] and not same_output(result["stdout"], chain["expected"])
    if failure_type == "invalid_nushell":
        return result["parse_error"]
    if failure_type == "runtime_or_environment_failure":
        return not result["ok"] and not result["parse_error"]
    if failure_type == "grader_false_fail":
        return result["ok"] and result["stdout"].strip() != chain["expected"].strip() and _loosely_equal(result["stdout"], chain["expected"])
    if failure_type == "wrong_output_or_side_effect":
        return result["ok"] and (result["stdout"].strip() != chain["expected"].strip() or bool(result["changed"]))
    if failure_type == "grader_false_pass":
        return result["ok"] and same_output(result["stdout"], chain["expected"])
    return differs

def confirmed_breaks(chain: dict, index: int) -> dict:
    """For one chain, the first candidate of each failure type that execution confirms."""
    found = {}
    for failure_type in BREAK_LABELS:
        if failure_type == "wrong_tool_or_malformed_call":
            options = _malformed_calls(chain)
            completion, span, note = options[index % len(options)]
            found[failure_type] = {"failure_type": failure_type, "code": chain["code"], "completion": completion, "span_text": span, "note": note}
            continue
        # mutations of the sampled structure first, then edits of the text, then the habits seen in model output
        candidates = structural_breaks(chain, failure_type, index) + _RULES[failure_type](chain) + _from_the_sketch(chain, failure_type)
        if failure_type in ("wrong_data_source", "runtime_or_environment_failure"):
            candidates = _assumed_layout(chain) + candidates
        if failure_type in ("invalid_nushell", "runtime_or_environment_failure"):
            candidates = candidates + _shell_habits(chain)
        shift = index % len(candidates) if candidates else 0
        for code, span, note in (candidates[shift:] + candidates[:shift])[:12]:
            if code != chain["code"] and span in code and confirms(failure_type, code, chain):
                found[failure_type] = {"failure_type": failure_type, "code": code, "completion": wrap_tool_call(code), "span_text": span, "note": note}
                break
    return found

def build_samples(chains: list[dict], breaks_per_chain: int) -> list[dict]:
    """Per chain: the reference call, its verified alternative form, and `breaks_per_chain` confirmed
    breaks. Which types a chain gets is decided after all chains are tried: each takes the types that are
    so far rarest in the dataset, so a type that few chains can produce is not crowded out."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        possible = list(pool.map(lambda pair: confirmed_breaks(pair[1], pair[0]), enumerate(chains)))
    save_nu_cache()
    used = {label: 0 for label in BREAK_LABELS}
    chosen = [[] for _ in chains]
    for i in sorted(range(len(chains)), key=lambda i: len(possible[i])):  # the least flexible chains choose first
        for label in sorted(possible[i], key=lambda label: (used[label], BREAK_LABELS.index(label)))[:breaks_per_chain]:
            chosen[i].append(possible[i][label])
            used[label] += 1
    samples = []
    for chain, breaks in zip(chains, chosen):
        base = {"chain_id": chain["chain_id"], "env": chain["env"], "family": chain["family"], "category": chain["category"]}
        valid = {"failure_type": "valid", "is_broken": False, "span_text": "", "note": FAILURE_TYPES["valid"]}
        samples.append({**base, **valid, "form": "reference", "code": chain["code"], "completion": wrap_tool_call(chain["code"])})
        if chain["alt_ok"]:
            samples.append({**base, **valid, "form": "alternative", "code": chain["alt_code"], "completion": wrap_tool_call(chain["alt_code"])})
        for item in breaks:
            assert item["span_text"] in item["completion"], (chain["chain_id"], item["failure_type"])
            samples.append({**base, **item, "is_broken": item["failure_type"] not in NOT_BROKEN, "form": "break"})
    return samples

def preview_break(chain, failure_type, index):
    """One break, start to finish: the working command, the first broken variant of this type that
    running it confirms, the part that is wrong, and why."""
    before = chain["code"]
    after, span, reason = confirmed_break(chain, index, failure_type)
    return after is not None

def confirmed_break(chain, index, failure_type):
    """The call, the broken span and the reason of the first variant of this type that execution confirms."""
    found = confirmed_breaks(chain, index).get(failure_type)
    if not found:
        return None, None, "no variant of this type survives execution for this chain"
    return found["code"], found["span_text"], found["note"]

BREAK_SCOPE = {"confirmed_break": confirmed_break}
