#!/usr/bin/env python3
"""Build citable shell-command usage frequencies and a Unix -> Nushell command map.

    .venv/bin/python scripts/fetch_command_usage.py                       # default sources
    .venv/bin/python scripts/fetch_command_usage.py --include-restricted  # also the UCI data (see below)

Writes
    data/command_usage.json   per-source counts of the utility at the head of each pipeline stage,
                              plus a documented combination
    data/unix_to_nushell.json equivalences stated in the Nushell book, with the table rows as evidence

Raw downloads are cached under data/usage_cache/ (git-ignored). Each cached file gets a sidecar
``<file>.meta.json`` holding its URL, retrieval date and sha256, so a rerun with the same cache
reproduces both outputs byte for byte. Every remote source is pinned to a revision (git commit,
Hugging Face revision, Zenodo record version), and the Zenodo file is checked against the md5 that
Zenodo publishes.

Counting rules (applied identically to every source that contains command lines)
    * A command line is split into pipeline stages on ``|``, ``|&``, ``;``, ``&&``, ``||``, ``&`` and
      newlines. ``$(...)``, backticks and ``<(...)``/``>(...)`` are parsed as further command lines.
    * The head of each stage is counted once. Leading ``VAR=value`` assignments and redirections are
      skipped. ``then``/``do``/``else``/``elif``/``!``/``{``/``(`` are skipped and the command after
      them is counted. ``if``/``while``/``until`` are counted and so is the command that follows
      (the condition). ``for``/``case``/``select`` are counted and the rest of the stage is ignored.
      ``fi``/``done``/``esac``/``}`` are not counted.
    * Wrappers in WRAPPERS (sudo, doas, env, xargs, time, nohup, nice, exec, command, builtin,
      timeout, watch) are counted, their own options are skipped, and the wrapped utility is counted
      too when one is present. ``find ... -exec/-execdir/-ok/-okdir CMD`` also counts CMD.
    * A head containing ``/`` is reduced to its basename, except ``./x``, ``../x`` and ``~/x`` which
      are counted as ``<./script>``. Heads that are not plain names (variables, ``-flags``, numbers,
      dotfiles, prompt residue such as ``user@host:~$``, ``diff`` output such as ``2c1``, globs) are
      not counted; they are reported as ``unparsed_stages``.
    * Within a source, commands below ``min_count = max(2, round(2e-6 * total_heads))`` are dropped
      from ``counts`` (the dropped mass is reported). Relative frequencies use the full total.
"""

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import re
import sqlite3
import sys
import tarfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "usage_cache"
OUT_USAGE = ROOT / "data" / "command_usage.json"
OUT_MAP = ROOT / "data" / "unix_to_nushell.json"
USER_AGENT = "command-usage-fetcher/1.0 (python urllib)"

NL2BASH_COMMIT = "d6b9f5bdff45621d190134e31ab63b7bf7002190"
HF_BASH_HISTORY_REV = "4683648ef83c4b5026e219916b7f993a64b82621"
ZENODO_ALIAS_RECORD = "4007049"
ZENODO_ALIAS_MD5 = "ec57aa4d7a9b2a7e6e3815206fda4ddd"
NUSHELL_BOOK_COMMIT = "c4b4da1849e6a844d6914876e4d4f591bb0604fd"

SOURCES = {
    "nl2bash": {
        "title": "NL2Bash corpus, data/bash/all.cm (Lin et al., LREC 2018)",
        "url": f"https://raw.githubusercontent.com/TellinaTool/nl2bash/{NL2BASH_COMMIT}/data/bash/all.cm",
        "homepage": "https://github.com/TellinaTool/nl2bash",
        "license": "MIT for data/bash (data/bash/LICENSE and README: 'The dataset data/bash is separately licensed under MIT license'); the repository code is GPL-3.0",
        "kind": "task corpus",
        "note": "One-liners collected from Q&A sites and tutorials and written by programmers to answer natural-language "
                "file-system/text tasks. Reflects what experts write to solve posed tasks, not what people type day to day. "
                "Heavily skewed towards find (and xargs/grep/-exec pipelines); interactive staples such as cd, git, vim, ssh are nearly absent.",
        "cache": "nl2bash/all.cm",
    },
    "hf_bash_history": {
        "title": "spignelon/bash_history (Hugging Face dataset, bash.txt)",
        "url": f"https://huggingface.co/datasets/spignelon/bash_history/resolve/{HF_BASH_HISTORY_REV}/bash.txt",
        "homepage": "https://huggingface.co/datasets/spignelon/bash_history",
        "license": "MIT as declared by the uploader in the dataset card; the card gives no provenance. The companion project "
                   "(github.com/spignelon/CleverTerm, GPL-3.0) says the data came from the Kaggle dataset thuyngandao/bashlogs and from "
                   ".bash_history files uploaded to GitHub and gists, whose original licenses are not documented",
        "kind": "shell history",
        "note": "Concatenated real interactive bash histories of an unknown number of (mostly Linux sysadmin/developer) users, one "
                "command line per line, repeats kept. Closest of the sources to day-to-day typing. Unknown deduplication, unknown "
                "per-user weighting (a few heavy users may dominate), some prompt/output residue lines (not counted).",
        "cache": "hf_bash_history/bash.txt",
    },
    "shell_aliases_github": {
        "title": "Shell Aliases on GitHub, results.db (Schröder & Cito, Zenodo 10.5281/zenodo.4007049; paper: 'An Empirical Investigation of Command-Line Customization', EMSE 2022, arXiv:2012.10206)",
        "url": f"https://zenodo.org/api/records/{ZENODO_ALIAS_RECORD}/files/results.db/content",
        "homepage": f"https://zenodo.org/records/{ZENODO_ALIAS_RECORD}",
        "license": "CC-BY-4.0 (Zenodo record metadata)",
        "kind": "alias corpus",
        "note": "Bodies of 2.2M alias definitions from 844k shell files in 304k GitHub repositories (Dec 2019-Jan 2020), counted "
                "with the same parser (alias.value column). Counts how often a utility appears in an alias someone bothered to "
                "define, not how often it is run: favours git, ls, cd, docker, kubectl and long commands; copied dotfile "
                "frameworks (e.g. oh-my-zsh plugin files committed into many repos) are counted once per copy.",
        "cache": "schroeder_cito_aliases/results.db",
    },
    "uci_unix_user_data": {
        "title": "UNIX User Data (Terran Lane, UCI KDD Archive, doi:10.24432/C5302K)",
        "url": "https://archive.ics.uci.edu/static/public/141/unix+user+data.zip",
        "homepage": "https://archive.ics.uci.edu/dataset/141/unix+user+data",
        "license": "README: research use only; summaries may be published but literal tokens or token sequences may not be "
                   "published without consent of the originators; no inclusion in a commercial product. Excluded unless --include-restricted",
        "kind": "shell history",
        "note": "Sanitised tcsh histories of 8 Purdue users (9 sets) over up to 2 years, late 1990s. Arguments are replaced "
                "by <n> but command names, flags and metacharacters are kept; line boundaries are lost, so every bare word "
                "token is treated as a command head. Small, old, academic population.",
        "cache": "uci_unix_user_data/unix_user_data.zip",
        "restricted": True,
    },
}
DEFAULT_SOURCES = ["nl2bash", "hf_bash_history", "shell_aliases_github"]

NUSHELL_PAGES = {
    "coming_from_bash.md": f"https://raw.githubusercontent.com/nushell/nushell.github.io/{NUSHELL_BOOK_COMMIT}/book/coming_from_bash.md",
    "nushell_map.md": f"https://raw.githubusercontent.com/nushell/nushell.github.io/{NUSHELL_BOOK_COMMIT}/book/nushell_map.md",
}
# Checked and found to have no Bash/Unix column, so nothing is parsed from them.
NUSHELL_PAGES_WITHOUT_BASH = [
    f"https://raw.githubusercontent.com/nushell/nushell.github.io/{NUSHELL_BOOK_COMMIT}/book/nushell_map_imperative.md",
    f"https://raw.githubusercontent.com/nushell/nushell.github.io/{NUSHELL_BOOK_COMMIT}/book/nushell_map_functional.md",
]
NUSHELL_TREE_URL = f"https://api.github.com/repos/nushell/nushell.github.io/git/trees/{NUSHELL_BOOK_COMMIT}?recursive=1"

# Written by the script's author, not by the Nushell book. Listed with no weight and kept out of "map".
# Each entry is dropped automatically if the book does state a mapping for that utility, or if the
# Nushell command is not in the book's commands/docs listing.
SUGGESTED = {
    "sort": (["sort"], "Nushell has a builtin `sort`; the book only maps Bash `sort` to `sort-by`"),
    "less": (["explore"], "the book mentions explore for paging internal output only in a note"),
    "more": (["explore"], "as less"),
    "seq": (["seq"], "Nushell builtin `seq`"),
    "sleep": (["sleep"], "Nushell builtin `sleep`"),
    "whoami": (["whoami"], "Nushell builtin `whoami`"),
    "basename": (["path basename"], "Nushell builtin `path basename`"),
    "dirname": (["path dirname"], "Nushell builtin `path dirname`"),
    "realpath": (["path expand"], "Nushell builtin `path expand`"),
    "readlink": (["path expand"], "`path expand` resolves symlinks"),
    "env": (["$env", "with-env"], "`$env` record / `with-env`"),
    "printenv": (["$env"], "`$env` record"),
    "cut": (["select", "split column", "split row"], "column/field selection"),
    "tr": (["str replace", "str upcase", "str downcase"], "character translation"),
    "awk": (["split column", "where", "each", "get", "math sum"], "field-wise processing"),
    "jq": (["from json", "get", "where", "to json"], "structured JSON processing"),
    "xargs": (["each", "par-each"], "the book maps `xargs -P` to par-each; plain xargs is each"),
    "tac": (["reverse"], "Nushell builtin `reverse`"),
    "rev": (["str reverse"], "Nushell builtin `str reverse`"),
    "nl": (["enumerate"], "Nushell builtin `enumerate`"),
    "paste": (["zip", "str join"], ""),
    "split": (["chunks"], "Nushell builtin `chunks`"),
    "md5sum": (["hash md5"], "Nushell builtin `hash md5`"),
    "sha256sum": (["hash sha256"], "Nushell builtin `hash sha256`"),
    "base64": (["encode base64", "decode base64"], "Nushell builtins"),
    "ln": (["mklink"], ""),
    "source": (["source"], "Nushell keyword `source`"),
    ".": (["source"], ""),
    "test": (["path exists", "path type"], "file tests"),
    "[": (["path exists", "path type"], "file tests"),
    "true": (["true"], ""),
    "watch": (["watch"], "Nushell builtin `watch` (watches files, not a command repeat)"),
    "cal": (["cal"], "Nushell builtin `cal`"),
    "ping": ([], ""),
}

WRAPPERS = {"sudo", "doas", "env", "xargs", "time", "nohup", "nice", "exec", "command", "builtin", "timeout", "watch"}
WRAPPER_OPTS_WITH_ARG = {
    "sudo": {"-u", "-g", "-h", "-p", "-U", "-C", "-r", "-t", "-D", "-R", "-T", "--user", "--group", "--host", "--prompt",
             "--other-user", "--close-from", "--role", "--type", "--chdir", "--chroot", "--command-timeout"},
    "doas": {"-u", "-C"},
    "env": {"-u", "-C", "--unset", "--chdir"},
    "xargs": {"-I", "-L", "-n", "-P", "-s", "-d", "-E", "-a", "--arg-file", "--delimiter", "--max-args", "--max-procs",
              "--max-lines", "--max-chars", "--process-slot-var"},
    "time": {"-f", "-o", "--format", "--output"},
    "nohup": set(),
    "nice": {"-n", "--adjustment"},
    "exec": {"-a"},
    "command": set(),
    "builtin": set(),
    "timeout": {"-s", "-k", "--signal", "--kill-after"},
    "watch": {"-n", "--interval", "-d", "--differences", "-q", "--equexit"},
}
SKIP_THEN_CONTINUE = {"then", "do", "else", "elif", "!", "{", "(", "((", "coproc"}
COUNT_THEN_CONTINUE = {"if", "while", "until"}
COUNT_THEN_STOP = {"for", "case", "select", "function"}
NOT_COUNTED = {"fi", "done", "esac", "}", ")", "in", ";;", "))"}
SEP_OPS = {"|", "|&", ";", "&&", "||", "&", "\n", ";;", "(", ")"}
NAME_RE = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_.+-]*$")
ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\+?=")
PLACEHOLDERS = {"command", "command1", "command2", "cmd", "cmd1", "cmd2", "<command>", "<command1>", "<command2>"}


# --------------------------------------------------------------------------------------------- download

def cached(source_id, rel, url, md5=None):
    path = CACHE / rel
    meta_path = path.with_name(path.name + ".meta.json")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}", file=sys.stderr)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        tmp = path.with_name(path.name + ".part")
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        tmp.rename(path)
        if meta_path.exists():
            meta_path.unlink()
    if not meta_path.exists():
        sha, h_md5 = hashlib.sha256(), hashlib.md5()
        with open(path, "rb") as fh:
            while chunk := fh.read(1 << 22):
                sha.update(chunk)
                h_md5.update(chunk)
        if md5 and h_md5.hexdigest() != md5:
            raise SystemExit(f"{path}: md5 {h_md5.hexdigest()} != published {md5}; delete it and rerun")
        meta = {"url": url, "retrieved": dt.date.today().isoformat(), "sha256": sha.hexdigest(), "bytes": path.stat().st_size}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return path, json.loads(meta_path.read_text())


# --------------------------------------------------------------------------------------------- shell parsing

def _read_balanced(s, i, open_ch="(", close_ch=")"):
    """s[i] is just past an opening paren; return (inner_text, index_after_close)."""
    depth, j, quote = 1, i, None
    while j < len(s):
        c = s[j]
        if quote:
            if c == "\\" and quote == '"':
                j += 2
                continue
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c == "\\":
            j += 2
            continue
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[i:j], j + 1
        j += 1
    return s[i:], len(s)


def tokenize(line):
    """Return (tokens, substitutions). Tokens are ('W', word) or ('OP', op) or ('R', redir)."""
    toks, subs = [], []
    word, in_word = [], False
    i, n = 0, len(line)

    def end_word():
        nonlocal word, in_word
        if in_word:
            toks.append(("W", "".join(word)))
        word, in_word = [], False

    while i < n:
        c = line[i]
        if c in " \t\r":
            end_word(); i += 1; continue
        if c == "\n":
            end_word(); toks.append(("OP", "\n")); i += 1; continue
        if c == "#" and not in_word:
            while i < n and line[i] != "\n":
                i += 1
            continue
        if c == "\\":
            if i + 1 < n and line[i + 1] == "\n":
                i += 2; continue
            word.append(line[i + 1] if i + 1 < n else ""); in_word = True; i += 2; continue
        if c == "'":
            j = line.find("'", i + 1)
            j = n if j < 0 else j
            word.append(line[i + 1:j]); in_word = True; i = j + 1; continue
        if c == '"':
            j, buf = i + 1, []
            while j < n and line[j] != '"':
                if line[j] == "\\" and j + 1 < n:
                    buf.append(line[j + 1]); j += 2; continue
                if line.startswith("$(", j) and not line.startswith("$((", j):
                    inner, j = _read_balanced(line, j + 2)
                    subs.append(inner); buf.append("$(...)"); continue
                if line[j] == "`":
                    k = line.find("`", j + 1)
                    k = n if k < 0 else k
                    subs.append(line[j + 1:k]); buf.append("$(...)"); j = k + 1; continue
                buf.append(line[j]); j += 1
            word.append("".join(buf)); in_word = True; i = j + 1; continue
        if line.startswith("$((", i):
            _, i = _read_balanced(line, i + 3)
            if i < n and line[i] == ")":
                i += 1
            word.append("$((...))"); in_word = True; continue
        if line.startswith("$(", i):
            inner, i = _read_balanced(line, i + 2)
            subs.append(inner); word.append("$(...)"); in_word = True; continue
        if c == "`":
            j = line.find("`", i + 1)
            j = n if j < 0 else j
            subs.append(line[i + 1:j]); word.append("$(...)"); in_word = True; i = j + 1; continue
        if c in "<>" and i + 1 < n and line[i + 1] == "(" and not in_word:
            inner, i = _read_balanced(line, i + 2)
            subs.append(inner); toks.append(("W", "<(...)")); continue
        if c in "<>" or (c == "&" and line.startswith("&>", i)):
            fd = "".join(word) if in_word and "".join(word).isdigit() else None
            if fd is None:
                end_word()
            else:
                word, in_word = [], False
            m = re.match(r"&?[<>]+[|&]?(\d+|-)?", line[i:])
            op = m.group(0)
            i += len(op)
            if op.endswith("&") and i < n and line[i] in "0123456789-":
                op += line[i]; i += 1
            needs_target = not re.search(r"&(\d+|-)$", op)
            toks.append(("R", op if needs_target else op + "!"))
            continue
        if c in "|;&()":
            end_word()
            two = line[i:i + 2]
            if two in ("||", "&&", "|&", ";;", "(("):
                op = two if two != "((" else "("
                i += 2 if two != "((" else 1
            else:
                op = c; i += 1
            toks.append(("OP", op)); continue
        word.append(c); in_word = True; i += 1
    end_word()
    return toks, subs


def stages(line, depth=0):
    """Yield lists of words, one per pipeline stage, including stages inside substitutions."""
    toks, subs = tokenize(line)
    cur, skip_target = [], False
    for kind, val in toks:
        if kind == "OP":
            if cur:
                yield cur
            cur, skip_target = [], False
            continue
        if kind == "R":
            skip_target = not val.endswith("!")
            continue
        if skip_target:
            skip_target = False
            continue
        cur.append(val)
    if cur:
        yield cur
    if depth < 4:
        for s in subs:
            yield from stages(s, depth + 1)


DIFF_RESIDUE_RE = re.compile(r"^\d+(,\d+)?[acd]\d+(,\d+)?$")  # `diff` output pasted into histories


def normalize_head(word):
    if word in ("[", "[["):
        return "["
    if word.startswith(("./", "../", "~/")):
        return "<./script>"
    if "/" in word:
        word = word.rstrip("/").rsplit("/", 1)[-1]
    if not word or "$" in word or word.startswith("-") or word.isdigit() or DIFF_RESIDUE_RE.match(word):
        return None
    if not NAME_RE.match(word) or (word.startswith(".") and word != "."):
        return None
    return word


def _skip_wrapper_opts(wrapper, words, i):
    with_arg = WRAPPER_OPTS_WITH_ARG.get(wrapper, set())
    while i < len(words):
        w = words[i]
        if w == "--":
            return i + 1
        if wrapper in ("env",) and ASSIGN_RE.match(w):
            i += 1; continue
        if not w.startswith("-") or w == "-":
            if wrapper == "nice" and re.fullmatch(r"-?\d+", w):
                i += 1; continue
            if wrapper == "timeout":  # first positional is the duration
                return i + 1 if re.fullmatch(r"[\d.]+[smhd]?", w) else i
            return i
        if wrapper == "command" and w in ("-v", "-V"):
            return len(words)
        if wrapper == "nice" and re.fullmatch(r"-\d+", w):
            i += 1; continue
        if w in with_arg:
            i += 2; continue
        i += 1
    return i


def stage_heads(words):
    """Return (heads, ok) for one stage."""
    heads, i, ok = [], 0, True
    while i < len(words):
        w = words[i]
        if ASSIGN_RE.match(w):
            i += 1; continue
        if w in SKIP_THEN_CONTINUE:
            i += 1; continue
        if w in NOT_COUNTED:
            i += 1; continue
        if w.endswith("()") or (i + 1 < len(words) and words[i + 1] == "()"):
            break  # function definition
        if w in ("[[", "]]"):
            heads.append("[")
            break
        head = normalize_head(w)
        if head is None:
            break
        heads.append(head)
        if w in COUNT_THEN_STOP:
            break
        if w in COUNT_THEN_CONTINUE:
            i += 1; continue
        if w in WRAPPERS:
            i = _skip_wrapper_opts(w, words, i + 1)
            continue
        if head == "find":
            for k in range(i + 1, len(words)):
                if words[k] in ("-exec", "-execdir", "-ok", "-okdir") and k + 1 < len(words):
                    end = next((m for m in range(k + 1, len(words)) if words[m] in (";", "+")), len(words))
                    inner, _ = stage_heads(words[k + 1:end])
                    heads.extend(inner)
        break
    return heads, bool(heads)


def pipeline_length(line):
    """1 + number of top-level ``|``/``|&`` operators in the line (substitutions are not counted)."""
    toks, _ = tokenize(line)
    return 1 + sum(1 for kind, val in toks if kind == "OP" and val in ("|", "|&"))


def count_line(line, counter):
    """Count heads of every stage of one command line. Returns (stages_seen, stages_unparsed)."""
    seen = bad = 0
    for st in stages(line):
        if all(w in NOT_COUNTED for w in st):
            continue
        seen += 1
        hs, ok = stage_heads(st)
        if not hs:
            bad += 1
        counter.update(hs)
    return seen, bad


# --------------------------------------------------------------------------------------------- sources

def lines_nl2bash(path):
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.strip():
            yield raw


def lines_hf_history(path):
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.strip():
            yield raw


def lines_aliases(path):
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        for (value,) in con.execute("SELECT value FROM alias ORDER BY id"):
            if value and value.strip():
                yield value
    finally:
        con.close()


def count_uci(path, counter):
    """Token stream; arguments are sanitised to <n>, so each bare-word token is a command head."""
    meta_toks = {"**SOF**", "**EOF**"}
    n_tokens = n_heads = 0
    with zipfile.ZipFile(path) as zf:
        inner = tarfile.open(fileobj=io.BytesIO(zf.read("UNIX_user_data.tar.gz")), mode="r:gz")
        names = sorted(m.name for m in inner.getmembers() if re.search(r"USER\d+\.gz$", m.name))
        for name in names:
            data = gzip.decompress(inner.extractfile(name).read()).decode("latin-1")
            for tok in data.split():
                n_tokens += 1
                if tok in meta_toks or tok.startswith(("-", "<", "$")) or not re.search(r"[A-Za-z]", tok):
                    continue
                head = normalize_head(tok)
                if head and head not in NOT_COUNTED:
                    counter[head] += 1
                    n_heads += 1
    return n_tokens, n_heads


def build_source(sid):
    spec = SOURCES[sid]
    md5 = ZENODO_ALIAS_MD5 if sid == "shell_aliases_github" else None
    path, meta = cached(sid, spec["cache"], spec["url"], md5)
    counter = Counter()
    pipes = None
    stats = {}
    if sid == "uci_unix_user_data":
        n_tokens, n_heads = count_uci(path, counter)
        stats = {"commands_parsed": n_heads, "tokens_read": n_tokens}
    else:
        reader = {"nl2bash": lines_nl2bash, "hf_bash_history": lines_hf_history, "shell_aliases_github": lines_aliases}[sid]
        n_lines = n_lines_ok = n_stages = n_bad = 0
        pipes = Counter()
        for line in reader(path):
            n_lines += 1
            before = sum(counter.values())
            s, b = count_line(line, counter)
            n_stages += s
            n_bad += b
            if sum(counter.values()) > before:
                n_lines_ok += 1
                pipes[pipeline_length(line)] += 1
        stats = {"commands_parsed": n_lines_ok, "input_lines": n_lines, "pipeline_stages": n_stages, "unparsed_stages": n_bad}
    total = sum(counter.values())
    min_count = max(2, round(2e-6 * total))
    kept = {k: v for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])) if v >= min_count}
    dropped = total - sum(kept.values())
    entry = {
        "id": sid, "title": spec["title"], "url": spec["url"], "homepage": spec["homepage"], "license": spec["license"],
        "retrieved": meta["retrieved"], "sha256": meta["sha256"], "kind": spec["kind"], **stats,
        "heads_counted": total, "distinct_heads": len(counter), "min_count": min_count,
        "distinct_kept": len(kept), "heads_dropped_below_min_count": dropped, "note": spec["note"],
    }
    pipes = {str(k): pipes[k] for k in sorted(pipes)} if pipes is not None else None
    return entry, kept, total, pipes


# --------------------------------------------------------------------------------------------- nushell map

def md_table_rows(text, start_heading=None):
    """Yield (line_no, raw_line, cells) for table rows, optionally only in the section under start_heading."""
    active = start_heading is None
    header_seen = False
    for no, line in enumerate(text.splitlines(), 1):
        if start_heading is not None and line.startswith("#"):
            active = line.strip().lstrip("#").strip().rstrip(":") == start_heading
            header_seen = False
            continue
        if not active or not line.lstrip().startswith("|"):
            continue
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
        if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
            continue
        if not header_seen:
            header_seen = True
            yield no, line, cells, True
            continue
        yield no, line, cells, False


def backticked(cell):
    return [m.strip() for m in re.findall(r"`([^`]*)`", re.sub(r"<kbd>.*?</kbd>", "", cell))]


def bash_leading_utility(snippet):
    for st in stages(snippet):
        hs, _ = stage_heads(st)
        hs = [h for h in hs if h not in PLACEHOLDERS and h != "<./script>"]
        if hs:
            return hs[0]
        if any(w not in PLACEHOLDERS for w in st[:1]):
            # first stage had a non-placeholder head that is not a utility (e.g. a path): stop
            if st and st[0] not in PLACEHOLDERS:
                return None
    return None


def split_nu_pipeline(s):
    parts, buf, depth, quote, i = [], [], 0, None, 0
    while i < len(s):
        c = s[i]
        if quote:
            buf.append(c)
            if c == quote:
                quote = None
        elif c in "\"'`":
            quote = c; buf.append(c)
        elif c in "([{":
            depth += 1; buf.append(c)
        elif c in ")]}":
            depth -= 1; buf.append(c)
        elif c == "|" and depth == 0:
            parts.append("".join(buf)); buf = []
        else:
            buf.append(c)
        i += 1
    parts.append("".join(buf))
    return [p.strip() for p in parts]


def nu_commands(snippet, nu_cmds):
    """Return (builtins, other_heads) for a Nushell snippet: the head of each pipeline stage."""
    found, other = [], []
    for st in split_nu_pipeline(snippet):
        st = re.sub(r"^(let|mut)\s+[A-Za-z_][\w-]*\s*=\s*", "", st)
        st = re.sub(r"^(out\+err|o\+e|out|err|o|e)>>?\s*$", "", st)
        words = st.split()
        if not words or words[0][0] in "$\"'[({^0123456789<" or re.fullmatch(r"(out\+err|o\+e|out|err|o|e)>>?", words[0]):
            continue
        for k in (3, 2, 1):
            cand = " ".join(words[:k])
            if cand in nu_cmds:
                found.append(cand)
                break
        else:
            other.append(words[0])
    return found, other


def build_nushell_map(usage_combined, usage_counts):
    pages = {name: cached("nushell", f"nushell_book/{name}", url)[0].read_text(encoding="utf-8") for name, url in NUSHELL_PAGES.items()}
    tree_path, _ = cached("nushell", "nushell_book/tree.json", NUSHELL_TREE_URL)
    tree = json.loads(tree_path.read_text())
    nu_cmds = {p["path"][len("commands/docs/"):-3].replace("_", " ") for p in tree["tree"]
               if p["path"].startswith("commands/docs/") and p["path"].endswith(".md")}
    mapping, evidence, stated_no_builtin = {}, {}, {}

    def add(unix, nu_list, other, page, no, raw, bash_snip, nu_snip):
        ev = {"page": page, "line": no, "row": raw.strip(), "bash": bash_snip, "nu": nu_snip,
              "nu_builtins": nu_list, "nu_other_heads": other}
        evidence.setdefault(unix, []).append(ev)
        if nu_list:
            lst = mapping.setdefault(unix, [])
            for c in nu_list:
                if c not in lst:
                    lst.append(c)
        else:
            stated_no_builtin.setdefault(unix, None)

    # coming_from_bash.md: "Command Equivalents" table, columns Bash | Nu | Task
    prev_bash_alts = []
    for no, raw, cells, is_header in md_table_rows(pages["coming_from_bash.md"], "Command Equivalents"):
        if is_header or len(cells) < 2:
            continue
        bash_alts = backticked(cells[0])
        if cells[0]:
            prev_bash_alts = bash_alts
        else:
            bash_alts = prev_bash_alts  # continuation row of the previous Bash entry
        nu_alts = backticked(cells[1])
        pairs = list(zip(bash_alts, nu_alts)) if len(bash_alts) == len(nu_alts) > 1 else [(b, n) for b in bash_alts for n in nu_alts] or [(b, "") for b in bash_alts]
        for b, nsnip in pairs:
            unix = bash_leading_utility(b)
            if not unix:
                continue
            found, other = nu_commands(nsnip, nu_cmds) if nsnip else ([], [])
            add(unix, found, other, "coming_from_bash.md", no, raw, b, nsnip)

    # nushell_map.md: Nushell | SQL | .Net LINQ | PowerShell | Bash
    bash_col = None
    for no, raw, cells, is_header in md_table_rows(pages["nushell_map.md"]):
        if is_header:
            bash_col = cells.index("Bash")
            continue
        if bash_col is None or len(cells) <= bash_col:
            continue
        nu_parts = [backticked(p) for p in re.split(r"<br\s*/?>", cells[0])]
        nu_parts = [p for p in nu_parts if p]
        bash_parts = [backticked(p) for p in re.split(r"<br\s*/?>", cells[bash_col])]
        bash_parts = [p for p in bash_parts if p]
        if not bash_parts or not nu_parts:
            continue
        if len(bash_parts) == len(nu_parts) > 1:
            pairs = list(zip(bash_parts, nu_parts))
        else:
            pairs = [([b for part in bash_parts for b in part], [n for part in nu_parts for n in part])]
        for bsnips, nsnips in pairs:
            nu_found, nu_other = [], []
            for ns in nsnips:
                f, o = nu_commands(ns, nu_cmds)
                nu_found += [x for x in f if x not in nu_found]
                nu_other += o
            for bs in bsnips:
                unix = bash_leading_utility(bs)
                if unix:
                    add(unix, nu_found, nu_other, "nushell_map.md", no, raw, bs, " / ".join(nsnips))

    stated_no_builtin = sorted(u for u in stated_no_builtin if u not in mapping)
    suggested = {}
    for unix, (cmds, why) in sorted(SUGGESTED.items()):
        if unix in mapping or not cmds:
            continue
        ok = [c for c in cmds if c in nu_cmds or c.startswith("$")]
        if ok:
            suggested[unix] = {"nu": ok, "why": why, "weight": None}

    # coverage of the usage data by the stated map
    coverage = {"method": "share of usage mass whose head utility has an entry in 'map'"}
    tot = sum(usage_combined.values())
    coverage["combined"] = {"commands_in_usage": len(usage_combined),
                            "commands_mapped": sum(1 for c in usage_combined if c in mapping),
                            "mass_mapped": round(sum(v for c, v in usage_combined.items() if c in mapping) / tot, 6) if tot else 0}
    for sid, counts in usage_counts.items():
        t = sum(counts.values())
        coverage[sid] = {"commands_in_usage": len(counts), "commands_mapped": sum(1 for c in counts if c in mapping),
                         "mass_mapped_of_kept": round(sum(v for c, v in counts.items() if c in mapping) / t, 6) if t else 0}
    unmapped = [c for c, _ in sorted(usage_combined.items(), key=lambda kv: (-kv[1], kv[0])) if c not in mapping]
    coverage["top_unmapped_combined"] = unmapped[:60]

    out = {
        "generated_by": "scripts/fetch_command_usage.py",
        "sources": list(NUSHELL_PAGES.values()),
        "checked_without_bash_column": NUSHELL_PAGES_WITHOUT_BASH,
        "nushell_command_list": {"url": NUSHELL_TREE_URL, "count": len(nu_cmds),
                                 "use": "names of commands/docs/*.md; a Nu-side head counts as a builtin only if it is in this list"},
        "method": ("Rows of the 'Command Equivalents' table in coming_from_bash.md and of nushell_map.md (Bash column). "
                   "Bash side: leading utility of each backticked snippet (placeholders like 'command' skipped). Nu side: head "
                   "of each pipeline stage of each backticked snippet that names a Nushell builtin (longest match against the "
                   "command list). When a cell lists the same number of alternatives on both sides they are paired in order; "
                   "otherwise every Bash alternative gets every Nu command of the row. Rows with an empty Bash cell continue the row above."),
        "map": {k: mapping[k] for k in sorted(mapping)},
        "evidence": {k: evidence[k] for k in sorted(evidence)},
        "stated_without_nu_builtin": stated_no_builtin,
        "suggested_not_in_source": suggested,
        "coverage": coverage,
    }
    return out


# --------------------------------------------------------------------------------------------- main

def dump(obj, path):
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-restricted", action="store_true",
                    help="also count the UCI UNIX User Data (research-only licence); it is kept out of 'combined'")
    ap.add_argument("--skip", action="append", default=[], choices=sorted(SOURCES),
                    help="leave a source out (e.g. while its download is incomplete); repeatable")
    args = ap.parse_args()
    sids = [s for s in DEFAULT_SOURCES + (["uci_unix_user_data"] if args.include_restricted else []) if s not in args.skip]

    sources, counts, rel, pipe_lengths, pipe_rel = [], {}, {}, {}, {}
    for sid in sids:
        entry, kept, total, pipes = build_source(sid)
        sources.append(entry)
        counts[sid] = kept
        if pipes is not None:
            pipe_lengths[sid] = pipes
        if not SOURCES[sid].get("restricted"):
            rel[sid] = {k: v / total for k, v in kept.items()}
            if pipes is not None:
                n = sum(pipes.values())
                pipe_rel[sid] = {k: v / n for k, v in pipes.items()}
    pipe_keys = sorted({k for r in pipe_rel.values() for k in r}, key=int)
    pipe_combined = {k: round(sum(r.get(k, 0.0) for r in pipe_rel.values()) / len(pipe_rel), 8) for k in pipe_keys}

    names = sorted({k for r in rel.values() for k in r})
    combined_raw = {k: sum(r.get(k, 0.0) for r in rel.values()) / len(rel) for k in names}
    combined = {k: round(v, 8) for k, v in sorted(combined_raw.items(), key=lambda kv: (-kv[1], kv[0])) if round(v, 8) > 0}
    usage = {
        "generated_by": "scripts/fetch_command_usage.py",
        "counting_rules": __doc__.split("Counting rules", 1)[1].split("\n", 1)[1].strip(),
        "sources": sources,
        "counts": counts,
        "combined_method": (f"Unweighted mean over the {len(rel)} sources {sorted(rel)} of each command's relative frequency "
                            "in that source (count / heads_counted, where heads_counted includes commands dropped below "
                            "min_count; a command absent from a source contributes 0). Each source therefore weighs 1/"
                            f"{len(rel)} regardless of size. Values are rounded to 8 decimals and sum to slightly less than 1. "
                            "Restricted-licence sources are never included."),
        "combined": combined,
        "pipeline_lengths_method": ("Per command line that yielded at least one counted head: 1 + the number of top-level "
                                    "'|' or '|&' operators (pipes inside $(...)/backticks are not counted; lines joined by "
                                    "';'/'&&' count the pipes of all their parts). Only sources made of raw command lines "
                                    "have it; the UCI token stream has no line boundaries and is omitted. "
                                    "pipeline_lengths_combined uses the same rule as combined: unweighted mean over "
                                    f"{sorted(pipe_rel)} of each source's relative frequency."),
        "pipeline_lengths": pipe_lengths,
        "pipeline_lengths_combined": pipe_combined,
    }
    dump(usage, OUT_USAGE)

    nu_map = build_nushell_map(combined, {k: v for k, v in counts.items()})
    dump(nu_map, OUT_MAP)

    # summary
    print(f"wrote {OUT_USAGE.relative_to(ROOT)} and {OUT_MAP.relative_to(ROOT)}\n")
    for s in sources:
        print(f"{s['id']:<22} kind={s['kind']:<14} commands_parsed={s['commands_parsed']:>9,} heads={s['heads_counted']:>9,} "
              f"distinct_kept={s['distinct_kept']:>6,} min_count={s['min_count']}")
    width = 18
    cols = [s["id"] for s in sources]
    print(f"\n{'rank':>4} {'combined':<{width}} {'share':>8}  " + "  ".join(f"{c[:width]:<{width}}" for c in cols))
    tops = {c: list(counts[c].items())[:40] for c in cols}
    for i, (name, v) in enumerate(list(combined.items())[:40]):
        row = f"{i + 1:>4} {name:<{width}} {v:>8.4f}  "
        row += "  ".join(f"{(tops[c][i][0] + ' ' + str(tops[c][i][1])) if i < len(tops[c]) else '':<{width}}" for c in cols)
        print(row)
    print("\npipeline length distribution (share of command lines):")
    for sid, d in pipe_lengths.items():
        n = sum(d.values())
        print(f"  {sid:<22} " + "  ".join(f"{k}:{v / n:.3f}" for k, v in list(d.items())[:6]) + f"  (n={n:,})")
    print(f"  {'combined':<22} " + "  ".join(f"{k}:{v:.3f}" for k, v in list(pipe_combined.items())[:6]))
    cov = nu_map["coverage"]["combined"]
    print(f"\nNushell map: {len(nu_map['map'])} Unix utilities mapped; {cov['commands_mapped']}/{cov['commands_in_usage']} "
          f"commands in 'combined' are mapped, covering {cov['mass_mapped']:.1%} of combined usage mass")
    print("top unmapped:", ", ".join(nu_map["coverage"]["top_unmapped_combined"][:30]))


if __name__ == "__main__":
    main()
