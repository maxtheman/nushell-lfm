#!/usr/bin/env python3
"""Count how often each Nushell command appears in the example code of the Nushell book and cookbook.

This is the Nushell-native prior for the chain generator: which commands Nushell's own documentation
reaches for, and how long its example pipelines are. Unlike data/command_usage.json it involves no
translation from another shell.

    mise run fetch-nu-examples

Reads assets/nu_commands.json (the command list of the installed nu) and writes
data/nushell_examples_usage.json. Downloads are cached under data/usage_cache/nushell_docs/ and pinned
to one commit of nushell/nushell.github.io, so a rerun gives the same file.
"""

import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMIT = "c4b4da18"  # nushell/nushell.github.io
FOLDERS = ("book/", "cookbook/")
CACHE = ROOT / "data" / "usage_cache" / "nushell_docs"
OUT = ROOT / "data" / "nushell_examples_usage.json"


def fetch(url: str, path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "dataset-builder"}), timeout=60) as response:
            path.write_bytes(response.read())
    return path.read_text(encoding="utf-8", errors="replace")


def split_top(code: str, sep: str) -> list[str]:
    """Split at `sep` outside quotes and brackets."""
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


def main():
    names = sorted((c["name"] for c in json.loads((ROOT / "assets" / "nu_commands.json").read_text())), key=len, reverse=True)
    tree = json.loads(fetch(f"https://api.github.com/repos/nushell/nushell.github.io/git/trees/{COMMIT}?recursive=1", CACHE / "tree.json"))
    full = tree["sha"]
    pages = sorted(e["path"] for e in tree["tree"] if e["path"].endswith(".md") and e["path"].startswith(FOLDERS))
    counts, lengths, blocks, lines_seen = {}, {}, 0, 0
    for page in pages:
        text = fetch(f"https://raw.githubusercontent.com/nushell/nushell.github.io/{full}/{page}", CACHE / page)
        for block in re.findall(r"```(?:nu|nushell)\n(.*?)```", text, re.S):
            blocks += 1
            for line in block.splitlines():
                line = line.strip()
                if line.startswith(("> ", "$ ", "~> ")):  # a prompt marks an input line; the rest is output
                    line = line.split(" ", 1)[1]
                elif "│" in line or "─" in line or line.startswith(("#", "╭", "╰", "├")) or not line:
                    continue
                stages = [s.strip() for s in split_top(line, "|") if s.strip()]
                heads = []
                for stage in stages:
                    stage = re.sub(r"^(let|mut|const) \w+ = \(?", "", stage)
                    head = next((n for n in names if stage == n or stage.startswith(n + " ")), None)
                    if head:
                        heads.append(head)
                if not heads:
                    continue
                lines_seen += 1
                lengths[len(stages)] = lengths.get(len(stages), 0) + 1
                for head in heads:
                    counts[head] = counts.get(head, 0) + 1
    total = sum(counts.values())
    seen = sum(lengths.values())
    OUT.write_text(
        json.dumps(
            {
                "generated_by": "scripts/fetch_nushell_examples.py",
                "source": {
                    "title": "Nushell book and cookbook, example code blocks",
                    "url": f"https://github.com/nushell/nushell.github.io/tree/{full}",
                    "license": "MIT (repository LICENSE)",
                    "pages": len(pages), "code_blocks": blocks, "command_lines": lines_seen,
                    "note": "Counts the command at the head of each pipeline stage in nu code blocks. Documentation examples favour what the authors chose to teach; this is a prior for what is idiomatic, not a measurement of what users run.",
                },
                "counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                "share": {k: round(v / total, 6) for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))},
                "pipeline_lengths": {str(k): round(v / seen, 6) for k, v in sorted(lengths.items())},
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    top = ", ".join(f"{k} {v}" for k, v in list(sorted(counts.items(), key=lambda kv: -kv[1]))[:25])
    print(f"{len(pages)} pages, {blocks} code blocks, {lines_seen} command lines, {len(counts)} commands seen\ntop: {top}")
    print("pipeline lengths:", {k: round(v / seen, 3) for k, v in sorted(lengths.items())[:6]})


if __name__ == "__main__":
    main()
