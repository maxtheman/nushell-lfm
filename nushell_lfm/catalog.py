"""Every built-in Nushell command with its input and output types, as the generator sees it."""

import json

from .core import ROOT

NU_COMMANDS = json.loads((ROOT / "assets" / "nu_commands.json").read_text(encoding="utf-8"))

def signature(command: dict) -> str:
    pairs = "; ".join(f"{io['input']} → {io['output']}" for io in command["input_output"][:4])
    params = " ".join(("<{}>" if p["required"] else "[{}]").format(p["name"]) for p in command["params"] if not p["name"].startswith("--"))
    return f"{command['name']} {params}".strip() + f"   ({pairs})"

COMMAND_CATALOG = [{"name": c["name"], "category": c["category"], "usage": c["description"], "syntax": signature(c)} for c in NU_COMMANDS]
ALL_CATEGORIES = ["search", "text_manipulation", "filesystem", "tables_records", "pipelines_recovery"]
CATEGORY_NOTES = {
    "search": "finding files and lines across the project",
    "text_manipulation": "reading, splitting, parsing and reshaping text and logs",
    "filesystem": "listing, sizing and naming paths",
    "tables_records": "filtering and summarising JSON and TOML data",
    "pipelines_recovery": "offering choices or a default when a path or key is missing",
}
