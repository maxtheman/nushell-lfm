#!/usr/bin/env python3
"""Check the WebAssembly build of Nushell against native nu on every sample of the dataset.

    mise run check-wasm
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import notebook

_, defs = notebook.app.run()
cases = []
for sample in defs["samples"]:
    if sample["failure_type"] == "wrong_tool_or_malformed_call":
        continue  # no Nushell source to run
    native = defs["run_nu"](sample["code"], sample["env"])
    cases.append(
        {
            "id": f"{sample['chain_id']}/{sample['form']}/{sample['failure_type']}",
            "env": sample["env"],
            "type": sample["failure_type"],
            "code": sample["code"],
            "ok": native["ok"],
            "stdout": native["stdout"],
            "parse_error": native["parse_error"],
        }
    )
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
    json.dump({"environments": {e["id"]: e["files"] for e in defs["environments"]}, "cases": cases}, handle)
sys.exit(subprocess.run(["node", str(ROOT / "scripts" / "check_nu_wasm.mjs"), handle.name]).returncode)
