#!/usr/bin/env python3
"""Check the WebAssembly build of Nushell against native nu on the reference call of every chain.

    mise run check-wasm
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nushell_lfm.environments import run_nu
from nushell_lfm.pipeline import build

built = build()
cases = []
for chain in built["chains"]:
    native = run_nu(chain["code"], chain["env"])
    cases.append({"id": chain["chain_id"], "env": chain["env"], "type": "valid", "code": chain["code"], "ok": native["ok"], "stdout": native["stdout"], "parse_error": native["parse_error"]})
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
    json.dump({"environments": {e["id"]: e["files"] for e in built["environments"]}, "cases": cases}, handle)
sys.exit(subprocess.run(["node", str(ROOT / "scripts" / "check_nu_wasm.mjs"), handle.name]).returncode)
