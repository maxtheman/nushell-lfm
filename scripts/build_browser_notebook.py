#!/usr/bin/env python3
"""Export the notebook so it runs in the browser (WebAssembly), into dist/notebook.

    mise run build-notebook

In the page, Python runs in Pyodide: environments and chains are generated there, Nushell is its WebAssembly
build (public/nu-wasm), and the buttons that need a model replay the saved runs. This script packs what the
page needs into public/bundle.zip (the nushell_lfm package, the data files, the saved results, and the
Nushell results of the default settings so the first load does not have to recompute them) and then runs
`marimo export html-wasm`.
"""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nushell_lfm import environments  # noqa: E402
from nushell_lfm.core import RUNS_DIR  # noqa: E402
from nushell_lfm.pipeline import build  # noqa: E402


class Seen(dict):
    """The Nushell result cache, remembering which entries a build reads."""

    def __init__(self, *args):
        super().__init__(*args)
        self.used = set()

    def __contains__(self, key):
        found = super().__contains__(key)
        if found:
            self.used.add(key)
        return found


def main() -> int:
    environments._cache = Seen(environments._cache)
    built = build()
    cache = {k: environments._cache[k] for k in environments._cache.used}
    print(f"dataset {built['dataset']['fingerprint']}: {len(cache)} cached Nushell results kept of {len(environments._cache)}")

    report = json.loads((RUNS_DIR / "target" / "report.json").read_text())
    report = {k: v for k, v in report.items() if k not in ("reference_logp", "probe")}
    report["history"] = [h for h in report["history"] if h["adapter"] == "P_good"]
    report["samples"] = [{k: v for k, v in s.items() if not k.startswith("guided")} for s in report["samples"]]
    report["execution"] = {k: v for k, v in report["execution"].items() if k in ("base", "P_good")}
    if "unused_held_out" in report:
        report["unused_held_out"] = {k: v for k, v in report["unused_held_out"].items() if k != "wrong"}

    out = ROOT / "public" / "bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted((ROOT / "nushell_lfm").rglob("*.py")):
            z.write(path, path.relative_to(ROOT))
        for name in ("assets/nu_commands.json", "data/command_usage.json", "data/unix_to_nushell.json", "data/nushell_examples_usage.json", "data/model_prices.json", "scripts/eval_endpoint.py", "assets/usage.gif"):
            z.write(ROOT / name, name)
        z.writestr("runs/nu-cache.json", json.dumps(cache))
        z.writestr("runs/target/report.json", json.dumps(report))
        for name in ("target/experiments.json", "generated/model_requests.json", "activation/examples.json"):
            z.write(RUNS_DIR / name, f"runs/{name}")
        for path in sorted((RUNS_DIR / "providers").glob("*.json")):
            saved = json.loads(path.read_text())
            z.writestr(f"runs/providers/{path.name}", json.dumps({k: v for k, v in saved.items() if k != "rows"}))
    print(f"wrote {out.relative_to(ROOT)}: {out.stat().st_size / 1e6:.1f} MB")

    target = ROOT / "dist" / "notebook"
    done = subprocess.run([sys.executable, "-m", "marimo", "export", "html-wasm", str(ROOT / "notebook.py"), "-o", str(target), "--mode", "run", "--no-show-code", "-f"], cwd=ROOT)
    return done.returncode


if __name__ == "__main__":
    sys.exit(main())
