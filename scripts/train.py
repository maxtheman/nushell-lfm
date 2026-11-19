#!/usr/bin/env python3
"""Build the dataset, train the LFM3 adapter and write the results the notebook and the demo show.

    mise run smoke                       # 8 rows, 4 held-out requests; written to runs/smoke (about two minutes)
    mise run train                       # the adapter at the chosen settings, scored on every held-out request
    mise run train -- --preset protocol  # sweep over rank and learning rate, placement ablation, final adapter
    mise run train -- --only export      # one stage: train, score, time, export or activation

Stages after training: `score` executes the saved adapter's calls on the held-out requests it was not compared
on, `export` adds the adapter to the 4-bit ONNX build for the browser and checks it, `activation` captures what
the 3D view draws.
"""

import argparse
import functools
import json
import sys
from pathlib import Path

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = {"batch": 4, "seed": 73, "probe_share": 0.15, "max_new": 220}
CHOSEN = {"rank": 8, "alpha": 16.0, "lr": 2e-4, "placement": "all_linear"}  # what the protocol picked on dataset a23bbb10902d
FULL = {"n_good": 10_000, "n_eval": 40, "n_loss": 24, "epochs": 2}
SWEEP = {"ranks": [8, 16], "learning_rates": [1e-4, 2e-4, 4e-4], "n_sweep": 200, "n_sweep_eval": 24, "full": FULL}
SMOKE = {"n_good": 8, "n_eval": 4, "n_loss": 4, "epochs": 1, "score_all": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=["smoke", "standard", "protocol"], default="standard")
    parser.add_argument("--only", choices=["train", "score", "time", "export", "activation"])
    args = parser.parse_args()

    from nushell_lfm import model_view
    from nushell_lfm.core import MODEL_FOLDERS, RUNS_DIR, model_path
    from nushell_lfm.dataset import save_dataset
    from nushell_lfm.export import BROWSER_DIR, check_browser_export, export_for_browser, write_browser_requests
    from nushell_lfm.pipeline import build
    from nushell_lfm.target import TARGET_DIR, run_target_experiments, run_target_pipeline, score_call, score_saved_target, spread, target_prompt, time_saved_target

    built = build()
    dataset = built["dataset"]
    save_dataset(dataset, built["settings"])
    print(f"dataset {dataset['fingerprint']}: {len(built['chains'])} chains in {len(built['environments'])} environments, "
          f"held out {dataset['dev_envs']}; requests {dataset['requests_by']}")
    base = {**BASE, "model_path": model_path(MODEL_FOLDERS["target"])}

    if args.preset == "smoke":
        report = run_target_pipeline({**base, **CHOSEN, **SMOKE, "out_dir": str(RUNS_DIR / "smoke")}, dataset, print)
        print("smoke:", json.dumps(report["execution"], indent=1))
        return 0
    if args.only in (None, "train"):
        if args.preset == "protocol":
            result = run_target_experiments(base, dataset, SWEEP, print)
            print("chosen:", result["chosen"], json.dumps(result["final"]["execution"], indent=1))
        else:
            report = run_target_pipeline({**base, **CHOSEN, **FULL}, dataset, print)
            print("execution:", json.dumps(report["execution"], indent=1))
    if args.only == "score":
        rest = score_saved_target(dataset, print)
        print("on the held-out requests used for nothing else:", json.dumps({k: v for k, v in rest.items() if k != "wrong"}, indent=1))
    if args.only == "time":
        print("seconds per request:", json.dumps(time_saved_target(dataset, print), indent=1))
    if args.only in (None, "export"):
        report = json.loads((TARGET_DIR / "report.json").read_text())
        rows = spread(dataset["target"]["dev"])
        merged = export_for_browser(base["model_path"], TARGET_DIR / "adapter_good.safetensors", report["config"]["rank"], report["config"]["alpha"], print)
        checked = check_browser_export(rows[:24], score_call, target_prompt, report["config"]["max_new"], log=print)
        write_browser_requests(rows, built["environments"])
        (BROWSER_DIR / "export.json").write_text(json.dumps({"merge": merged, "check": checked, "mlx_P_good": report["execution"]["P_good"]}, indent=1), encoding="utf-8")
    if args.only in (None, "activation"):
        examples = model_view.capture_activations(model_view.pick_rows(dataset, 4), log=print)
        model_view.save_payload(RUNS_DIR / "activation" / "examples.json", examples, model_view.model_layout())
        print(f"activation: {len(examples)} examples, with and without the adapter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
