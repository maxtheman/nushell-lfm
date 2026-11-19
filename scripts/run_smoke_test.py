#!/usr/bin/env python3
"""Run the notebook's pipeline from the command line: dataset, grader, target.

The notebook defines every step as a plain function; this script calls the same functions, so a
passing run here means the notebook's own code works end to end.

    mise run smoke        # smallest sizes, a few minutes
    mise run standard     # the notebook's standard preset
"""

import argparse
import functools
import json
import sys
from collections import Counter
from pathlib import Path

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PRESETS = {
    "smoke": {
        "grader": {"n_train": 20, "n_dev": 20, "epochs": 1},
        "target": {"n_good": 8, "n_bad": 8, "n_probe": 4, "n_eval": 4, "n_loss": 4, "epochs": 1, "guidance": [["P_bad", 0.5, 0.1], ["weak P_good", 0.5, 0.1]]},
    },
    "standard": {
        "grader": {"n_train": 400, "n_dev": 120, "epochs": 2},
        "target": {
            "n_good": 10_000, "n_bad": 120, "n_probe": 36, "n_eval": 40, "n_loss": 24, "epochs": 2,
            "guidance": [["P_bad", 0.25, 0.1], ["P_bad", 0.5, 0.1], ["P_bad", 1.0, 0.1], ["P_bad", 0.5, 0.03], ["P_bad", 0.5, 0.3], ["weak P_good", 0.5, 0.1], ["weak P_good", 1.0, 0.1], ["base", 0.5, 0.1]],
        },
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=PRESETS, default="smoke")
    parser.add_argument("--skip-grader", action="store_true", help="Reuse the grader adapter already in runs/grader/.")
    args = parser.parse_args()
    sizes = PRESETS[args.preset]

    import notebook

    _, defs = notebook.app.run()
    dataset, models, folders = defs["dataset"], defs["MODELS_DIR"], defs["MODEL_FOLDERS"]
    defs["save_dataset"](dataset)
    grader_rows = dataset["grader"]
    print(f"environments: {len(defs['environments'])} · held out: {', '.join(dataset['dev_envs'])}")
    print(f"chains: {len(defs['verified_chains'])} verified · in held-out environments: {len(dataset['dev_chains'])}")
    print(f"grader rows: {len(grader_rows['train'])} train / {len(grader_rows['dev'])} dev")
    print("failure types (train):", dict(Counter(r["failure_type"] for r in grader_rows["train"])))
    print(f"target rows: {len(dataset['target']['train'])} train / {len(dataset['target']['dev'])} dev\n")

    grader_path = str(models / folders["grader"])
    if not args.skip_grader:
        config = {
            "name": "all_linear-r8", "model_path": grader_path, "placement": "all_linear", "rank": 8, "alpha": 16.0,
            "lr": 2e-4, "dropout": 0.05, "clip": 1.0, "batch": 4, "epochs": sizes["grader"]["epochs"], "seed": 73,
            "guidance_weights": [0.0, 1.0, 1.5],
        }
        report = defs["run_grader_sweep"]([config], dataset, sizes["grader"]["n_train"], sizes["grader"]["n_dev"], print, keep=False)
        print("grader metrics:", json.dumps(report["runs"][0]["metrics"], indent=2), "\n")

    report = defs["run_target_pipeline"](
        {
            **sizes["target"], "model_path": str(models / folders["target"]), "grader_model_path": grader_path,
            "rank": 16, "alpha": 32.0, "lr": 2e-4, "batch": 4, "seed": 73, "probe_share": 0.15, "max_new": 220,
            "out_dir": str(ROOT / "runs" / "smoke"),  # never replaces the real adapters and reports
        },
        dataset,
        print,
    )
    print("\nexecution on held-out requests:", json.dumps(report["execution"], indent=2))
    print("probe:", json.dumps(report["probe"], indent=2))
    print("loss:", json.dumps(report["loss"], indent=2))
    print(f"\nDone in {report['seconds']} s (target stage), peak memory {report['peak_memory_gb']} GB.")


if __name__ == "__main__":
    main()
