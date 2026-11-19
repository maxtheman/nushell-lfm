#!/usr/bin/env python3
"""Run the full protocol for both models and write the results the notebook displays.

Grader:  sweep (rank × learning rate) → adapter-placement ablation → final adapter → guidance table.
Target:  sweep (rank × learning rate) → adapter-placement ablation → full pipeline (P_good, probe, P_bad,
         guidance against P_bad, against a weaker P_good, and against the base model) → browser export →
         attention capture for the 3D view.

    mise run experiments            # everything (a few hours on a 24 GB machine)
    mise run experiments -- --only grader
"""

import argparse
import functools
import json
import sys
from pathlib import Path

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GRADER = {"ranks": [8, 16], "learning_rates": [1e-4, 2e-4, 4e-4], "n_sweep": 160, "n_dev": 100, "n_full": 2400, "n_dev_final": 400, "epochs": 2}
TARGET = {
    "ranks": [8, 16], "learning_rates": [1e-4, 2e-4, 4e-4], "n_sweep": 200, "n_sweep_eval": 24,
    "full": {
        "n_good": 10_000, "n_bad": 160, "n_probe": 48, "n_eval": 40, "n_loss": 24, "epochs": 2,
        "guidance": [
            ["P_bad", 0.25, 0.1], ["P_bad", 0.5, 0.1], ["P_bad", 1.0, 0.1], ["P_bad", 0.5, 0.03], ["P_bad", 0.5, 0.3],
            ["weak P_good", 0.5, 0.1], ["weak P_good", 1.0, 0.1], ["base", 0.5, 0.1],
        ],
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["grader", "grader-final", "target", "unused", "weak", "export", "activation"])
    args = parser.parse_args()

    import notebook

    _, defs = notebook.app.run()
    dataset, models, folders = defs["dataset"], defs["MODELS_DIR"], defs["MODEL_FOLDERS"]
    defs["save_dataset"](dataset, {"chain_summary": defs["chain_summary"], "priors": defs["PRIOR_INFO"]})
    print(f"dataset {dataset['fingerprint']}: {len(defs['verified_chains'])} chains in {len(defs['environments'])} environments, "
          f"held out {dataset['dev_envs']}; requests {dataset['requests_by']}")
    grader_path, target_path = str(models / folders["grader"]), str(models / folders["target"])

    if args.only in (None, "grader"):
        result = defs["run_grader_experiments"](grader_path, dataset, GRADER, print)
        print("grader chosen:", result["chosen"], json.dumps(result["final"]["metrics"], indent=1))
        print("grader guidance:", json.dumps(result["guidance"], indent=1))
    if args.only == "grader-final":  # retrain only the final adapter, with the settings the sweep and ablation chose
        file = defs["GRADER_DIR"] / "experiments.json"
        experiments = json.loads(file.read_text())
        chosen = next(r["config"] for r in experiments["sweep"] + experiments["placement_ablation"] if r["config"]["name"] == experiments["chosen"])
        final = defs["run_grader_sweep"]([{**chosen, "name": "final-" + chosen["name"], "epochs": GRADER["epochs"]}], dataset, GRADER["n_full"], GRADER["n_dev_final"], print)
        run = final["runs"][0]
        experiments.update(
            sizes=GRADER, guidance=run["guidance"],
            final={"config": run["config"], "metrics": run["metrics"], "seconds": run["seconds"], "trainable_parameters": run["trainable_parameters"], "train_rows": run["train_rows"]},
        )
        file.write_text(json.dumps(experiments, indent=1), encoding="utf-8")
        print("grader final:", json.dumps(run["metrics"], indent=1))
    if args.only in (None, "target"):
        base = {"model_path": target_path, "grader_model_path": grader_path, "batch": 4, "seed": 73, "probe_share": 0.15, "max_new": 220}
        result = defs["run_target_experiments"](base, dataset, TARGET, print)
        print("target chosen:", result["chosen"])
        print("execution:", json.dumps(result["final"]["execution"], indent=1))
        print("reference log-probability under guidance:", json.dumps(result["final"]["reference_logp"], indent=1))
    if args.only in (None, "target", "unused"):
        rest = defs["score_saved_target"](dataset, print)
        print("P_good on the held-out requests not used anywhere else:", json.dumps({k: v for k, v in rest.items() if k != "wrong"}, indent=1))
    if args.only == "weak":  # only needed for adapters trained before the weak snapshot was saved
        defs["rebuild_weak_target"](dataset, print)
    if args.only in (None, "export"):
        report = json.loads((defs["TARGET_DIR"] / "report.json").read_text())
        rows = sorted(dataset["target"]["dev"], key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))
        merged = defs["export_for_browser"](target_path, defs["TARGET_DIR"] / "adapter_good.safetensors", report["config"]["rank"], report["config"]["alpha"], print)
        checked = defs["check_browser_export"](rows[:24], defs["score_call"], defs["target_prompt"], report["config"]["max_new"], log=print)
        defs["write_browser_requests"](rows, defs["environments"])
        (defs["BROWSER_DIR"] / "export.json").write_text(
            json.dumps({"merge": merged, "check": checked, "mlx_P_good": report["execution"]["P_good"]}, indent=1), encoding="utf-8"
        )
    if args.only in (None, "activation"):
        import activation_view

        examples = activation_view.capture_activations(
            defs, activation_view.pick_rows(defs, 4), max_examples=4, stats_rows=activation_view.scored_rows(defs, 40), log=print
        )
        activation_view.save_payload(ROOT / "runs" / "activation" / "examples.json", examples, activation_view.model_layout(defs))
        print(f"activation: {len(examples)} examples, both models with and without their adapters")


if __name__ == "__main__":
    main()
