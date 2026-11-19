#!/usr/bin/env python3
"""Two more ways of steering P_good, scored by execution on every held-out request.

1. Away from itself with the request removed: what the request contributes is added again, weighted.
2. Away from a P_bad trained on P_good's real mistakes, collected by cross-fitting: five adapters, each
   trained on four fifths of the training chains and asked for the calls of the remaining fifth.

    .venv/bin/python experiments/run_guidance_followup.py                 # everything (about four hours)
    .venv/bin/python experiments/run_guidance_followup.py --stage request # only the first part (under an hour)
    .venv/bin/python experiments/run_guidance_followup.py --stage table   # print the table from what is saved

Results are kept in runs/target/decoders.json and runs/target/crossfit.json; finished parts are not repeated.
"""

import argparse
import functools
import json
import random
import sys
from pathlib import Path

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

# Held-out requests of dataset edd14fe576df that do not contain what their reference call needs (read by hand
# from P_good's wrong calls; see notes/lab-notes.md). Matched by how the request starts.
FAULTY = [
    "List what is in this folder; then keep the first 2;",
    "Find the paths in the journal directory; for each file, count the occurrences of the word checksum",
    "Imagine grabbing the master recipe card (conf/settings.toml)",
    "Rewrite the request in an analogy style, same domain",
    "Imagine opening the master inventory ledger for the storage shelf",
    "I’m like a treasure hunter searching every attic and basement",
]
REQUEST_ARMS = [
    {"name": "P_good"},
    {"name": "away from no-request, w 0.25", "blank_request": True, "weight": 0.25, "floor": 0.1},
    {"name": "away from no-request, w 0.5", "blank_request": True, "weight": 0.5, "floor": 0.1},
    {"name": "away from no-request, w 1.0", "blank_request": True, "weight": 1.0, "floor": 0.1},
    {"name": "away from no-request, w 0.5, floor 0.03", "blank_request": True, "weight": 0.5, "floor": 0.03},
]
BAD_ARMS = [
    {"name": "away from cross-fit P_bad, w 0.5", "adapter": "bad_crossfit", "weight": 0.5, "floor": 0.1},
    {"name": "away from cross-fit P_bad, w 1.0", "adapter": "bad_crossfit", "weight": 1.0, "floor": 0.1},
    {"name": "away from cross-fit P_bad, w 0.5, floor 0.03", "adapter": "bad_crossfit", "weight": 0.5, "floor": 0.03},
    {"name": "away from cross-fit + generated P_bad, w 0.5", "adapter": "bad_crossfit_mixed", "weight": 0.5, "floor": 0.1},
]


def table(saved: dict, dataset: dict) -> None:
    speech = {r["id"]: r["speech"] for r in dataset["target"]["dev"]}
    faulty = {i for i, text in speech.items() if any(text.startswith(start) for start in FAULTY)}
    base = {r["id"]: r["correct"] for r in saved["arms"]["P_good"]["rows"]} if "P_good" in saved["arms"] else None
    print(f"\n{len(faulty)} of {len(speech)} held-out requests are marked faulty and left out of the second column")
    print(f"\n{'decoder':<48} {'all':>10} {'faulty left out':>18} {'fixed':>6} {'broke':>6}")
    for name, arm in saved["arms"].items():
        rows = arm["rows"]
        kept = [r for r in rows if r["id"] not in faulty]
        fixed = sum(r["correct"] and not base[r["id"]] for r in rows) if base else 0
        broke = sum(base[r["id"]] and not r["correct"] for r in rows) if base else 0
        right, right_kept = sum(r["correct"] for r in rows), sum(r["correct"] for r in kept)
        print(f"{name:<48} {right:>4}/{len(rows)} {right_kept:>10}/{len(kept)} ({right_kept / len(kept):.1%}) {fixed:>5} {broke:>6}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["request", "crossfit", "bad", "table", "all"], default="all")
    args = parser.parse_args()

    from defs import load_defs

    defs = load_defs()
    dataset = defs["dataset"]
    target_dir = defs["TARGET_DIR"]
    print(f"dataset {dataset['fingerprint']}")

    if args.stage in ("request", "all"):
        defs["score_decoders"](dataset, REQUEST_ARMS, print)
    if args.stage in ("crossfit", "all"):
        found = defs["crossfit_wrong_calls"](dataset, 5, print)
        print("cross-fit:", {k: f"{v['correct']}/{v['rows']}" for k, v in found["parts"].items()})
    if args.stage in ("bad", "all"):
        found = json.loads((target_dir / "crossfit.json").read_text(encoding="utf-8"))
        train = {r["id"]: r for r in dataset["target"]["train"]}
        wrong = [w for part in found["parts"].values() for w in part["wrong"] if w["completion"].endswith("</tool_call>")]
        own = [{**train[w["id"]], "completion": w["completion"]} for w in wrong]
        print(f"P_good's own wrong calls on training requests it was not fitted on: {len(own)} of {sum(p['rows'] for p in found['parts'].values())}")
        if not (target_dir / "adapter_bad_crossfit.safetensors").exists():
            defs["train_bad_adapter"](own, "bad_crossfit", print)
        if not (target_dir / "adapter_bad_crossfit_mixed.safetensors").exists():
            setting = {r["chain_id"]: r for r in dataset["target"]["train"]}
            pool = [g for g in dataset["grader"]["train"] if g["is_broken"] and g["chain_id"] in setting]
            random.Random(73).shuffle(pool)
            generated = [{**setting[g["chain_id"]], "id": g["id"], "speech": g["prompt"], "completion": g["completion"]} for g in pool[: max(0, 160 - len(own))]]
            print(f"mixed P_bad: {len(own)} own + {len(generated)} generated breaks")
            defs["train_bad_adapter"](own + generated, "bad_crossfit_mixed", print)
        defs["score_decoders"](dataset, BAD_ARMS, print)
    path = target_dir / "decoders.json"
    if path.exists():
        table(json.loads(path.read_text(encoding="utf-8")), dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
