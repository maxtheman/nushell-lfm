#!/usr/bin/env python3
"""Capture two held-out rows and check what the 3D view will draw (loads the model; about a minute).

    mise exec -- .venv/bin/python scripts/check_model_view.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nushell_lfm import model_view as mv  # noqa: E402
from nushell_lfm.pipeline import build  # noqa: E402


def main() -> int:
    layout = mv.model_layout()
    examples = mv.capture_activations(mv.pick_rows(build()["dataset"], 2), max_examples=2)
    attn_layers = [i for i, t in enumerate(layout["layer_types"]) if t != "conv"]
    failures = []
    print(f"\nlayer types: {len(layout['layer_types'])} layers, attention at {attn_layers}, conv kernel {layout['conv_window']}")
    for ex in examples:
        st, v = ex["stats"], ex["lfm3"]
        t = len(v["tokens"])
        print(f"\n== {ex['id']}\n   request: {ex['request']!r}")
        print(f"   recomputed attention reproduces the module output to {st['verify_rel']:.3g} (relative); rows sum to 1 within {st['row_sum_err']:.2g}; masks {st['mask_kinds']}")
        if st["verify_rel"] > 2e-2 or st["pregate_rel"] > 2e-2 or st["row_sum_err"] > 1e-3:
            failures.append(f"{ex['id']}: recomputation does not reproduce the module output")
        print(f"   {v['n_total']} tokens, drawn {v['n_in']} request + {v['n_out']} call; adapter {v['lora']}")
        if sorted(int(k) for k in v["attn"]["base"]) != attn_layers or not v["attn"]["lora"]:
            failures.append(f"{ex['id']}: attention layers missing")
            continue
        moved = np.array(v["stream"]["moved"])
        if moved.shape != (len(layout["layer_types"]), t):
            failures.append(f"{ex['id']}: residual stream has shape {moved.shape}")
        print(f"   residual stream moved by the LoRA: mean {moved.mean():.3f}, largest {moved.max():.3f} of its size (layer {int(moved.max(axis=1).argmax()) + 1})")
        changes = []
        for li in attn_layers:
            b, l = v["attn"]["base"][str(li)], v["attn"]["lora"][str(li)]
            changes.append(float(np.abs(mv._unpack_lower(l["m"], l["s"], t)[4] - mv._unpack_lower(b["m"], b["s"], t)[4]).max()))
        print("   largest change in attention weight per attention layer: " + " ".join(f"{c:.3f}" for c in changes))
        if max(changes) <= 0:
            failures.append(f"{ex['id']}: the LoRA does not change attention")
        w = v["written"]
        n = len(w["base"]["top"])
        print(f"   puts the reference token first: base {sum(p >= 0.5 for p in w['base']['p_reference'])}/{n}, with LoRA {sum(p >= 0.5 for p in w['lora']['p_reference'])}/{n}")
    size = len(json.dumps({"examples": examples, **layout}, separators=(",", ":")))
    print(f"\npayload JSON for {len(examples)} examples: {size / 1024:.1f} KiB")
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
