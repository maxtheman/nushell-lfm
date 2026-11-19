"""Capture two held-out rows with both models and check what the 3D view will draw.

Run: mise exec -- .venv/bin/python scripts/check_activation_view.py
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import activation_view as av  # noqa: E402
import notebook  # noqa: E402


def square(entry: dict, t: int) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(entry["m"]), dtype=np.uint8).astype(np.float32)
    return raw.reshape(t, t) / 255.0 * entry["s"][0]


def main() -> int:
    _, defs = notebook.app.run()
    layout = av.model_layout(defs)
    rows = av.pick_rows(defs, 2)
    examples = av.capture_activations(defs, rows, max_examples=2, stats_rows=av.scored_rows(defs, 6))
    attn_layers = [i for i, t in enumerate(layout["layer_types"]) if t != "conv"]
    failures = []
    print(f"\nlayer types: {len(layout['layer_types'])} layers, attention at {attn_layers}, conv kernel {layout['conv_window']}")
    for ex in examples:
        st = ex["stats"]
        print(f"\n== {ex['id']}\n   request: {ex['request']!r}")
        print(f"   recomputed attention reproduces the module output to {st['verify_rel']:.3g} (relative); rows sum to 1 within {st['row_sum_err']:.2g}; masks {st['mask_kinds']}")
        if st["verify_rel"] > 2e-2 or st["pregate_rel"] > 2e-2 or st["row_sum_err"] > 1e-3:
            failures.append(f"{ex['id']}: recomputation does not reproduce the module output")
        for name in ("lfm3", "d1"):
            v = ex[name]
            if v is None:
                failures.append(f"{ex['id']}: no {name} capture")
                continue
            t = len(v["tokens"])
            print(f"   {name}: {v['n_total']} tokens, drawn {v['n_in']} request + {v['n_out']} call; adapter {v['lora']}")
            if sorted(int(k) for k in v["attn"]["base"]) != attn_layers or not v["attn"]["lora"]:
                failures.append(f"{ex['id']} {name}: attention layers missing")
                continue
            moved = np.array(v["stream"]["moved"])
            if moved.shape != (len(layout["layer_types"]), t):
                failures.append(f"{ex['id']} {name}: residual stream has shape {moved.shape}")
            print(f"      residual stream moved by the LoRA: mean {moved.mean():.3f}, largest {moved.max():.3f} of its size (layer {int(moved.max(axis=1).argmax()) + 1})")
            changes = []
            for li in attn_layers:
                b, l = v["attn"]["base"][str(li)], v["attn"]["lora"][str(li)]
                if name == "d1":
                    change = float(np.abs(square(l, t) - square(b, t)).max())
                else:
                    change = float(np.abs(av._unpack_lower(l["m"], l["s"], t)[4] - av._unpack_lower(b["m"], b["s"], t)[4]).max())
                changes.append(change)
            print("      largest change in attention weight per attention layer: " + " ".join(f"{c:.3f}" for c in changes))
            if max(changes) <= 0:
                failures.append(f"{ex['id']} {name}: the LoRA does not change attention")
        w = ex["lfm3"]["written"]
        if w:
            n = len(w["base"]["top"])
            print(f"   lfm3 puts the reference token first: base {sum(p >= 0.5 for p in w['base']['p_reference'])}/{n}, with LoRA {sum(p >= 0.5 for p in w['lora']['p_reference'])}/{n}")
        g = ex["lfm3"]["guidance"]
        if g:
            margins = np.array(g["margin"])
            print(f"   guidance: P_good's margin median {np.median(margins):.2f} nats; a second token within floor 0.1 at {sum(a > 0 for a in g['alternatives']['0.1'])}/{len(margins)} tokens")
            for other, rec in g["against"].items():
                print(f"      away from {other}: mean |push| {np.abs(rec['push']).mean():.3f} nats; tokens changed at w 0.5, floor 0.1: {len(rec['flips']['0.5|0.1'])}")
        else:
            failures.append(f"{ex['id']}: no guidance record")
        if ex["d1"]:
            for variant in ("base", "lora"):
                a = ex["d1"]["answer"][variant]
                print(f"   d1 {variant:<4s}: {a['verdict']:<6s} P(broken) {a['p_broken']:.3f}  top3 {a['top3']}")
            if ex["d1"]["broken"]:
                b = ex["d1"]["broken"]
                print(f"   d1 on the broken variant (true {b['true_label']}): base {b['base']['verdict']} {b['base']['type_if_broken']}, with LoRA {b['lora']['verdict']} {b['lora']['type_if_broken']}")
    totals = json.loads((Path(defs["TARGET_DIR"]) / "guidance_tokens.json").read_text())
    print(f"\nguidance over {totals['calls']} calls, {totals['tokens']} tokens: second token within floor at {totals['with_alternative']}")
    size = len(json.dumps({"examples": examples, **layout}, separators=(",", ":")))
    print(f"payload JSON for {len(examples)} examples: {size / 1024:.1f} KiB ({size / 1024 / len(examples):.1f} KiB per example)")
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
