"""Measured activations of both models for the 3D view.

Four settings are captured for each example: the target (LFM3) and the grader (D1), each as the base model and
with its LoRA. LFM3 reads the request and writes the tool call; D1 reads the request and the call and answers
three questions about it. For every setting `capture_activations` records the attention weights of each
attention layer between the tokens that are drawn and the size of the residual stream after each of the 24
layers, so the view can show what a LoRA changes and where. For LFM3 it also records, token by token, what
guided decoding has to work with. `ActivationWidget` draws that data.
"""

from __future__ import annotations

import base64
import gc
import json
import time
from pathlib import Path

import anywidget
import numpy as np
import traitlets

MAX_REQUEST_TOKENS = 28
MAX_CODE_TOKENS = 64
_CALL_HEAD = "<function=run_nushell>\n<parameter=code>\n"


# ----------------------------------------------------------------------------------------------------------------
# Encoding helpers
# ----------------------------------------------------------------------------------------------------------------
def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).decode("ascii")


def _pack_lower(mats: np.ndarray) -> tuple[str, list[float]]:
    """[M, T, T] causal matrices -> one base64 string of the M lower triangles (diagonal included), each stored as
    uint8 of value / scale * 255 with its own scale (the matrix max), so small weights keep their precision."""
    m, t, _ = mats.shape
    rows, cols = np.tril_indices(t)
    out, scales = [], []
    for i in range(m):
        tri = mats[i][rows, cols]
        scale = float(tri.max()) if tri.size else 0.0
        scales.append(round(scale, 7))
        out.append(np.rint(tri / scale * 255.0) if scale > 0 else np.zeros_like(tri))
    return _b64(np.clip(np.concatenate(out), 0, 255)), scales


def _unpack_lower(blob: str, scales: list[float], t: int) -> np.ndarray:
    """Inverse of `_pack_lower`, used by the check script."""
    raw = np.frombuffer(base64.b64decode(blob), dtype=np.uint8).astype(np.float32)
    n = t * (t + 1) // 2
    rows, cols = np.tril_indices(t)
    mats = np.zeros((len(scales), t, t), dtype=np.float32)
    for i, scale in enumerate(scales):
        mats[i][rows, cols] = raw[i * n : (i + 1) * n] / 255.0 * scale
    return mats


def _free():
    import mlx.core as mx

    gc.collect()
    mx.clear_cache()


def _read_adapter(path: Path, tries: int = 5) -> dict:
    """Load a safetensors adapter fully into memory; retry if the file is being rewritten."""
    import mlx.core as mx

    last = None
    for attempt in range(tries):
        try:
            weights = mx.load(str(path))
            mx.eval(list(weights.values()))
            return dict(weights)
        except Exception as err:  # partial file while another process writes it
            last = err
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"could not read {path}: {last}")


def _display(text: str) -> str:
    return text.replace("\n", "⏎").replace("\t", "⇥")


# ----------------------------------------------------------------------------------------------------------------
# Token selection
# ----------------------------------------------------------------------------------------------------------------
def _code_of(completion: str, parse_tool_call) -> str:
    code = parse_tool_call(completion)
    if code is not None:
        return code
    start = completion.find(_CALL_HEAD)
    return completion[start + len(_CALL_HEAD) :] if start >= 0 else completion


def _visible_tokens(tokenizer, prompt: str, row: dict, parse_tool_call) -> dict:
    """Tokenize prompt + reference completion once with offsets and keep the tokens of the request text and of the
    Nushell code. Character spans of code tokens are relative to the code string."""
    text = prompt + row["completion"]
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    marker = "Request: " + row["speech"]
    at = prompt.rfind(marker)
    if at < 0:
        at = prompt.rfind(row["speech"]) - len("Request: ")
        if at < -len("Request: ") + 1:
            raise ValueError(f"request text not found in prompt for {row.get('id')}")
    req_lo = at + len("Request: ")
    req_hi = req_lo + len(row["speech"])
    code = _code_of(row["completion"], parse_tool_call)
    code_lo = len(prompt) + row["completion"].find(code)
    code_hi = code_lo + len(code)

    def pick(lo, hi, cap, seg):
        found = []
        for pos, (s, e) in enumerate(offsets):
            if e <= lo or s >= hi or e <= s:
                continue
            a, b = max(s, lo), min(e, hi)
            found.append({"t": _display(text[a:b]), "seg": seg, "pos": pos, "span": [a - code_lo, b - code_lo] if seg == "out" else None})
        return found[:cap], len(found) > cap

    req, req_cut = pick(req_lo, req_hi, MAX_REQUEST_TOKENS, "in")
    out, out_cut = pick(code_lo, code_hi, MAX_CODE_TOKENS, "out")
    return {"ids": ids, "tokens": req + out, "n_in": len(req), "n_out": len(out), "code": code, "truncated": {"in": req_cut, "out": out_cut}}


def _code_tokens(tokenizer, code: str) -> list[dict]:
    enc = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True)
    out = []
    for s, e in enc["offset_mapping"]:
        if e > s:
            out.append({"t": _display(code[s:e]), "span": [s, e]})
        if len(out) >= MAX_CODE_TOKENS:
            break
    return out


# ----------------------------------------------------------------------------------------------------------------
# Attention capture
# ----------------------------------------------------------------------------------------------------------------
class _AttentionRecorder:
    """Temporarily replaces `Attention.__call__` with a wrapper that recomputes the attention weights exactly
    (softmax(q kᵀ · scale + mask), each KV head shared by consecutive query heads) and checks that mixing `v`
    with them, gating and projecting, reproduces the module's own output."""

    def __init__(self, lm, visible: list[int]):
        import sys

        self.index = {id(layer.self_attn): i for i, layer in enumerate(lm.layers) if layer.layer_type != "conv"}
        self.cls = type(next(layer.self_attn for layer in lm.layers if layer.layer_type != "conv"))
        self.sdpa = sys.modules[self.cls.__module__].scaled_dot_product_attention
        self.visible = visible
        self.records: dict[int, dict] = {}
        self.original = None

    def __enter__(self):
        import mlx.core as mx

        original = self.original = self.cls.__call__
        scaled_dot_product_attention = self.sdpa
        rec = self

        def wrapper(module, x, mask=None, cache=None, position_ids=None):
            out = original(module, x, mask, cache, position_ids)
            li = rec.index.get(id(module))
            if li is None or cache is not None:
                return out
            b, n, _ = x.shape
            h, kv, d = module.n_heads, module.n_kv_heads, module.head_dim
            q = module.q_norm(module.q_proj(x).reshape(b, n, h, d))
            k = module.k_norm(module.k_proj(x).reshape(b, n, kv, d))
            v = module.v_proj(x).reshape(b, n, kv, d).transpose(0, 2, 1, 3)
            q = module.rope(q.transpose(0, 2, 1, 3), 0, position_ids).astype(mx.float32)
            k = module.rope(k.transpose(0, 2, 1, 3), 0, position_ids).astype(mx.float32)
            rep = h // kv
            q = q.reshape(b, kv, rep, n, d)
            scores = (q @ k[:, :, None].swapaxes(-1, -2)) * module.scale  # [b, kv, rep, n, n]
            if isinstance(mask, str):
                allowed = mx.arange(n)[:, None] >= mx.arange(n)[None]
                scores = mx.where(allowed, scores, -mx.inf)
            elif mask is not None:
                scores = mx.where(mask, scores, -mx.inf) if mask.dtype == mx.bool_ else scores + mask.astype(mx.float32)
            weights = mx.softmax(scores, axis=-1)
            mixed = (weights @ v.astype(mx.float32)[:, :, None]).reshape(b, h, n, d)
            # Pre-gate check against the library's own attention on the same q, k, v.
            pre = scaled_dot_product_attention(
                q.reshape(b, h, n, d).astype(x.dtype), k.astype(x.dtype), v, cache=None, scale=module.scale, mask=mask
            ).astype(mx.float32)
            pre_err = mx.max(mx.abs(mixed - pre)) / mx.maximum(mx.max(mx.abs(pre)), 1e-12)
            mixed = mixed.transpose(0, 2, 1, 3) * mx.sigmoid(module.gate_proj(x).astype(mx.float32))[..., None]
            mine = module.o_proj(mixed.astype(x.dtype).reshape(b, n, -1)).astype(mx.float32)
            ref = out.astype(mx.float32)
            vis = mx.array(rec.visible)
            sub = weights[0][:, :, vis]  # [kv, rep, t, n]
            inside = sub[:, :, :, vis]  # [kv, rep, t, t]
            mx.eval(inside, sub, mine, ref, pre_err)
            w_all = np.array(sub)
            w_vis = np.array(inside)
            rec.records[li] = {
                "groups": w_vis.mean(axis=1),  # [4, t, t]
                "avg": w_vis.mean(axis=(0, 1)),  # [t, t]
                "outside_groups": 1.0 - w_vis.sum(axis=-1).mean(axis=1),  # [4, t]
                "outside_avg": 1.0 - w_vis.sum(axis=-1).mean(axis=(0, 1)),  # [t]
                "row_sum_err": float(np.abs(w_all.sum(axis=-1) - 1.0).max()),
                "pregate_rel": float(pre_err.item()),
                "verify_max_abs": float(mx.max(mx.abs(mine - ref)).item()),
                "verify_rel": float((mx.sqrt(mx.sum((mine - ref) ** 2)) / mx.maximum(mx.sqrt(mx.sum(ref**2)), 1e-12)).item()),
                "masked": None if mask is None else ("causal" if isinstance(mask, str) else f"array{tuple(mask.shape)}"),
            }
            del scores, weights, mixed, mine, ref, sub, inside, pre
            return out

        self.cls.__call__ = wrapper
        return self

    def __exit__(self, *exc):
        self.cls.__call__ = self.original
        return False


class _ResidualRecorder:
    """Keeps the residual stream after every layer at the visible tokens: `stream[layer]` is [t, hidden]."""

    def __init__(self, lm, visible: list[int]):
        self.cls = type(lm.layers[0])
        self.index = {id(layer): i for i, layer in enumerate(lm.layers)}
        self.visible = visible
        self.stream: dict[int, np.ndarray] = {}
        self.original = None

    def __enter__(self):
        import mlx.core as mx

        original = self.original = self.cls.__call__
        rec = self

        def wrapper(module, x, mask=None, cache=None, position_ids=None):
            out = original(module, x, mask, cache, position_ids)
            li = rec.index.get(id(module))
            if li is not None and cache is None:
                rec.stream[li] = np.array(out[0][mx.array(rec.visible)].astype(mx.float32))
            return out

        self.cls.__call__ = wrapper
        return self

    def __exit__(self, *exc):
        self.cls.__call__ = self.original
        return False


def _record(lm, visible: list[int], forward):
    """Run `forward()` once with both recorders on. Returns (its result, attention records, residual stream)."""
    import mlx.core as mx

    with _AttentionRecorder(lm, visible) as attention, _ResidualRecorder(lm, visible) as residual:
        result = forward()
        mx.eval(result)
    mx.clear_cache()
    return result, attention.records, residual.stream


def _pack_square(records: dict[int, dict]) -> dict:
    """All-head average only, whole matrix: D1's full-attention layers read the state in both directions."""
    out = {}
    for li, r in sorted(records.items()):
        avg = r["avg"].astype(np.float32)
        scale = float(avg.max()) or 1.0
        out[str(li)] = {
            "m": _b64(np.clip(np.rint(avg / scale * 255.0), 0, 255)), "s": [round(scale, 7)], "sq": True,
            "o": _b64(np.rint(np.clip(r["outside_avg"], 0, 1) * 255).reshape(-1)),
        }
    return out


def _stream_summary(base: dict[int, np.ndarray], lora: dict[int, np.ndarray] | None) -> dict:
    """Per layer and token: the size of the residual stream in the base model, and how far the LoRA moved it,
    as a share of that size."""
    layers = sorted(base)
    size = np.stack([np.linalg.norm(base[li], axis=-1) for li in layers])
    out = {"size": [[round(float(v), 2) for v in row] for row in size]}
    if lora is not None:
        moved = np.stack([np.linalg.norm(lora[li] - base[li], axis=-1) for li in layers]) / np.maximum(size, 1e-9)
        out["moved"] = [[round(float(v), 4) for v in row] for row in moved]
    return out


def _pack_variant(records: dict[int, dict]) -> dict:
    out = {}
    for li, r in sorted(records.items()):
        blob, scales = _pack_lower(np.concatenate([r["groups"], r["avg"][None]], axis=0).astype(np.float32))
        outside = np.concatenate([r["outside_groups"], r["outside_avg"][None]], axis=0)
        out[str(li)] = {"m": blob, "s": scales, "o": _b64(np.rint(np.clip(outside, 0, 1) * 255).reshape(-1))}
    return out


# ----------------------------------------------------------------------------------------------------------------
# Grader
# ----------------------------------------------------------------------------------------------------------------
def _pick_broken(defs: dict, row: dict) -> dict | None:
    rows = [g for g in defs["dataset"]["grader"]["dev"] if g.get("chain_id") == row.get("chain_id") and g.get("is_broken") and g.get("form") == "break"]
    if not rows:
        return None
    plain = {"hardcoded_result", "wrong_tool_or_malformed_call"}  # code is a literal or not a call: little to point at
    rows.sort(key=lambda g: (g.get("prompt") != row["speech"], g.get("failure_type") in plain, len(g.get("span_text") or ""), g["id"]))
    return rows[0]


def _grade(defs: dict, model, prompt: str, completion: str, code: str, code_tokens: list[dict], span_text: str | None = None) -> dict:
    item = defs["encode_grader_row"](model.engine, {"prompt": prompt, "completion": completion, "is_broken": bool(span_text), "span_text": span_text})
    return _answers(defs, item, defs["grader_readouts"](model, item), prompt, completion, code, code_tokens, span_text)


def _answers(defs: dict, item: dict, outputs, prompt: str, completion: str, code: str, code_tokens: list[dict], span_text: str | None = None) -> dict:
    """D1's three answers from its readout logits: broken or not, the failure type, and where the fault is
    (start and end pointers, mapped onto `code_tokens`)."""
    import mlx.core as mx
    import mlx.nn as nn

    labels = defs["FAILURE_LABELS"]
    yes_no, choice, pointer = outputs
    p_yes = mx.softmax(yes_no)
    log_type = nn.log_softmax(choice)
    p_type = mx.softmax(choice)
    p_ptr = mx.softmax(pointer, axis=-1)
    mx.eval(p_yes, p_type, p_ptr, log_type)
    log_type = [round(float(v), 4) for v in np.array(log_type).astype(float)]
    p_broken = float(p_yes[0].item())
    p_type = np.array(p_type).astype(float)
    p_ptr = np.array(p_ptr).astype(float)  # [2, n_positions]
    offsets = item["positions"].offsets
    head = f"User request: {prompt}\n\nAssistant tool call:\n"
    code_lo = len(head) + completion.find(code)
    mapped = np.zeros((2, len(code_tokens)))
    on_code = np.zeros(2)
    for i, off in enumerate(offsets[: p_ptr.shape[1]]):
        if off is None or off[1] <= off[0]:
            continue
        a, b = off[0] - code_lo, off[1] - code_lo
        for j, tok in enumerate(code_tokens):
            s, e = tok["span"]
            overlap = min(b, e) - max(a, s)
            if overlap > 0:
                share = overlap / (b - a)
                mapped[:, j] += p_ptr[:, i] * share
                on_code += p_ptr[:, i] * share
    order = np.argsort(-p_type)[:3]
    broken_types = [(labels[i], p_type[i]) for i in np.argsort(-p_type) if labels[i] != "valid"]
    true_span = None
    if span_text:
        lo = code.find(span_text)
        if lo >= 0:
            hi = lo + len(span_text)
            hits = [j for j, t in enumerate(code_tokens) if t["span"][1] > lo and t["span"][0] < hi]
            true_span = {"text": span_text, "tokens": [hits[0], hits[-1]] if hits else None}
        else:
            true_span = {"text": span_text, "tokens": None}
    return {
        "verdict": "broken" if p_broken > 0.5 else "valid",
        "p_broken": round(p_broken, 4),
        "failure_type": labels[int(np.argmax(p_type))],
        "type_if_broken": broken_types[0][0],
        "top3": [[labels[i], round(float(p_type[i]), 4)] for i in order],
        "log_type": log_type,
        "start": [round(float(x), 4) for x in mapped[0]],
        "end": [round(float(x), 4) for x in mapped[1]],
        "p_nothing": [round(float(p_ptr[0, 0]), 4), round(float(p_ptr[1, 0]), 4)],
        "p_on_code": [round(float(x), 4) for x in on_code],
        "true_span": true_span,
    }


def _d1_visible(item: dict, prompt: str, completion: str, code: str) -> dict:
    """The tokens D1 reads that are drawn: the request and the Nushell code, by their place in its packed row."""
    head = f"User request: {prompt}\n\nAssistant tool call:\n"
    state = head + completion
    req_lo = len("User request: ")
    req_hi = req_lo + len(prompt)
    code_lo = len(head) + completion.find(code)
    code_hi = code_lo + len(code)

    def pick(lo, hi, cap, seg):
        found = []
        for pos, off in enumerate(item["positions"].offsets):
            if off is None or off[1] <= off[0] or off[1] <= lo or off[0] >= hi:
                continue
            a, b = max(off[0], lo), min(off[1], hi)
            found.append({"t": _display(state[a:b]), "seg": seg, "pos": pos, "span": [a - code_lo, b - code_lo] if seg == "out" else None})
        return found[:cap], len(found) > cap

    req, req_cut = pick(req_lo, req_hi, MAX_REQUEST_TOKENS, "in")
    out, out_cut = pick(code_lo, code_hi, MAX_CODE_TOKENS, "out")
    return {"tokens": req + out, "n_in": len(req), "n_out": len(out), "truncated": {"in": req_cut, "out": out_cut}}


def _grader_pass(defs: dict, examples: list[dict], rows: list[dict], log) -> None:
    """D1 over each example's request and reference call, as the base model and with the grader LoRA: attention,
    residual stream and the three answers. One broken variant of the call is graded too (answers only)."""
    import mlx.core as mx
    from mlx_vlm import load

    gdir = Path(defs["GRADER_DIR"])
    if not (gdir / "adapter.safetensors").exists() or not (gdir / "adapter_config.json").exists():
        for ex in examples:
            ex["d1"] = None
        return
    cfg = json.loads((gdir / "adapter_config.json").read_text(encoding="utf-8"))
    weights = _read_adapter(gdir / "adapter.safetensors")
    model, processor = load(defs["model_path"](defs["MODEL_FOLDERS"]["grader"]))
    try:
        lm, layers, _ = defs["attach_lora"](model, cfg["rank"], cfg["alpha"], 0.0, cfg["placement"])
        lm.load_weights(list(weights.items()), strict=False)
        defs["set_lora"](layers, training=False)
        on = cfg["alpha"] / cfg["rank"]
        tok = getattr(processor, "tokenizer", processor)
        for ex, row in zip(examples, rows):
            prompt, completion, code = row["speech"], row["completion"], ex["code"]
            item = defs["encode_grader_row"](model.engine, {"prompt": prompt, "completion": completion, "is_broken": False, "span_text": None})
            sel = _d1_visible(item, prompt, completion, code)
            visible = [t["pos"] for t in sel["tokens"]]
            code_tokens = [{"t": t["t"], "span": t["span"]} for t in sel["tokens"][sel["n_in"] :]]
            runs = {}
            for name, scale in (("base", 0.0), ("lora", on)):
                defs["set_lora"](layers, scale=scale)
                outputs, attention, stream = _record(lm, visible, lambda: list(defs["grader_readouts"](model, item)))
                runs[name] = {"attention": attention, "stream": stream, "answer": _answers(defs, item, outputs, prompt, completion, code, code_tokens)}
                runs[name]["answer"].update({"kind": "reference", "true_label": "valid"})
            broken = None
            broken_row = _pick_broken(defs, row)
            if broken_row is not None:
                bcode = _code_of(broken_row["completion"], defs["parse_tool_call"])
                btoks = _code_tokens(tok, bcode)
                broken = {}
                for name, scale in (("base", 0.0), ("lora", on)):
                    defs["set_lora"](layers, scale=scale)
                    broken[name] = _grade(defs, model, broken_row["prompt"], broken_row["completion"], bcode, btoks, broken_row.get("span_text"))
                broken.update({
                    "id": broken_row["id"], "prompt": broken_row["prompt"], "code": bcode, "tokens": [t["t"] for t in btoks],
                    "true_label": broken_row["failure_type"], "note": broken_row.get("note"),
                })
            ex["d1"] = {
                "tokens": sel["tokens"], "n_in": sel["n_in"], "n_out": sel["n_out"], "n_total": len(item["ids"]),
                "truncated": {k: bool(v) for k, v in sel["truncated"].items()},
                "attn": {"base": _pack_square(runs["base"]["attention"]), "lora": _pack_square(runs["lora"]["attention"])},
                "stream": _stream_summary(runs["base"]["stream"], runs["lora"]["stream"]),
                "answer": {"base": runs["base"]["answer"], "lora": runs["lora"]["answer"]},
                "broken": broken,
                "labels": list(defs["FAILURE_LABELS"]),
                "lora": {"rank": cfg["rank"], "alpha": cfg["alpha"], "placement": cfg["placement"], "name": cfg.get("name")},
            }
            a = ex["d1"]["answer"]
            log(f"[activation] D1 {row['id']}: {len(item['ids'])} tokens · base says {a['base']['verdict']} ({a['base']['p_broken']:.2f}), with LoRA {a['lora']['verdict']} ({a['lora']['p_broken']:.2f})")
    finally:
        del model, processor, weights
        _free()


# ----------------------------------------------------------------------------------------------------------------
# Guided decoding, token by token
# ----------------------------------------------------------------------------------------------------------------
GUIDANCE_WEIGHTS = (0.25, 0.5, 1.0)
GUIDANCE_FLOORS = (0.03, 0.1, 0.3)


def _next_token_logp(lm, ids: list[int], positions: list[int]) -> np.ndarray:
    """log p(token at each position | everything before it), as [len(positions), vocabulary]."""
    import mlx.core as mx
    import mlx.nn as nn

    logits = lm(mx.array([ids])).logits[0]
    rows = nn.log_softmax(logits[mx.array([p - 1 for p in positions])].astype(mx.float32), axis=-1)
    mx.eval(rows)
    out = np.array(rows)
    del logits, rows
    mx.clear_cache()
    return out


def _guidance_tokens(logp: dict[str, np.ndarray], ids: list[int], positions: list[int], tokenizer=None) -> dict:
    """What guided decoding has to work with at each token of the reference call, read with the reference call as
    the text so far. `logp` holds [tokens, vocabulary] log-probabilities for "good" and for each model it can be
    steered away from.

    At one token, P_good prefers `a` over its runner-up `b` by the margin m = log p_good(a) − log p_good(b).
    Guidance scores x by log p_good(x) + w·(log p_good(x) − log p_other(x)) over the tokens with
    p_good(x) ≥ floor · p_good(a). So `b` is even eligible only if m ≤ ln(1/floor), and it wins only if
    w · push > m, where push = [log p_good(b) − log p_other(b)] − [log p_good(a) − log p_other(a)]."""
    good = logp["good"]
    n = good.shape[0]
    order = np.argsort(-good, axis=-1)[:, :2]
    a, b = order[:, 0], order[:, 1]
    rows = np.arange(n)
    margin = good[rows, a] - good[rows, b]
    ref = np.array([ids[p] for p in positions])
    text = (lambda i: _display(tokenizer.decode([int(i)]))) if tokenizer is not None else (lambda i: "")
    out = {
        "margin": [round(float(v), 3) for v in margin],
        "alternatives": {str(f): [int(v) for v in (good >= good[rows, a][:, None] + np.log(f)).sum(axis=-1) - 1] for f in GUIDANCE_FLOORS},
        "top": [text(i) for i in a], "second": [text(i) for i in b],
        "top_is_reference": [bool(v) for v in a == ref],
        "against": {},
    }
    for name, other in logp.items():
        if name == "good":
            continue
        push = (good[rows, b] - other[rows, b]) - (good[rows, a] - other[rows, a])
        flips = {}
        for w in GUIDANCE_WEIGHTS:
            for f in GUIDANCE_FLOORS:
                keep = good >= good[rows, a][:, None] + np.log(f)
                chosen = np.where(keep, good + w * (good - other), -1e9).argmax(axis=-1)
                flips[f"{w}|{f}"] = [
                    {"at": int(j), "to": text(chosen[j]), "from_reference": bool(a[j] == ref[j]), "to_reference": bool(chosen[j] == ref[j])}
                    for j in np.flatnonzero(chosen != a)
                ]
        out["against"][name] = {"push": [round(float(v), 3) for v in push], "flips": flips}
    return out


def _guidance_totals(per_row: list[dict]) -> dict:
    """Add the per-token records of many calls up: how often guidance has a choice at all, and what it does."""
    margins = np.concatenate([np.array(r["margin"]) for r in per_row])
    wrong = sum(int(not v) for r in per_row for v in r["top_is_reference"])
    total = {
        "calls": len(per_row), "tokens": int(margins.size), "top_is_not_reference": wrong,
        "margin_quantiles": {q: round(float(np.quantile(margins, float(q))), 3) for q in ("0.05", "0.25", "0.5", "0.75")},
        "with_alternative": {f: int(sum(int(v > 0) for r in per_row for v in r["alternatives"][f])) for f in per_row[0]["alternatives"]},
        "against": {},
    }
    for name in per_row[0]["against"]:
        pushes = np.concatenate([np.array(r["against"][name]["push"]) for r in per_row])
        flips = {}
        for key in per_row[0]["against"][name]["flips"]:
            found = [f for r in per_row for f in r["against"][name]["flips"][key]]
            flips[key] = {
                "changed": len(found), "away_from_reference": sum(f["from_reference"] for f in found),
                "onto_reference": sum(f["to_reference"] for f in found),
            }
        total["against"][name] = {"mean_abs_push": round(float(np.abs(pushes).mean()), 3), "flips": flips, "push": [round(float(v), 3) for v in pushes]}
    total["margin"] = [round(float(v), 3) for v in margins]  # every token, for the histogram
    total["top_is_reference"] = [bool(v) for r in per_row for v in r["top_is_reference"]]
    return total


# ----------------------------------------------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------------------------------------------
def capture_activations(defs, rows, *, max_examples=4, stats_rows=None, log=print) -> list[dict]:
    """Both models over each row's request and reference call (at most `max_examples` rows).

    Each returned dict has `"lfm3"` and `"d1"`: the drawn tokens, packed attention for `"base"` and `"lora"`,
    the residual stream, and what the model answered. `"lfm3"` also has `"guidance"` (see `_guidance_tokens`).
    With `stats_rows`, the same guidance record is added up over those rows and written to
    `runs/target/guidance_tokens.json`."""
    import mlx.core as mx
    from mlx_vlm import load

    rows = list(rows)[:max_examples]
    mx.set_cache_limit(512 * 1024**2)
    target_dir = Path(defs["TARGET_DIR"])
    lora_cfg = {"rank": 16, "alpha": 32.0, "placement": "all_linear"}
    report = target_dir / "report.json"
    if report.exists():
        try:
            cfg = json.loads(report.read_text(encoding="utf-8")).get("config", {})
            lora_cfg.update({k: cfg[k] for k in ("rank", "alpha", "placement") if k in cfg})
        except (OSError, json.JSONDecodeError):
            pass
    adapters = {name: _read_adapter(target_dir / f"adapter_{name}.safetensors") for name in ("good", "bad", "weak") if (target_dir / f"adapter_{name}.safetensors").exists()}
    on = lora_cfg["alpha"] / lora_cfg["rank"]

    examples = []
    model, processor = load(defs["model_path"](defs["MODEL_FOLDERS"]["target"]))
    try:
        tok = getattr(processor, "tokenizer", processor)
        lm, layers, _ = defs["attach_lora"](model, lora_cfg["rank"], lora_cfg["alpha"], 0.0, lora_cfg["placement"])
        defs["set_lora"](layers, training=False)

        def use(name):  # "base" is the model with the adapter switched off
            if name == "base":
                defs["set_lora"](layers, scale=0.0)
            else:
                lm.load_weights(list(adapters[name].items()), strict=False)
                defs["set_lora"](layers, scale=on)

        def guidance_of(ids, positions):
            if "good" not in adapters:
                return None
            logp = {}
            for name in ("good", "bad", "weak", "base"):
                if name == "base" or name in adapters:
                    use(name)
                    logp[name] = _next_token_logp(lm, ids, positions)
            return _guidance_tokens(logp, ids, positions, tok), logp

        for row in rows:
            prompt = defs["target_prompt"](tok, row)
            sel = _visible_tokens(tok, prompt, row, defs["parse_tool_call"])
            visible = [t["pos"] for t in sel["tokens"]]
            forward = lambda: lm(mx.array([sel["ids"]])).logits
            use("base")
            _, base, base_stream = _record(lm, visible, forward)
            lora = lora_stream = None
            if "good" in adapters:
                use("good")
                _, lora, lora_stream = _record(lm, visible, forward)
            checks = [r for rec in (base, lora or {}) for r in rec.values()]
            out_positions = visible[sel["n_in"] :]
            guidance = written = None
            found = guidance_of(sel["ids"], out_positions)
            if found:
                guidance, logp = found
                reference = np.array([sel["ids"][p] for p in out_positions])
                at = np.arange(len(out_positions))
                written = {
                    name: {
                        "top": [_display(tok.decode([int(i)])) for i in logp[key].argmax(axis=-1)],
                        "p_reference": [round(float(v), 4) for v in np.exp(logp[key][at, reference])],
                    }
                    for name, key in (("base", "base"), ("lora", "good"))
                }
            examples.append({
                "id": row["id"],
                "chain_id": row.get("chain_id"),
                "request": row["speech"],
                "code": sel["code"],
                "lfm3": {
                    "tokens": sel["tokens"], "n_in": sel["n_in"], "n_out": sel["n_out"], "n_total": len(sel["ids"]),
                    "truncated": {k: bool(v) for k, v in sel["truncated"].items()},
                    "attn": {"base": _pack_variant(base), "lora": _pack_variant(lora) if lora else None},
                    "stream": _stream_summary(base_stream, lora_stream),
                    "written": written,
                    "guidance": guidance,
                    "lora": dict(lora_cfg) if lora else None,
                },
                "d1": None,
                "sliding_window": int(getattr(lm.config, "sliding_window", 0) or 0),
                "stats": {
                    "pregate_rel": max(r["pregate_rel"] for r in checks),
                    "verify_max_abs": max(r["verify_max_abs"] for r in checks),
                    "verify_rel": max(r["verify_rel"] for r in checks),
                    "row_sum_err": max(r["row_sum_err"] for r in checks),
                    "mask_kinds": sorted({str(r["masked"]) for r in checks}),
                },
            })
            log(f"[activation] LFM3 {row['id']}: {len(sel['ids'])} tokens, {sel['n_in']} request + {sel['n_out']} code shown")

        if stats_rows and "good" in adapters:
            per_row = []
            for done, row in enumerate(stats_rows, 1):
                prompt = defs["target_prompt"](tok, row)
                sel = _visible_tokens(tok, prompt, row, defs["parse_tool_call"])
                positions = [t["pos"] for t in sel["tokens"][sel["n_in"] :]]
                per_row.append(guidance_of(sel["ids"], positions)[0])
                if done % 10 == 0 or done == len(stats_rows):
                    log(f"[activation] guidance, token by token: {done}/{len(stats_rows)} calls")
            totals = _guidance_totals(per_row)
            totals["note"] = "read with the reference call as the text so far; at most %d tokens of each call" % MAX_CODE_TOKENS
            (target_dir / "guidance_tokens.json").write_text(json.dumps(totals, indent=1), encoding="utf-8")
    finally:
        del model, processor, adapters
        _free()
    _grader_pass(defs, examples, rows, log)
    return examples


def pick_rows(defs, n: int = 4) -> list[dict]:
    """Held-out target rows from distinct categories, preferring ones with a broken grader variant of the same
    request and code of moderate length (so there is something to draw without truncation)."""
    dev = defs["dataset"]["target"]["dev"]
    broken = {}
    for g in defs["dataset"]["grader"]["dev"]:
        if g.get("is_broken") and g.get("form") == "break":
            broken.setdefault(g["chain_id"], set()).add(g.get("prompt"))

    def rank(r):
        code = defs["parse_tool_call"](r["completion"]) or r["completion"]
        prompts = broken.get(r.get("chain_id"), set())
        return (not prompts, r["speech"] not in prompts, abs(len(code) - 110) // 20, r["id"])

    out, seen = [], set()
    for r in sorted(dev, key=rank):
        if r.get("category") in seen:
            continue
        out.append(r)
        seen.add(r.get("category"))
        if len(out) >= n:
            break
    return out


def load_guidance_totals(target_dir) -> dict | None:
    """What `capture_activations` added up over the scored calls, or None before the first capture."""
    path = Path(target_dir) / "guidance_tokens.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def scored_rows(defs, n: int = 40) -> list[dict]:
    """The held-out requests the report's decoders were scored on, in the same order."""
    dev = defs["dataset"]["target"]["dev"]
    return sorted(dev, key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))[:n]


# ----------------------------------------------------------------------------------------------------------------
# Widget
# ----------------------------------------------------------------------------------------------------------------
_CSS = """
.lfm3-root { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; user-select: none; color: inherit; max-width: 100%; }
.lfm3-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px; padding: 2px 0 8px; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 11px; }
.lfm3-bar label { display: flex; align-items: center; gap: 6px; cursor: pointer; white-space: nowrap; }
.lfm3-bar label.off { opacity: 0.45; cursor: default; }
.lfm3-play { width: 30px; height: 30px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: transparent; color: inherit; font-size: 13px; line-height: 1; cursor: pointer; flex-shrink: 0; }
.lfm3-play:hover { background: rgba(148,163,184,0.2); }
.lfm3-layer { display: flex; align-items: center; gap: 8px; flex: 1 1 260px; min-width: 200px; }
.lfm3-layer input[type=range] { flex: 1; min-width: 80px; accent-color: #6366f1; }
.lfm3-layer-badge { white-space: nowrap; min-width: 190px; }
.lfm3-bar select { font: inherit; color: inherit; background: transparent; border: 1px solid rgba(148,163,184,0.5); border-radius: 6px; padding: 2px 4px; max-width: 100%; }
.lfm3-stage { position: relative; width: 100%; height: min(480px, 64vh); }
.lfm3-stage canvas { width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; }
.lfm3-stage canvas:active { cursor: grabbing; }
.lfm3-setting { position: absolute; top: 6px; left: 8px; pointer-events: none; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 11px; line-height: 1.4; max-width: 60%; }
.lfm3-setting strong { font-size: 13px; }
.lfm3-legend { position: absolute; bottom: 6px; right: 8px; pointer-events: none; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 10px; line-height: 1.5; text-align: right; max-width: 46%; }
.lfm3-legend i { display: inline-block; width: 74px; height: 8px; border-radius: 2px; vertical-align: middle; margin: 0 5px; }
.lfm3-carousel-wrap { border-top: 1px solid rgba(148,163,184,0.3); padding: 10px 0 2px; display: flex; align-items: center; gap: 8px; }
.lfm3-car-label { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 11px; opacity: 0.75; white-space: nowrap; }
.lfm3-nav { flex-shrink: 0; width: 28px; height: 28px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.5); background: transparent; color: inherit; font-size: 16px; line-height: 1; cursor: pointer; }
.lfm3-nav:hover:not(:disabled) { background: rgba(148,163,184,0.2); }
.lfm3-nav:disabled { opacity: 0.3; cursor: default; }
.lfm3-carousel { flex: 1; min-width: 0; display: flex; gap: 10px; overflow-x: auto; scrollbar-width: none; align-items: stretch; position: relative; scroll-snap-type: x mandatory; scroll-behavior: smooth; }
.lfm3-carousel::-webkit-scrollbar { display: none; }
.lfm3-card { flex: 0 0 min(560px, 86%); scroll-snap-align: start; border-radius: 8px; border: 1px solid rgba(148,163,184,0.35); padding: 10px 12px; font-family: ui-sans-serif, system-ui, sans-serif; display: flex; flex-direction: column; gap: 8px; box-sizing: border-box; }
.lfm3-card.active { border-color: #6366f1; box-shadow: 0 0 0 1px #6366f1; }
.lfm3-card.inactive { opacity: 0.55; cursor: pointer; }
.lfm3-card.inactive:hover { opacity: 1; }
.lfm3-dots { display: flex; gap: 5px; justify-content: center; padding-top: 6px; }
.lfm3-dots i { width: 6px; height: 6px; border-radius: 999px; background: rgba(148,163,184,0.5); cursor: pointer; }
.lfm3-dots i.on { background: #6366f1; }
.lfm3-row { display: block; font-family: inherit; font-size: 13px; line-height: 1.6; white-space: pre-wrap; overflow-wrap: anywhere; }
.lfm3-row.lfm3-code, .lfm3-out { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px; }
.lfm3-out { opacity: 0.8; }
.lfm3-tag { font-size: 10px; opacity: 0.7; flex: 0 0 100%; margin-top: 1px; }
.lfm3-tag b { font-weight: 700; opacity: 1; }
.lfm3-tok { border-radius: 3px; white-space: pre-wrap; }
.lfm3-meta { font-size: 10px; opacity: 0.8; line-height: 1.4; white-space: normal; }
"""

_ESM = r"""
const DECODED = new Map();

function b64bytes(s) {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// Packed attention -> {groups: [G][T][T], avg: [T][T], outside: [T], causal}. LFM3 layers are stored as lower
// triangles (4 key-value groups and their average); D1 layers as whole matrices of the average.
function decodeLayer(key, entry, T) {
  if (DECODED.has(key)) return DECODED.get(key);
  const raw = b64bytes(entry.m), out = b64bytes(entry.o);
  let res;
  if (entry.sq) {
    const avg = Array.from({ length: T }, (_, q) => Float32Array.from({ length: T }, (_, k) => raw[q * T + k] / 255 * entry.s[0]));
    res = { groups: [avg], avg, outside: Array.from({ length: T }, (_, q) => out[q] / 255), causal: false };
  } else {
    const n = T * (T + 1) / 2;
    const mats = entry.s.map((scale, m) => {
      const mat = Array.from({ length: T }, () => new Float32Array(T));
      let i = m * n;
      for (let q = 0; q < T; q++) for (let k = 0; k <= q; k++) mat[q][k] = raw[i++] / 255 * scale;
      return mat;
    });
    res = { groups: mats.slice(0, 4), avg: mats[4], outside: Array.from({ length: T }, (_, q) => out[4 * T + q] / 255), causal: true };
  }
  DECODED.set(key, res);
  return res;
}

const lerp = (a, b, t) => a + (b - a) * t;
const mix3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
const DARK = [22, 30, 48], MID = [109, 71, 214], HOT = [252, 211, 77], UP = [239, 68, 68], DOWN = [59, 130, 246];
// how much there is: dark -> violet -> yellow
const amount = (v) => { const t = Math.max(0, Math.min(1, v)); return t < 0.5 ? mix3(DARK, MID, t * 2) : mix3(MID, HOT, t * 2 - 1); };
// how much it changed: blue = less, red = more
const change = (v) => { const t = Math.pow(Math.min(1, Math.abs(v)), 0.6); return mix3(DARK, v >= 0 ? UP : DOWN, t); };
const css = (c, a = 1) => `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a})`;
const hexToRgb = (h) => { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
const pct = (v, d = 0) => `${(100 * v).toFixed(d)}%`;

function render({ model, el }) {
  el.innerHTML = `
    <div class="lfm3-root">
      <div class="lfm3-bar">
        <button class="lfm3-play" aria-label="Play through the layers">▶</button>
        <div class="lfm3-layer"><input class="lfm3-slider" type="range" min="0" max="23" step="1" /><span class="lfm3-layer-badge"></span></div>
        <label class="lbl-d1"><input class="chk-d1" type="checkbox" /> <span>D1</span></label>
        <label class="lbl-lora"><input class="chk-lora" type="checkbox" /> <span>LoRA</span></label>
        <label class="lbl-guide"><span>Guidance</span>
          <select class="sel-guide"><option value="off">off</option><option value="bad">away from P_bad</option><option value="weak">away from weak P_good</option><option value="base">away from the base model</option></select>
          <select class="sel-weight"><option value="0.25">w 0.25</option><option value="0.5">w 0.5</option><option value="1">w 1.0</option></select>
        </label>
      </div>
      <div class="lfm3-stage">
        <canvas></canvas>
        <div class="lfm3-setting"></div>
        <div class="lfm3-legend"></div>
      </div>
      <div class="lfm3-carousel-wrap">
        <span class="lfm3-car-label">[In / Out]</span>
        <button class="lfm3-nav lfm3-prev" aria-label="Previous example">‹</button>
        <div class="lfm3-carousel"></div>
        <button class="lfm3-nav lfm3-next" aria-label="Next example">›</button>
      </div>
      <div class="lfm3-dots"></div>
    </div>
  `;
  const $ = (q) => el.querySelector(q);
  const canvas = $("canvas"), ctx = canvas.getContext("2d");
  const playBtn = $(".lfm3-play"), slider = $(".lfm3-slider"), layerBadge = $(".lfm3-layer-badge");
  const chkD1 = $(".chk-d1"), chkLora = $(".chk-lora"), lblD1 = $(".lbl-d1"), lblLora = $(".lbl-lora");
  const selGuide = $(".sel-guide"), selWeight = $(".sel-weight"), lblGuide = $(".lbl-guide");
  const settingEl = $(".lfm3-setting"), legendEl = $(".lfm3-legend");
  const carousel = $(".lfm3-carousel"), prevBtn = $(".lfm3-prev"), nextBtn = $(".lfm3-next"), dots = $(".lfm3-dots");

  let yaw = -0.3, pitch = 0.22, zoom = 1.0;
  let curLayer = model.get("layer_idx") || 0;
  let hoverTok = null, scrolledEx = null;
  let scene = null, attnNow = null;         // attnNow: what the attention block is showing (for the cards)
  let blend = 0, blendTo = 0, cursor = 0;   // blend: 0 = convolution lit, 1 = attention lit; cursor: lit slab, eased
  let playing = false, playTimer = 0, raf = 0;

  // ---- what is selected -----------------------------------------------------------------------------------
  const examples = () => model.get("examples") || [];
  const exIdx = () => Math.max(0, Math.min(examples().length - 1, model.get("active_example_idx") || 0));
  const getEx = () => examples()[exIdx()];
  const layerTypes = () => model.get("layer_types") || [];
  const layerType = (li) => layerTypes()[li] || "conv";
  const maxLayer = () => Math.max(0, layerTypes().length - 1);
  const convK = () => model.get("conv_window") || 0;
  const hasD1 = (ex) => !!(ex && ex.d1);
  const isD1 = (ex) => !!model.get("show_d1") && hasD1(ex);
  const view = (ex) => (ex ? (isD1(ex) ? ex.d1 : ex.lfm3) : null);        // the model on screen
  const hasLora = (v) => !!(v && v.attn && v.attn.lora);
  const loraOn = (v) => !!model.get("show_lora") && hasLora(v);
  const guide = (ex) => {                                                // which guidance is drawn, if any
    const g = model.get("guidance") || "off";
    if (g === "off" || !ex) return null;
    if (isD1(ex)) return "base";                                         // the grader has one scheme on disk
    return ex.lfm3.guidance && ex.lfm3.guidance.against[g] ? g : null;
  };
  const weight = () => Number(model.get("guidance_weight") || 0.5);
  const FLOOR = "0.1";
  const nTok = (v) => (v ? v.tokens.length : 0);
  const activeTok = () => {
    const v = view(getEx());
    if (!v) return null;
    const t = hoverTok !== null ? hoverTok : model.get("active_token_idx");
    return Math.max(0, Math.min(nTok(v) - 1, t ?? 0));
  };
  const modelKey = (ex) => (isD1(ex) ? "d1" : "lfm3");
  const layerAttn = (ex, li, variant) => {
    const v = view(ex), a = v && v.attn && v.attn[variant];
    if (!a || !a[String(li)]) return null;
    return decodeLayer(`${ex.id}|${modelKey(ex)}|${li}|${variant}`, a[String(li)], nTok(v));
  };
  const posOf = (v, t) => v.tokens[t].pos;
  const inConvWindow = (v, q, k) => { const d = posOf(v, q) - posOf(v, k); return d >= 0 && d < convK(); };

  // ---- the scene ------------------------------------------------------------------------------------------
  // x: left to right, and also the tokens, in order; y: down the page, the way the data flows; z: depth.
  // Everything is in the same place on every layer; only what is lit and what the faces show changes.
  const SX = -5.4, CX = -2.0, AX = 1.5, VX = 5.6;            // whole model, opened layer, attention, convolution
  const TW = 2.6, D = 0.5;                                     // a box that is one cell per token is TW wide
  const SLAB = 0.17, PITCH = 0.235, Y0 = -2.75;
  const slabY = (i) => Y0 + i * PITCH;

  function buildScene() {
    const ex = getEx(), v = view(ex);
    attnNow = null;
    if (!v) return { boxes: [], wires: [], nodes: [], labels: [] };
    const li = curLayer, lt = layerType(li), convLayer = lt === "conv";
    const nT = nTok(v);
    const d1 = isD1(ex), lora = loraOn(v), g = guide(ex);
    const placement = (v.lora && v.lora.placement) || "all_linear";
    const adapts = (part) => lora && (placement === "all_linear" || placement === part + "_only");
    const boxes = [], wires = [], nodes = [], labels = [];
    const flat = (rows, cols, val) => Array.from({ length: rows }, () => new Array(cols).fill(val));
    const plain = (nx, ny, nz) => ({ front: flat(ny, nx, 0.5), top: flat(nz, nx, 0.5), side: flat(ny, nz, 0.5) });
    const add = (o) => { const b = { z: 0, nx: 1, ny: 1, nz: 1, dz: D, kind: "plain", color: "#64748b", alpha: 1, ...o }; if (!b.grid) b.grid = plain(b.nx, b.ny, b.nz); boxes.push(b); return b; };
    const perToken = (o, values, kind) => add({ ...o, wx: TW, nx: nT, kind, perToken: true, grid: { front: [values], top: [values], side: [[0.5]] } });
    const wire = (pts, color = "#94a3b8", alpha = 1, dashed = false) => wires.push({ pts: pts.map(([x, y]) => [x, y, 0]), color, alpha, dashed });
    const node = (x, y, label, alpha = 1) => nodes.push({ x, y, z: D / 2, label, alpha });
    const label = () => {};  // the picture carries no words: what goes in and what comes out is the text under it
    const tab = (x, y, part, alpha) => { if (adapts(part)) add({ id: "lora", x, y, z: 0.1, wx: 0.12, hy: 0.22, color: "#f59e0b", alpha }); };
    const L = SX - TW / 2 - 0.2, R = SX + TW / 2;                // left and right edges of the whole-model column

    // -- the whole model, top to bottom: tokens in, embedding, 24 layers, final norm, what it answers
    perToken({ id: "tokens", x: SX, y: Y0 - 0.74, hy: 0.16 }, v.tokens.map((t) => (t.seg === "in" ? 0 : 1)), "seg");
    label(L, Y0 - 0.74, d1 ? "reads: the request, the call" : "reads: the request, the call so far", { align: "right" });
    add({ id: "embed", x: SX, y: Y0 - 0.42, wx: TW, hy: 0.14, color: "#6366f1" });
    label(L, Y0 - 0.42, "embedding", { align: "right", alpha: 0.65 });
    const stream = v.stream || {};
    const changed = lora && stream.moved;
    const rows = changed ? stream.moved : stream.size;
    const top = rows ? Math.max(1e-9, ...rows.flat()) : 1;
    layerTypes().forEach((type, i) => {
      const b = perToken({ id: `layer${i}`, layer: i, x: SX, y: slabY(i), hy: SLAB }, rows ? rows[i].map((x) => x / top) : new Array(nT).fill(0.5), rows ? (changed ? "change" : "amount") : "plain");
      add({ id: "type", layer: i, x: R + 0.09, y: slabY(i), wx: 0.12, hy: SLAB, color: type === "conv" ? "#0ea5e9" : type === "full_attention" ? "#7c3aed" : "#db2777" });
      b.cursor = true;
    });
    const yTop = slabY(maxLayer()) + PITCH;
    add({ id: "final_norm", x: SX, y: yTop, wx: TW, hy: 0.12, color: "#94a3b8" });
    label(L, yTop, "norm", { align: "right", alpha: 0.65 });
    const yHead = yTop + 0.4;
    if (d1) {
      const a = (v.answer || {})[lora ? "lora" : "base"];
      if (a) {
        const pb = a.p_broken, t3 = a.top3.map((t) => t[1]);
        const ptr = v.tokens.map((_, t) => { const j = t - v.n_in; return j >= 0 ? Math.max(a.start[j] || 0, a.end[j] || 0) : NaN; });
        const pm = Math.max(1e-9, ...ptr.filter((p) => !Number.isNaN(p)));
        add({ id: "d1_broken", x: SX - 1.05, y: yHead, wx: 0.5, hy: 0.26, nx: 2, kind: "amount", grid: { front: [[pb, 1 - pb]], top: [[pb, 1 - pb]], side: [[0.5]] } });
        add({ id: "d1_type", x: SX - 0.3, y: yHead, wx: 0.8, hy: 0.26, nx: 3, kind: "amount", grid: { front: [t3], top: [t3], side: [[0.5]] } });
        add({ id: "d1_span", x: SX + 0.75, y: yHead, wx: 1.1, hy: 0.26, nx: nT, kind: "amount", perToken: false, grid: { front: [ptr.map((p) => p / pm)], top: [ptr.map((p) => p / pm)], side: [[0.5]] } });
        label(L, yHead, "answers: broken? · which type · where", { align: "right" });
      }
    } else {
      const w = v.written && v.written[lora ? "lora" : "base"];
      perToken({ id: "head", x: SX, y: yHead, hy: 0.26 }, v.tokens.map((_, t) => { const j = t - v.n_in; return j >= 0 && w ? w.p_reference[j] : NaN; }), "amount");
      label(L, yHead, "writes: the next token of the call", { align: "right" });
    }

    // -- guidance: a bar chart standing under the answer, one pair of bars per token of the call
    const xOf = (t, n = nT) => SX - TW / 2 + (t + 0.5) / n * TW;
    if (g && !d1) {
      // One bar per token of the call. Its full height is how far ahead P_good's token is of the runner-up (in
      // nats); the red part is what guidance takes off that lead, a blue cap is what it adds. The runner-up can
      // only win where the grey part reaches the ground, and only under the line (the plausibility floor).
      const G = v.guidance, A = G.against[g], w = weight(), NAT = 0.12, CAP = 13, yB = yHead + 0.45 + CAP * NAT, n = G.margin.length, cell = TW / n;
      const flipped = new Set((A.flips[`${w}|${FLOOR}`] || []).map((f) => f.at));
      const bar = (j, from, to, color) => { const lo = Math.min(CAP, Math.max(0, from)), hi = Math.min(CAP, Math.max(0, to)); if (hi - lo > 0.02) add({ id: "margin", x: xOf(j, n), y: yB - (lo + hi) / 2 * NAT, wx: cell * 0.7, hy: (hi - lo) * NAT, dz: 0.2, color }); };
      G.margin.forEach((m, j) => {
        const left = m - w * A.push[j];
        bar(j, 0, Math.min(m, left), flipped.has(j) ? "#facc15" : "#94a3b8");
        if (left < m) bar(j, left, m, "#ef4444"); else bar(j, m, left, "#3b82f6");
      });
      add({ id: "floor", x: SX, y: yB - Math.log(1 / Number(FLOOR)) * NAT, wx: TW + 0.2, hy: 0.012, dz: 0.3, color: "#f59e0b", alpha: 0.8 });
      add({ id: "floor", x: SX, y: yB + 0.01, wx: TW + 0.2, hy: 0.012, dz: 0.3, color: "#475569" });
      label(L, yB - 0.75, "guided decoding", { align: "right" });
    } else if (g && d1) {
      const a = v.answer, names = v.labels || [], w = 1.5, yB = yHead + 1.65, cell = TW / names.length;
      const guided = a.base.log_type.map((b, i) => b + w * (a.lora.log_type[i] - b));
      const norm = Math.log(guided.reduce((s, x) => s + Math.exp(x), 0));
      names.forEach((_, i) => {
        [[a.base.log_type[i], "#3b82f6", -0.28], [a.lora.log_type[i], "#94a3b8", 0], [guided[i] - norm, "#facc15", 0.28]].forEach(([lp, color, dx]) => {
          const h = Math.max(0.012, Math.exp(lp) * 1.15);
          add({ id: "type_bar", x: xOf(i, names.length) + dx * cell, y: yB - h / 2, wx: cell * 0.26, hy: h, dz: 0.2, color });
        });
      });
      label(L, yB - 0.6, "the failure types, left to right", { align: "right" });
    }

    // -- the lit layer, opened up: norm, operator, norm, add; norm, MLP, norm, add
    const y = { n1: -2.55, n2: 0.62, add1: 1.08, n3: 1.5, w13: 2.08, mul: 2.55, w2: 2.95, n4: 3.4, add2: 3.82 };
    wire([[R + 0.16, slabY(li)], [CX - 1.15, slabY(li)], [CX - 1.15, y.n1 - 0.4], [CX, y.n1 - 0.4], [CX, y.n1 - 0.07]], "#38bdf8");
    add({ id: "operator_norm", x: CX, y: y.n1, wx: 1.2, hy: 0.14, color: "#94a3b8" });
    label(CX + 0.66, y.n1 + 0.2, "norm", { alpha: 0.65 });
    add({ id: "post_operator_norm", x: CX, y: y.n2, wx: 1.2, hy: 0.14, color: "#94a3b8" });
    node(CX, y.add1, "+");
    wire([[CX, y.n2 + 0.07], [CX, y.add1 - 0.1]]);
    wire([[CX, y.n1 - 0.25], [CX - 0.85, y.n1 - 0.25], [CX - 0.85, y.add1], [CX - 0.12, y.add1]], "#38bdf8");
    add({ id: "ffn_norm", x: CX, y: y.n3, wx: 1.2, hy: 0.14, color: "#94a3b8" });
    wire([[CX, y.add1 + 0.1], [CX, y.n3 - 0.07]]);
    add({ id: "w1", x: CX - 0.5, y: y.w13, wx: 0.86, hy: 0.36, color: "#10b981" });
    add({ id: "w3", x: CX + 0.5, y: y.w13, wx: 0.86, hy: 0.36, color: "#10b981" });
    tab(CX - 0.5 + 0.5, y.w13, "mlp", 1); tab(CX + 0.5 + 0.5, y.w13, "mlp", 1);
    wire([[CX, y.n3 + 0.07], [CX - 0.5, y.w13 - 0.18]]); wire([[CX, y.n3 + 0.07], [CX + 0.5, y.w13 - 0.18]]);
    node(CX, y.mul, "×");
    wire([[CX - 0.5, y.w13 + 0.18], [CX - 0.08, y.mul - 0.06]]); wire([[CX + 0.5, y.w13 + 0.18], [CX + 0.08, y.mul - 0.06]]);
    add({ id: "w2", x: CX, y: y.w2, wx: 1.2, hy: 0.3, color: "#10b981" });
    tab(CX + 0.67, y.w2, "mlp", 1);
    wire([[CX, y.mul + 0.1], [CX, y.w2 - 0.15]]);
    label(CX + 1.1, y.mul, "MLP", { alpha: 0.65 });
    add({ id: "post_ffn_norm", x: CX, y: y.n4, wx: 1.2, hy: 0.14, color: "#94a3b8" });
    wire([[CX, y.w2 + 0.15], [CX, y.n4 - 0.07]]);
    node(CX, y.add2, "+");
    wire([[CX, y.n4 + 0.07], [CX, y.add2 - 0.1]]);
    wire([[CX, y.add1 + 0.2], [CX - 1.0, y.add1 + 0.2], [CX - 1.0, y.add2], [CX - 0.12, y.add2]], "#34d399");

    // -- attention and convolution, both always there; one is lit, and the light crosses over as the layer changes
    const aLit = 0.14 + 0.86 * blend, cLit = 0.14 + 0.86 * (1 - blend);
    const aCol = lt === "sliding_attention" ? "#db2777" : "#7c3aed";
    const yq = -3.25, yc = -1.15;                                             // q k v; centre of the attention square
    wire([[CX + 0.6, y.n1], [CX + 1.0, y.n1], [CX + 1.0, yq - 0.42], [AX, yq - 0.42], [AX, yq - 0.16]], aCol, aLit);
    [["q", -0.9, aCol], ["k", 0, "#a78bfa"], ["v", 0.9, "#a78bfa"]].forEach(([name, dx, color]) => {
      add({ id: name + "_proj", x: AX + dx, y: yq, wx: 0.7, hy: 0.28, color, alpha: aLit });
      tab(AX + dx + 0.41, yq, "attention", aLit);
      label(AX + dx - 0.05, yq - 0.26, name, { alpha: 0.7 * aLit, align: "center" });
    });
    add({ id: "q_norm", x: AX - 0.9, y: yq + 0.34, wx: 0.7, hy: 0.09, color: "#94a3b8", alpha: aLit });
    add({ id: "k_norm", x: AX, y: yq + 0.34, wx: 0.7, hy: 0.09, color: "#94a3b8", alpha: aLit });
    let grid = null, coreKind = "plain";
    const tF = activeTok();
    if (!convLayer) {
      const base = layerAttn(ex, li, "base"), tuned = layerAttn(ex, li, "lora");
      if (lora && base && tuned) {
        let m = 1e-9;
        for (let q = 0; q < nT; q++) for (let k = 0; k < nT; k++) m = Math.max(m, Math.abs(tuned.avg[q][k] - base.avg[q][k]));
        const cell = (a, b, q, k) => (base.causal && k > q ? NaN : (a[q][k] - b[q][k]) / m);
        grid = {
          front: Array.from({ length: nT }, (_, q) => Array.from({ length: nT }, (_, k) => cell(tuned.avg, base.avg, q, k))),
          top: tuned.groups.map((gr, gi) => Array.from({ length: nT }, (_, k) => cell(gr, base.groups[gi], tF, k))),
          side: [[0.5]],
        };
        coreKind = "change";
        attnNow = { kind: "change", row: (q) => Array.from({ length: nT }, (_, k) => (base.causal && k > q ? 0 : (tuned.avg[q][k] - base.avg[q][k]) / m)), raw: (q, k) => tuned.avg[q][k] - base.avg[q][k], scale: m };
      } else if (base) {
        const rowMax = (mat, q) => Math.max(1e-9, ...mat[q]);
        const cell = (mat, q, k) => (base.causal && k > q ? NaN : mat[q][k] / rowMax(mat, q));
        grid = {
          front: Array.from({ length: nT }, (_, q) => Array.from({ length: nT }, (_, k) => cell(base.avg, q, k))),
          top: base.groups.map((gr) => Array.from({ length: nT }, (_, k) => cell(gr, tF, k))),
          side: [[0.5]],
        };
        coreKind = "amount";
        attnNow = { kind: "amount", row: (q) => Array.from({ length: nT }, (_, k) => (base.causal && k > q ? 0 : base.avg[q][k] / rowMax(base.avg, q))), raw: (q, k) => base.avg[q][k] };
      }
    }
    add({ id: "attention", x: AX, y: yc, wx: TW, hy: TW, nx: nT, ny: nT, nz: grid ? grid.top.length : 4, dz: 0.6, color: aCol, alpha: aLit, kind: coreKind, grid: grid || plain(nT, nT, 4), square: true });
    label(AX - TW / 2, yc + TW / 2 + 0.16, convLayer ? "attention (not in this layer)" : `attention: each row is a token looking back${lt === "full_attention" ? "" : `, last ${ex.sliding_window || ""}`}`, { alpha: Math.max(0.35, aLit) });
    wire([[AX - 0.9, yq + 0.4], [AX - 0.9, yc - TW / 2]], aCol, aLit); wire([[AX, yq + 0.4], [AX, yc - TW / 2]], "#a78bfa", aLit); wire([[AX + 0.9, yq + 0.14], [AX + 0.9, yc - TW / 2]], "#a78bfa", aLit);
    add({ id: "gate_proj", x: AX + TW / 2 + 0.45, y: yc, wx: 0.5, hy: 0.3, color: "#64748b", alpha: aLit });
    tab(AX + TW / 2 + 0.78, yc, "attention", aLit);
    label(AX + TW / 2 + 0.2, yc - 0.28, "gate", { alpha: 0.7 * aLit });
    const yx = yc + TW / 2 + 0.5;
    node(AX, yx, "×", aLit);
    wire([[AX, yc + TW / 2], [AX, yx - 0.1]], aCol, aLit);
    wire([[AX + TW / 2 + 0.45, yc + 0.15], [AX + TW / 2 + 0.45, yx], [AX + 0.12, yx]], "#64748b", aLit);
    add({ id: "o_proj", x: AX, y: yx + 0.42, wx: 1.1, hy: 0.24, color: aCol, alpha: aLit });
    tab(AX + 0.62, yx + 0.42, "attention", aLit);
    wire([[AX, yx + 0.54], [AX, yx + 0.8], [CX + 1.0, yx + 0.8], [CX + 1.0, y.n2], [CX + 0.6, y.n2]], aCol, aLit);

    wire([[CX + 0.6, y.n1 - 0.05], [CX + 0.85, y.n1 - 0.05], [CX + 0.85, yq - 0.6], [VX, yq - 0.6], [VX, yq - 0.16]], "#0ea5e9", cLit);
    add({ id: "in_proj", x: VX, y: yq, wx: TW, hy: 0.28, color: "#38bdf8", alpha: cLit });
    tab(VX + TW / 2 + 0.07, yq, "conv", cLit);
    [["B", -0.9, "#0284c7"], ["x", 0, "#0ea5e9"], ["C", 0.9, "#0284c7"]].forEach(([name, dx, color]) => {
      add({ id: "conv_" + name, x: VX + dx, y: yq + 0.6, wx: 0.7, hy: 0.26, color, alpha: cLit });
      label(VX + dx - 0.05, yq + 0.34, name, { alpha: 0.7 * cLit, align: "center" });
      wire([[VX + dx, yq + 0.14], [VX + dx, yq + 0.47]], color, cLit);
    });
    node(VX - 0.45, yq + 1.15, "×", cLit);
    wire([[VX - 0.9, yq + 0.73], [VX - 0.52, yq + 1.08]], "#0284c7", cLit); wire([[VX, yq + 0.73], [VX - 0.38, yq + 1.08]], "#0ea5e9", cLit);
    perToken({ id: "conv1d", x: VX, y: yc + 0.1, hy: 0.7, color: "#2563eb", alpha: cLit, window: convLayer }, new Array(nT).fill(0.5), "plain").color = "#2563eb";
    wire([[VX - 0.45, yq + 1.25], [VX - 0.45, yc - 0.25]], "#2563eb", cLit);
    label(VX - TW / 2, yc + 0.62, convLayer ? `convolution: each token sees its last ${convK()}` : "convolution (not in this layer)", { alpha: Math.max(0.35, cLit) });
    node(VX, yx, "×", cLit);
    wire([[VX, yc + 0.45], [VX, yx - 0.1]], "#2563eb", cLit);
    wire([[VX + 0.9, yq + 0.73], [VX + 1.55, yq + 0.73], [VX + 1.55, yx], [VX + 0.12, yx]], "#0284c7", cLit);
    add({ id: "out_proj", x: VX, y: yx + 0.42, wx: 1.1, hy: 0.24, color: "#38bdf8", alpha: cLit });
    tab(VX + 0.62, yx + 0.42, "conv", cLit);
    wire([[VX, yx + 0.54], [VX, yx + 1.0], [CX + 0.85, yx + 1.0], [CX + 0.85, y.n2 + 0.05], [CX + 0.6, y.n2 + 0.05]], "#0ea5e9", cLit);
    return { boxes, wires, nodes, labels };
  }

  // ---- drawing --------------------------------------------------------------------------------------------
  function rotatePt(x, y, z) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const x1 = x * cy - z * sy, z1 = x * sy + z * cy;
    return { x: x1, y: (-y) * cp - z1 * sp, z: (-y) * sp + z1 * cp };
  }

  function computeCamera(W, H) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    const padX = 20, padTop = 44, padBottom = 10, fov = 16.0, baseDist = 26.0;
    const pts = [];
    for (const b of scene.boxes) {
      const hx = b.wx / 2, hy = b.hy / 2, hz = b.dz / 2;
      for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) pts.push([b.x + sx * hx, b.y + sy * hy, (b.z || 0) + sz * hz]);
    }
    for (const [x, y, z] of pts) {
      const r = rotatePt(x, y, z), persp = fov / Math.max(3.0, baseDist + r.z);
      const px = r.x * persp, py = -r.y * persp;
      if (px < minX) minX = px; if (px > maxX) maxX = px;
      if (py < minY) minY = py; if (py > maxY) maxY = py;
    }
    const availW = Math.max(80, W - padX * 2), availH = Math.max(80, H - padTop - padBottom);
    const fit = Math.min(availW / Math.max(0.1, maxX - minX), availH / Math.max(0.1, maxY - minY)) * zoom;
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    return {
      project(x, y, z) {
        const r = rotatePt(x, y, z), persp = fov / Math.max(3.0, baseDist + r.z);
        return { sx: W / 2 + (r.x * persp - cx) * fit, sy: padTop + availH / 2 + (-r.y * persp - cy) * fit, depth: r.z };
      },
    };
  }

  function cellColor(kind, val, rgb, shade) {
    if (Number.isNaN(val)) return [15, 23, 42];
    if (kind === "amount") return amount(val).map((c) => c * (0.7 + 0.3 * shade));
    if (kind === "change") return change(val).map((c) => c * (0.7 + 0.3 * shade));
    if (kind === "seg") return (val < 0.5 ? [45, 212, 191] : [251, 146, 60]).map((c) => c * 0.8 * shade);
    return rgb.map((c) => Math.min(255, c * shade * 0.85));
  }

  function drawFace(p00, p10, p11, p01, grid2d, kind, rgb, shade, alpha, hiR, hiC) {
    const rows = grid2d.length, cols = grid2d[0].length;
    const at = (u, v) => [
      (p00.sx + (p10.sx - p00.sx) * u) * (1 - v) + (p01.sx + (p11.sx - p01.sx) * u) * v,
      (p00.sy + (p10.sy - p00.sy) * u) * (1 - v) + (p01.sy + (p11.sy - p01.sy) * u) * v,
    ];
    const focus = (hiR && hiR.size > 0) || (hiC && hiC.size > 0);
    const dense = rows * cols > 600;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const a = at(c / cols, r / rows), b = at((c + 1) / cols, r / rows), d = at((c + 1) / cols, (r + 1) / rows), e = at(c / cols, (r + 1) / rows);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(d[0], d[1]); ctx.lineTo(e[0], e[1]); ctx.closePath();
        const inFocus = (!hiR || hiR.has(r)) && (!hiC || hiC.has(c));
        const col = cellColor(kind, grid2d[r][c], rgb, shade * (focus && !inFocus ? 0.72 : 1));
        ctx.fillStyle = css(col, alpha);
        ctx.fill();
        if (dense && !(focus && inFocus)) continue;
        ctx.strokeStyle = focus && inFocus ? `rgba(255,255,255,${0.7 * alpha})` : `rgba(15,23,42,${0.3 * alpha})`;
        ctx.lineWidth = focus && inFocus ? 0.9 : 0.4;
        ctx.stroke();
      }
    }
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
    ctx.save(); ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);
    if (!scene || !scene.boxes.length) { ctx.restore(); return; }
    const cam = computeCamera(W, H);
    const ink = getComputedStyle(el).color || "#0f172a";
    const tF = activeTok(), v = view(getEx());
    for (const w of scene.wires) {
      ctx.save(); ctx.globalAlpha = w.alpha; ctx.strokeStyle = w.color; ctx.lineWidth = w.dashed ? 1.1 : 1.4;
      if (w.dashed) ctx.setLineDash([4, 3]);
      ctx.beginPath();
      w.pts.forEach((pt, i) => { const p = cam.project(pt[0], pt[1], pt[2]); if (i === 0) ctx.moveTo(p.sx, p.sy); else ctx.lineTo(p.sx, p.sy); });
      ctx.stroke(); ctx.restore();
    }
    const order = scene.boxes.map((b) => ({ b, d: cam.project(b.x, b.y, b.z || 0).depth })).sort((a, b) => b.d - a.d);
    for (const { b } of order) {
      const z = b.z || 0, hx = b.wx / 2, hy = b.hy / 2, hz = b.dz / 2;
      const lit = b.layer === undefined ? 0 : Math.max(0, 1 - Math.abs(b.layer - cursor));   // the slab the cursor is on
      const grow = 1;
      const c = [
        cam.project(b.x - hx * grow, b.y - hy, z + hz), cam.project(b.x + hx * grow, b.y - hy, z + hz), cam.project(b.x + hx * grow, b.y + hy, z + hz), cam.project(b.x - hx * grow, b.y + hy, z + hz),
        cam.project(b.x - hx * grow, b.y - hy, z - hz), cam.project(b.x + hx * grow, b.y - hy, z - hz), cam.project(b.x + hx * grow, b.y + hy, z - hz), cam.project(b.x - hx * grow, b.y + hy, z - hz),
      ];
      const rgb = hexToRgb(b.color);
      // light the focused token's column; in the convolution, the tokens it can see; in the attention square, its row
      let cols = null, rowsLit = null;
      if (tF !== null && b.perToken) {
        cols = new Set([tF]);
        if (b.window) for (let k = 0; k < b.nx; k++) if (inConvWindow(v, tF, k)) cols.add(k);
      }
      if (tF !== null && b.square && b.kind !== "plain") rowsLit = new Set([tF]);
      const faces = [
        { idx: [0, 1, 2, 3], grid: b.grid.front, kind: b.kind, shade: 1.0, hiR: rowsLit, hiC: cols },
        { idx: [5, 4, 7, 6], grid: b.grid.front, kind: "plain", shade: 0.6, hiR: null, hiC: null },
        { idx: [4, 5, 1, 0], grid: b.grid.top, kind: b.kind, shade: 1.12, hiR: null, hiC: cols },
        { idx: [3, 2, 6, 7], grid: b.grid.top, kind: "plain", shade: 0.6, hiR: null, hiC: null },
        { idx: [1, 5, 6, 2], grid: b.grid.side, kind: "plain", shade: 0.82, hiR: null, hiC: null },
        { idx: [4, 0, 3, 7], grid: b.grid.side, kind: "plain", shade: 0.74, hiR: null, hiC: null },
      ];
      const seen = faces.map((f) => {
        const pts = f.idx.map((i) => c[i]);
        const cross = (pts[1].sx - pts[0].sx) * (pts[2].sy - pts[0].sy) - (pts[1].sy - pts[0].sy) * (pts[2].sx - pts[0].sx);
        return { ...f, pts, cross, depth: pts.reduce((a, p) => a + p.depth, 0) / 4 };
      }).filter((f) => f.cross > 0).sort((a, b) => b.depth - a.depth);
      for (const f of seen) {
        drawFace(f.pts[0], f.pts[1], f.pts[2], f.pts[3], f.grid, f.kind, rgb, f.shade, b.alpha, f.hiR, f.hiC);
        ctx.beginPath(); ctx.moveTo(f.pts[0].sx, f.pts[0].sy); ctx.lineTo(f.pts[1].sx, f.pts[1].sy); ctx.lineTo(f.pts[2].sx, f.pts[2].sy); ctx.lineTo(f.pts[3].sx, f.pts[3].sy); ctx.closePath();
        ctx.strokeStyle = lit > 0.05 ? `rgba(250,204,21,${lit})` : `rgba(15,23,42,${0.5 * b.alpha})`;
        ctx.lineWidth = lit > 0.05 ? 1.6 : 0.7;
        ctx.stroke();
      }
    }
    for (const n of scene.nodes) {
      const p = cam.project(n.x, n.y, n.z);
      ctx.save(); ctx.globalAlpha = n.alpha;
      ctx.beginPath(); ctx.arc(p.sx, p.sy, 6.5, 0, Math.PI * 2);
      ctx.fillStyle = "#0f172a"; ctx.fill(); ctx.strokeStyle = "#cbd5e1"; ctx.lineWidth = 1.2; ctx.stroke();
      ctx.fillStyle = "#f8fafc"; ctx.font = "bold 10px ui-monospace, SFMono-Regular, monospace";
      ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(n.label, p.sx, p.sy + 0.5);
      ctx.restore();
    }
    ctx.font = "10px ui-monospace, SFMono-Regular, monospace"; ctx.textBaseline = "middle";
    for (const l of scene.labels) {
      const p = cam.project(l.x, l.y, l.z);
      ctx.save(); ctx.globalAlpha = l.alpha; ctx.fillStyle = ink; ctx.textAlign = l.align;
      ctx.fillText(l.text, p.sx + (l.align === "right" ? -4 : 4), p.sy);
      ctx.restore();
    }
    ctx.restore();
  }

  // ---- words around the picture ---------------------------------------------------------------------------
  function updateText() {
    const ex = getEx(), v = view(ex), li = curLayer, lt = layerType(li);
    slider.max = String(maxLayer()); slider.value = String(li);
    layerBadge.textContent = `${li + 1} / ${layerTypes().length || 24} · ${lt === "conv" ? "convolution" : lt === "full_attention" ? "attention, all tokens" : "attention, sliding window"}`;
    lblD1.classList.toggle("off", !hasD1(ex)); chkD1.disabled = !hasD1(ex);
    lblLora.classList.toggle("off", !hasLora(v)); chkLora.disabled = !hasLora(v);
    chkD1.checked = isD1(ex); chkLora.checked = loraOn(v);
    const d1 = isD1(ex), lora = loraOn(v), g = guide(ex);
    const G = ex && ex.lfm3 && ex.lfm3.guidance;
    for (const o of selGuide.options) o.disabled = d1 ? !(o.value === "off" || o.value === "base") : !(o.value === "off" || (G && G.against[o.value]));
    selGuide.value = g || "off"; selWeight.value = String(weight()); selWeight.style.display = g && !d1 ? "" : "none";
    if (!v) { settingEl.textContent = "no examples"; legendEl.textContent = ""; return; }
    const who = d1 ? "D1 · the grader" : "LFM3 · the target";
    const adapter = v.lora ? `${v.lora.placement.replace("_", " ")}, rank ${v.lora.rank}` : "";
    settingEl.innerHTML = `<strong>${who}</strong> · ${lora ? `with its LoRA (${adapter})` : "base model, no LoRA"}`;
    const bar = (stops) => `<i style="background: linear-gradient(90deg, ${stops.join(",")})"></i>`;
    const moved = v.stream && v.stream.moved ? Math.max(...v.stream.moved.flat()) : null;
    let lines = lora ? [`less ${bar([css(DOWN), css(DARK), css(UP)])} more`] : [`none ${bar([css(DARK), css(MID), css(HOT)])} most`];
    if (g && !d1) lines = [`<span style="color:#94a3b8">■</span> lead · <span style="color:#ef4444">■</span> taken · <span style="color:#3b82f6">■</span> added · <span style="color:#f59e0b">▬</span> floor`];
    else if (g && d1) lines = [`<span style="color:#3b82f6">■</span> base · <span style="color:#94a3b8">■</span> LoRA · <span style="color:#ca8a04">■</span> guided`];
    legendEl.innerHTML = lines.filter(Boolean).join("<br/>");
  }

  function renderCarousel() {
    const exs = examples(), cur = exIdx(), convLayer = layerType(curLayer) === "conv", tF = activeTok();
    carousel.innerHTML = "";
    exs.forEach((ex, idx) => {
      const sel = idx === cur, v = view(ex), d1 = isD1(ex), lora = loraOn(v);
      const card = document.createElement("div");
      card.className = `lfm3-card ${sel ? "active" : "inactive"}`;
      if (!sel) card.onclick = () => selectExample(idx);
      const answer = d1 && v.answer ? v.answer[lora ? "lora" : "base"] : null;
      const written = !d1 && v.written ? v.written[lora ? "lora" : "base"] : null;
      const row = (title, toks, offset, paint, text = "") => {
        let cursor = 0;  // the tokens are laid back into the original text, so spaces and line breaks are the real ones
        const r = document.createElement("div"); r.className = "lfm3-row";
        if (offset >= v.n_in) r.classList.add("lfm3-code");
        toks.forEach((tok, i) => {
          const gi = offset + i, sp = document.createElement("span");
          const raw = (typeof tok === "string" ? tok : tok.t).replace(/⏎/g, "\n").replace(/⇥/g, "\t");
          const at = text.indexOf(raw, cursor);
          if (at >= cursor && at - cursor < 40) { if (at > cursor) r.appendChild(document.createTextNode(text.slice(cursor, at))); cursor = at + raw.length; }
          else if (i) r.appendChild(document.createTextNode(" "));
          sp.className = "lfm3-tok"; sp.textContent = raw;
          if (sel && offset >= 0) {
            sp.style.cursor = "pointer";
            let val = 0;
            if (tF !== null) val = convLayer ? (inConvWindow(v, tF, gi) ? 0.5 : 0) : attnNow ? attnNow.row(tF)[gi] : 0;
            if (gi === tF) { sp.style.outline = "1.5px solid #6366f1"; }
            const kind = !convLayer && attnNow ? attnNow.kind : "amount";
            if (Math.abs(val) > 0.06) { sp.style.background = css(kind === "change" ? change(val) : amount(val), 0.9); sp.style.color = "#fff"; }
            if (!convLayer && attnNow && tF !== null) sp.title = attnNow.kind === "change" ? `LoRA − base, from the outlined token: ${attnNow.raw(tF, gi) >= 0 ? "+" : ""}${attnNow.raw(tF, gi).toFixed(4)}` : `attention from the outlined token: ${pct(attnNow.raw(tF, gi), 1)}`;
            sp.onmouseenter = () => { hoverTok = gi; refresh(); };
            sp.onmouseleave = () => { hoverTok = null; refresh(); };
            sp.onclick = (e) => { e.stopPropagation(); model.set("active_token_idx", gi); model.save_changes(); };
          }
          if (paint) paint(sp, i);
          r.appendChild(sp);
        });
        if (cursor < text.length) { const rest = document.createElement("span"); rest.style.opacity = "0.5"; rest.textContent = text.slice(cursor); r.appendChild(rest); }
        return r;
      };
      const ins = v.tokens.slice(0, v.n_in), outs = v.tokens.slice(v.n_in);
      if (d1) {
        const under = (sp, j) => { const p = answer ? Math.max(answer.start[j] || 0, answer.end[j] || 0) : 0; if (p > 0.02) sp.style.boxShadow = `inset 0 -2px 0 rgba(239,68,68,${Math.min(1, 0.25 + p).toFixed(2)})`; };
        card.appendChild(row("", ins, 0, null, ex.request));
        card.appendChild(row("", outs, v.n_in, under, ex.code));
        if (answer) { const out = document.createElement("div"); out.className = "lfm3-out"; out.textContent = `${answer.verdict === "broken" ? "broken" : "not broken"} · ${answer.failure_type} ${pct(answer.top3[0][1])}`; card.appendChild(out); }
      } else {
        card.appendChild(row("", ins, 0, null, ex.request));
        card.appendChild(row("", outs, v.n_in, written ? (sp, j) => { if (written.top[j] !== outs[j].t && written.p_reference[j] < 0.5) sp.style.textDecoration = "underline wavy #ef4444"; } : null, ex.code));
      }
      carousel.appendChild(card);
      if (sel && scrolledEx !== idx) { scrolledEx = idx; carousel.scrollLeft = Math.max(0, card.offsetLeft - carousel.offsetLeft); }
    });
    prevBtn.disabled = cur <= 0; nextBtn.disabled = cur >= exs.length - 1;
    dots.innerHTML = "";
    exs.forEach((_, idx) => { const d = document.createElement("i"); if (idx === cur) d.className = "on"; d.onclick = () => selectExample(idx); dots.appendChild(d); });
  }

  // ---- moving between layers ------------------------------------------------------------------------------
  // The layer changes at once; what eases is the light: the cursor slides up the stack and the convolution and
  // attention blocks fade over to each other.
  function animate() {
    cancelAnimationFrame(raf);
    const step = () => {
      const dc = curLayer - cursor, db = blendTo - blend;
      cursor = Math.abs(dc) < 0.01 ? curLayer : cursor + dc * 0.22;
      blend = Math.abs(db) < 0.01 ? blendTo : blend + db * 0.2;
      scene = buildScene(); draw();
      if (cursor !== curLayer || blend !== blendTo) raf = requestAnimationFrame(step);
    };
    step();
  }
  function setLayer(li, tell = true) {
    curLayer = Math.max(0, Math.min(maxLayer(), li));
    blendTo = layerType(curLayer) === "conv" ? 0 : 1;
    animate(); updateText(); renderCarousel();
    if (tell && curLayer !== model.get("layer_idx")) { model.set("layer_idx", curLayer); model.save_changes(); }
  }
  function stop() { playing = false; clearInterval(playTimer); playBtn.textContent = "▶"; if (curLayer !== model.get("layer_idx")) { model.set("layer_idx", curLayer); model.save_changes(); } }
  function play() {
    if (playing) return stop();
    playing = true; playBtn.textContent = "❚❚";
    if (curLayer >= maxLayer()) setLayer(0, false);
    playTimer = setInterval(() => { if (curLayer >= maxLayer()) stop(); else setLayer(curLayer + 1, false); }, 900);
  }
  playBtn.onclick = play;
  slider.oninput = () => { if (playing) stop(); setLayer(Number(slider.value)); };

  function refresh() { scene = buildScene(); updateText(); renderCarousel(); draw(); }
  function selectExample(idx) {
    const exs = examples();
    if (idx < 0 || idx >= exs.length) return;
    hoverTok = null;
    model.set("active_example_idx", idx);
    model.set("active_token_idx", view(exs[idx]).n_in);
    model.save_changes();
  }
  prevBtn.onclick = () => selectExample(exIdx() - 1);
  nextBtn.onclick = () => selectExample(exIdx() + 1);
  chkD1.onchange = () => { model.set("show_d1", chkD1.checked); model.save_changes(); };
  chkLora.onchange = () => { model.set("show_lora", chkLora.checked); model.save_changes(); };
  selGuide.onchange = () => { model.set("guidance", selGuide.value); model.save_changes(); };
  selWeight.onchange = () => { model.set("guidance_weight", Number(selWeight.value)); model.save_changes(); };

  let drag = false, lx = 0, ly = 0;
  canvas.addEventListener("pointerdown", (e) => { drag = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    yaw += (e.clientX - lx) * 0.008; pitch = Math.max(-1.25, Math.min(1.25, pitch + (e.clientY - ly) * 0.008));
    lx = e.clientX; ly = e.clientY; draw();
  });
  canvas.addEventListener("pointerup", (e) => { drag = false; canvas.releasePointerCapture(e.pointerId); });

  const sync = () => {
    const ex = getEx(), v = view(ex);
    if (v && (model.get("active_token_idx") ?? 0) >= nTok(v)) { model.set("active_token_idx", v.n_in); model.save_changes(); }
    if (!playing) { curLayer = Math.max(0, Math.min(maxLayer(), model.get("layer_idx") || 0)); }
    blendTo = layerType(curLayer) === "conv" ? 0 : 1;
    animate(); updateText(); renderCarousel();
  };
  for (const k of ["examples", "layer_types", "conv_window", "active_example_idx", "layer_idx", "show_d1", "show_lora", "active_token_idx", "guidance", "guidance_weight"]) model.on(`change:${k}`, sync);
  model.on("change:examples", () => DECODED.clear());
  const ro = new ResizeObserver(() => draw());
  ro.observe(canvas);
  cursor = curLayer; blend = blendTo = layerType(curLayer) === "conv" ? 0 : 1;
  sync();
  return () => { ro.disconnect(); clearInterval(playTimer); cancelAnimationFrame(raf); };
}
export default { render };
"""


class ActivationWidget(anywidget.AnyWidget):
    """3D view of the 24 layers both models share, driven by what was captured: four settings (the target and the
    grader, each with and without its LoRA) over the same examples, and guided decoding token by token."""

    examples = traitlets.List([]).tag(sync=True)
    layer_types = traitlets.List([]).tag(sync=True)
    conv_window = traitlets.Int(0).tag(sync=True)
    layer_idx = traitlets.Int(0).tag(sync=True)
    active_example_idx = traitlets.Int(0).tag(sync=True)
    active_token_idx = traitlets.Int(0).tag(sync=True)
    show_d1 = traitlets.Bool(False).tag(sync=True)
    show_lora = traitlets.Bool(True).tag(sync=True)
    guidance = traitlets.Unicode("off").tag(sync=True)
    guidance_weight = traitlets.Float(0.5).tag(sync=True)

    _css = _CSS
    _esm = _ESM

    def __init__(self, examples=None, layer_types=None, conv_window=0, **kwargs):
        examples = list(examples or [])
        layer_types = list(layer_types or [])
        if "active_token_idx" not in kwargs and examples:
            kwargs["active_token_idx"] = examples[0]["lfm3"]["n_in"]
        super().__init__(examples=examples, layer_types=layer_types, conv_window=int(conv_window), **kwargs)


def model_layout(defs) -> dict:
    """Layer types and the conv kernel size read from the checkpoints' configs (no weights loaded). Both models
    are built on the same 24-layer backbone; the function checks that instead of assuming it."""
    def text_config(folder):
        cfg = json.loads((Path(defs["model_path"](defs["MODEL_FOLDERS"][folder])) / "config.json").read_text(encoding="utf-8"))
        return cfg.get("text_config", cfg)

    target, grader = text_config("target"), text_config("grader")
    if list(target["layer_types"]) != list(grader["layer_types"]):
        raise ValueError("the target and the grader do not share a layer layout; the view draws one stack for both")
    return {
        "layer_types": list(target["layer_types"]), "conv_window": int(target.get("conv_L_cache", 0)),
        "shape": {k: target.get(k) for k in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "vocab_size", "sliding_window")},
    }


def save_payload(path, examples, layout) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps({"examples": examples, **layout}, separators=(",", ":"))
    path.write_text(data, encoding="utf-8")
    return len(data)


def load_payload(path) -> dict | None:
    """The saved capture, or None when there is none or it was written by an older version of this file."""
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("examples") and "lfm3" in payload["examples"][0] else None
