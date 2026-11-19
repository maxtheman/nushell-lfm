"""Capture code taken out of the 3D view when it became LFM3-only: D1 (the grader) over the same examples, and
what guided decoding has to work with at each token of a call (`runs/target/guidance_tokens.json`, the numbers
behind the guidance section of notes/lab-notes.md). Kept as written for the follow-up; `capture_activations`
here is the old two-model version and takes the old `defs` dictionary of pipeline functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nushell_lfm.model_view import (
    MAX_CODE_TOKENS, _display, _free, _next_token_logp, _pack_variant, _read_adapter, _record, _stream_summary, _visible_tokens,
)

GUIDANCE_WEIGHTS = (0.25, 0.5, 1.0)
GUIDANCE_FLOORS = (0.03, 0.1, 0.3)


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
