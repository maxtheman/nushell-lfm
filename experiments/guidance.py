"""Guided decoding for the target, and the experiments around it: an adapter trained on wrong calls (P_bad),
steering away from it, from a weaker adapter, from the base model, or from the prompt without its request;
cross-fitting to collect the adapter's own mistakes; and the same adapter on low-bit base weights."""

from collections import Counter
import gc
import json
import math
import random
import time

from nushell_lfm.calls import parse_tool_call
from nushell_lfm.target import (
    TARGET_DIR, TARGET_SYSTEM, TARGET_TOOLS, _encode, _fit, _load as _load_plain, _mean_loss, _rate, score_call, spread, target_prompt,
)

from grader import GRADER_DIR, grade_calls

def _ternarize(lm, group_size: int = 64) -> None:
    """Replace every linear weight by its ternary version: within each group of `group_size` weights in a row,
    each weight becomes −s, 0 or +s. Weights smaller than 0.7 × the group's mean size go to 0 and s is the mean
    size of the rest (ternary weight networks). The values are kept as ordinary floats, so this measures what
    ternary weights do to the answers; it does not shrink the model in memory. The embedding goes to 4 bits."""
    import mlx.core as mx
    import mlx.nn as nn

    def ternary(w):
        out_dim, in_dim = w.shape
        groups = w.astype(mx.float32).reshape(out_dim, in_dim // group_size, group_size)
        size = mx.abs(groups)
        keep = size > 0.7 * size.mean(axis=-1, keepdims=True)
        scale = (size * keep).sum(axis=-1, keepdims=True) / mx.maximum(keep.sum(axis=-1, keepdims=True), 1)
        return (mx.sign(groups) * keep * scale).reshape(out_dim, in_dim).astype(w.dtype)

    for _, module in lm.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[1] % group_size == 0:
            module.weight = ternary(module.weight)
            mx.eval(module.weight)
    nn.quantize(lm, group_size=group_size, bits=4, class_predicate=lambda _, m: isinstance(m, nn.Embedding))

def _load(cfg: dict, seed: int):
    """The target's loader; with `cfg["bits"]` the base weights go back to full precision and are quantised again."""
    def requantise(model):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.utils import dequantize_model

        dequantize_model(model.language_model)
        if cfg["bits"] == "ternary":
            _ternarize(model.language_model, cfg.get("group_size", 64))
        else:
            nn.quantize(model.language_model, group_size=cfg.get("group_size", 64), bits=cfg["bits"])
        mx.eval(model.language_model.parameters())

    return _load_plain(cfg, seed, requantise if cfg.get("bits") else None)

def _generate(lm, tokenizer, row: dict, max_new: int, good=None, bad=None, weight: float = 0.0, floor: float = 0.1, bad_row: dict | None = None, blank_weight: float = 0.0) -> str:
    """Greedy decoding of one tool call. With `good` and `bad` adapter weights it is guided: each step runs
    both adapters, each with its own KV cache, and scores a token by
    log p_good + weight · (log p_good − log p_bad), over the tokens P_good itself finds plausible
    (probability at least `floor` × its top token). That restriction stops the difference from promoting
    tokens that are merely unlikely under P_bad.

    With `bad_row` the contrast reads a different prompt (for example the same row with the request blanked);
    without `bad` it then uses the same adapter, so the difference is what the request itself contributes.
    `blank_weight` adds that request contrast on top of an adapter contrast, so both steer at once."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    def use(weights):
        lm.update(tree_unflatten(list(weights.items())))

    def step(tokens, cache):
        out = lm(tokens, cache=cache)
        return nn.log_softmax((out.logits if hasattr(out, "logits") else out)[:, -1, :].astype(mx.float32), axis=-1)

    stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    tokens = mx.array([tokenizer.encode(target_prompt(tokenizer, row), add_special_tokens=False)])
    other_tokens = tokens if bad_row is None else mx.array([tokenizer.encode(target_prompt(tokenizer, bad_row), add_special_tokens=False)])
    blank_tokens = mx.array([tokenizer.encode(target_prompt(tokenizer, {**row, "speech": ""}), add_special_tokens=False)]) if blank_weight else None
    cache_good, cache_bad, cache_blank, written = lm.make_cache(), lm.make_cache(), lm.make_cache(), []
    for _ in range(max_new):
        if good is not None:
            use(good)
        scores = step(tokens, cache_good)
        blank = step(blank_tokens, cache_blank) if blank_weight else None
        if bad is not None or bad_row is not None or blank_weight:
            contrast = scores
            if bad is not None or bad_row is not None:
                if bad is not None:
                    use(bad)
                contrast = step(other_tokens, cache_bad)
            keep = scores >= mx.max(scores, axis=-1, keepdims=True) + math.log(floor)
            guided = scores + weight * (scores - contrast) + (blank_weight * (scores - blank) if blank_weight else 0.0)
            scores = mx.where(keep, guided, mx.array(-1e9, dtype=mx.float32))
        token = int(mx.argmax(scores, axis=-1).item())
        if token in stop:
            break
        written.append(token)
        if tokenizer.decode(written).rstrip().endswith("</tool_call>"):
            break
        tokens = other_tokens = blank_tokens = mx.array([[token]])
    return tokenizer.decode(written)

TARGET_GUIDANCE_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

def run_guidance_pipeline(cfg: dict, dataset: dict, log=print) -> dict:
    """The whole of step 7. Training chains are divided into fit chains and probe chains; the dev chains are
    used only for the final scores.

    1. Generate with the base model on dev requests, train P_good on the fit chains, then generate with
       P_good on dev requests and on probe requests.
    2. Execute everything. Execution decides what is wrong.
    3. Ask the grader what kind of wrong each failed probe call is. It is never asked whether a call is
       wrong: execution has already answered that.
    4. Train P_bad on the probe outputs that failed plus synthetic breaks of the failure types the grader
       found most often in them.
    5. Decode dev requests guided by P_good against P_bad and execute the results."""
    import mlx.core as mx
    out_dir = type(TARGET_DIR)(cfg.get("out_dir") or TARGET_DIR)  # the smoke test writes somewhere else
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rng = random.Random(cfg["seed"])
    train, dev = dataset["target"]["train"], dataset["target"]["dev"]
    chains = sorted({r["chain_id"] for r in train})
    rng.shuffle(chains)
    probe_chains = set(chains[: max(1, round(len(chains) * cfg["probe_share"]))])
    fit_rows = [r for r in train if r["chain_id"] not in probe_chains][: cfg["n_good"]]
    probe_rows = spread([r for r in train if r["chain_id"] in probe_chains])[: cfg["n_probe"]]
    eval_rows = spread(dev)[: cfg["n_eval"]]
    loss_rows = eval_rows[: cfg["n_loss"]]

    # 1. base outputs, P_good, P_good outputs
    model, tokenizer, lm, layers, n_params = _load(cfg, cfg["seed"])
    try:
        loss_before = _mean_loss(lm, tokenizer, loss_rows)  # LoRA starts as the identity, so this is the base model
        log(f"[target] base model: generating {len(eval_rows)} dev calls")
        base_out = [_generate(lm, tokenizer, r, cfg["max_new"]) for r in eval_rows]
        snapshots = {}
        history = _fit(lm, tokenizer, fit_rows, cfg, "P_good", log, snapshots)
        loss_good = _mean_loss(lm, tokenizer, loss_rows)
        log(f"[target] P_good: generating {len(eval_rows)} dev and {len(probe_rows)} probe calls")
        good_out = [_generate(lm, tokenizer, r, cfg["max_new"]) for r in eval_rows]
        probe_out = [_generate(lm, tokenizer, r, cfg["max_new"]) for r in probe_rows]
        mx.save_safetensors(str(out_dir / "adapter_good.safetensors"), dict(tree_flatten(lm.trainable_parameters())))
        if snapshots.get("weak"):  # kept so the 3D view can show autoguidance token by token
            mx.save_safetensors(str(out_dir / "adapter_weak.safetensors"), snapshots["weak"])
    finally:
        del model, lm, layers
        gc.collect()
        mx.clear_cache()

    # 2. execution
    base_scores = [score_call(t, r) for t, r in zip(base_out, eval_rows)]
    good_scores = [score_call(t, r) for t, r in zip(good_out, eval_rows)]
    probe_scores = [score_call(t, r) for t, r in zip(probe_out, probe_rows)]

    # 3. execution has said which probe calls are wrong; the grader says what kind of wrong each one is
    wrong = [(r, t) for r, t, sc in zip(probe_rows, probe_out, probe_scores) if not sc["correct"]]
    failure_types, weak_types = None, []
    if wrong and (GRADER_DIR / "adapter.safetensors").exists():
        log(f"[target] {len(wrong)} of {len(probe_rows)} probe calls are wrong by execution; asking the grader what kind")
        grades = grade_calls([{"prompt": r["speech"], "completion": t.strip()} for r, t in wrong], cfg["grader_model_path"])
        failure_types = Counter(g["type_if_broken"] for g in grades)
        weak_types = [name for name, _ in failure_types.most_common(3)]
    elif wrong:
        log("[target] no trained grader found: P_bad will use every failure type")

    # 4. P_bad data: the target's own failures, topped up with synthetic breaks of its weak types
    own = [
        {**r, "completion": t.strip(), "source": "target output"}
        for r, t, s in zip(probe_rows, probe_out, probe_scores)
        if not s["correct"] and t.strip().endswith("</tool_call>")
    ]
    pool = [g for g in dataset["grader"]["train"] if g["is_broken"] and (not weak_types or g["failure_type"] in weak_types)]
    rng.shuffle(pool)
    setting = {r["chain_id"]: r for r in train + dev}  # a chain's files, context and references
    synthetic = [
        {**setting[g["chain_id"]], "id": g["id"], "speech": g["prompt"], "completion": g["completion"], "source": g["failure_type"]}
        for g in pool[: max(0, cfg["n_bad"] - len(own))]
    ]
    bad_rows = own + synthetic
    held_out_bad = [
        {**setting[g["chain_id"]], "id": g["id"], "speech": g["prompt"], "completion": g["completion"]}
        for g in dataset["grader"]["dev"]
        if g["is_broken"] and (not weak_types or g["failure_type"] in weak_types)
    ][: cfg["n_loss"]]

    # 5. P_bad, then guided decoding
    model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"] + 64)
    try:
        bad_before = _mean_loss(lm, tokenizer, held_out_bad)
        history += _fit(lm, tokenizer, bad_rows, cfg, "P_bad", log)
        bad_after = _mean_loss(lm, tokenizer, held_out_bad)
        bad_on_good = _mean_loss(lm, tokenizer, loss_rows)
        bad_weights = dict(tree_flatten(lm.trainable_parameters()))
        mx.save_safetensors(str(out_dir / "adapter_bad.safetensors"), bad_weights)
        good_weights = mx.load(str(out_dir / "adapter_good.safetensors"))
        # three things P_good can be steered away from: the adapter trained on mistakes, its own weaker
        # self (autoguidance), and the base model
        against = {
            "P_bad": bad_weights,
            "weak P_good": snapshots.get("weak"),
            "base": {k: (mx.zeros_like(v) if k.endswith("lora_b") else v) for k, v in good_weights.items()},
        }
        guided = {}
        for other, weight, floor in cfg["guidance"]:  # decoding only, no retraining
            if against.get(other) is None:
                continue
            name = f"vs {other} w={weight} floor={floor}"
            log(f"[target] guided decoding {name}: {len(eval_rows)} dev calls")
            outs = [_generate(lm, tokenizer, r, cfg["max_new"], good_weights, against[other], weight, floor) for r in eval_rows]
            guided[name] = {"outputs": outs, "scores": [score_call(t, r) for t, r in zip(outs, eval_rows)]}
        # the same comparison without sampling: how probable the reference call is under each guided
        # distribution, log p = log p_good + w · (log p_good − log p_other), renormalised over the vocabulary
        logp_grid = {other: {} for other, weights in against.items() if weights is not None}

        def reference_logits(weights, row):
            lm.update(tree_unflatten(list(weights.items())))
            inputs, expected, start = _encode(tokenizer, row)
            hidden = lm.model(mx.array([inputs]))[:, start:]
            logits = lm.model.embed_tokens.as_linear(hidden) if lm.config.tie_word_embeddings else lm.lm_head(hidden)
            return nn.log_softmax(logits.astype(mx.float32), axis=-1)[0], mx.array(expected)

        totals = {other: {w: [] for w in TARGET_GUIDANCE_GRID} for other in logp_grid}
        for row in loss_rows:
            good_lp, expected = reference_logits(good_weights, row)
            for other in logp_grid:
                other_lp, _ = reference_logits(against[other], row)
                for w in TARGET_GUIDANCE_GRID:
                    mixed = nn.log_softmax(good_lp + w * (good_lp - other_lp), axis=-1)
                    totals[other][w].append(float(mx.take_along_axis(mixed, expected[:, None], axis=-1).mean().item()))
        for other in logp_grid:
            logp_grid[other] = {str(w): sum(v) / max(1, len(v)) for w, v in totals[other].items()}
    finally:
        del model, lm, layers
        gc.collect()
        mx.clear_cache()

    arms = {"base": base_scores, "P_good": good_scores, **{f"guided {name}": g["scores"] for name, g in guided.items()}}
    report = {
        "created": time.time(),
        "dataset": dataset["fingerprint"],
        "config": cfg,
        "trainable_parameters": n_params,
        "rows": {"fit": len(fit_rows), "probe": len(probe_rows), "eval": len(eval_rows), "bad_own": len(own), "bad_synthetic": len(synthetic)},
        "history": history,
        "loss": {
            "good_dev_base": loss_before, "good_dev_P_good": loss_good, "good_dev_P_bad": bad_on_good,
            "bad_dev_base": bad_before, "bad_dev_P_bad": bad_after,
        },
        "execution": {
            name: {"well_formed": _rate(scores, "well_formed"), "runs": _rate(scores, "runs"), "correct": _rate(scores)}
            for name, scores in arms.items()
        },
        "reference_logp": logp_grid,
        "probe": {
            "correct": _rate(probe_scores),
            "wrong": len(wrong),
            "weak_types": weak_types,
            "failure_types": dict(failure_types) if failure_types else None,
        },
        "samples": [
            {
                "request": r["speech"],
                "reference": parse_tool_call(r["completion"]),
                "base": base_out[i].strip(), "base_correct": base_scores[i]["correct"],
                "P_good": good_out[i].strip(), "P_good_correct": good_scores[i]["correct"],
                **{f"guided {name}": g["outputs"][i].strip() for name, g in guided.items()},
                **{f"guided {name} correct": g["scores"][i]["correct"] for name, g in guided.items()},
            }
            for i, r in enumerate(eval_rows)
        ],
        "seconds": round(time.monotonic() - started, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / 1024**3, 2),
    }
    (out_dir / "guidance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "prompt_format.json").write_text(
        json.dumps({"system": TARGET_SYSTEM, "tools": TARGET_TOOLS, "assistant_prefix": "<tool_call>\n", "stop": ["</tool_call>", "<|im_end|>"]}, indent=2),
        encoding="utf-8",
    )
    log(f"[target] executed-correct on dev: " + ", ".join(f"{k} {v['correct']:.0%}" for k, v in report["execution"].items()))
    return report

def crossfit_wrong_calls(dataset: dict, folds: int = 5, log=print) -> dict:
    """P_good's mistakes on training requests it was not fitted on. The training chains are split into
    `folds` parts; for each part an adapter with the final settings is trained on the other parts and
    writes the calls for this one, which are executed. Every wrong call is kept with its request, in
    `runs/target/crossfit.json`. A part that is already there is not repeated."""
    import mlx.core as mx

    cfg = json.loads((TARGET_DIR / "report.json").read_text(encoding="utf-8"))["config"]
    path = TARGET_DIR / "crossfit.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"dataset": dataset["fingerprint"], "folds": folds, "parts": {}}
    if saved.get("dataset") != dataset["fingerprint"] or saved.get("folds") != folds:
        saved = {"dataset": dataset["fingerprint"], "folds": folds, "parts": {}}
    train = dataset["target"]["train"]
    chains = sorted({r["chain_id"] for r in train})
    random.Random(cfg["seed"] + 5).shuffle(chains)
    for k in range(folds):
        if str(k) in saved["parts"]:
            continue
        held = set(chains[k::folds])
        fit_rows, out_rows = [r for r in train if r["chain_id"] not in held], [r for r in train if r["chain_id"] in held]
        model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"] + k)
        try:
            _fit(lm, tokenizer, fit_rows, cfg, f"cross-fit {k + 1}/{folds}", log)
            outputs = []
            for done, row in enumerate(out_rows, 1):
                outputs.append(_generate(lm, tokenizer, row, cfg["max_new"]))
                if done % 40 == 0 or done == len(out_rows):
                    log(f"[target cross-fit {k + 1}/{folds}] wrote {done}/{len(out_rows)} calls")
        finally:
            del model, lm, layers
            gc.collect()
            mx.clear_cache()
        scores = [score_call(t, r) for t, r in zip(outputs, out_rows)]
        saved["parts"][str(k)] = {
            "rows": len(out_rows), "correct": sum(s["correct"] for s in scores),
            "wrong": [{"id": r["id"], "completion": t.strip(), "runs": s["runs"]} for r, t, s in zip(out_rows, outputs, scores) if not s["correct"]],
        }
        path.write_text(json.dumps(saved, indent=1), encoding="utf-8")
        log(f"[target cross-fit {k + 1}/{folds}] {saved['parts'][str(k)]['correct']} of {len(out_rows)} correct")
    return saved

def train_bad_adapter(rows: list[dict], name: str, log=print) -> None:
    """Fit an adapter with the final settings on calls that are known to be wrong, as `runs/target/adapter_<name>.safetensors`."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    cfg = json.loads((TARGET_DIR / "report.json").read_text(encoding="utf-8"))["config"]
    model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"] + 64)
    try:
        _fit(lm, tokenizer, rows, cfg, name, log)
        mx.save_safetensors(str(TARGET_DIR / f"adapter_{name}.safetensors"), dict(tree_flatten(lm.trainable_parameters())))
    finally:
        del model, lm, layers
        gc.collect()
        mx.clear_cache()

def score_decoders(dataset: dict, arms: list[dict], log=print) -> dict:
    """Write and execute the call for every held-out request under each decoder in `arms`, and keep which
    requests each one gets right in `runs/target/decoders.json`. An arm is {name, weight, floor} plus either
    `adapter` (steer P_good away from `runs/target/adapter_<adapter>.safetensors`) or `blank_request: true`
    (steer P_good away from itself reading the prompt with the request removed). An arm with neither is
    plain P_good. Arms that are already there are not repeated."""
    import mlx.core as mx

    cfg = json.loads((TARGET_DIR / "report.json").read_text(encoding="utf-8"))["config"]
    path = TARGET_DIR / "decoders.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if saved.get("dataset") != dataset["fingerprint"]:
        saved = {"dataset": dataset["fingerprint"], "arms": {}}
    rows = spread(dataset["target"]["dev"])
    todo = [a for a in arms if a["name"] not in saved["arms"]]
    if not todo:
        return saved
    model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"])
    try:
        good = mx.load(str(TARGET_DIR / "adapter_good.safetensors"))
        for arm in todo:
            other = mx.load(str(TARGET_DIR / f"adapter_{arm['adapter']}.safetensors")) if arm.get("adapter") else None
            outputs = []
            for done, row in enumerate(rows, 1):
                outputs.append(
                    _generate(
                        lm, tokenizer, row, cfg["max_new"], good=good, bad=other, weight=arm.get("weight", 0.0), floor=arm.get("floor", 0.1),
                        bad_row={**row, "speech": ""} if arm.get("blank_request") else None, blank_weight=arm.get("blank_weight", 0.0),
                    )
                )
                if done % 40 == 0 or done == len(rows):
                    log(f"[decoders] {arm['name']}: {done}/{len(rows)} calls")
            scores = [score_call(t, r) for t, r in zip(outputs, rows)]
            saved["arms"][arm["name"]] = {
                **{k: v for k, v in arm.items() if k != "name"},
                "well_formed": _rate(scores, "well_formed"), "runs": _rate(scores, "runs"), "correct": _rate(scores),
                "rows": [{"id": r["id"], "correct": s["correct"], "code": s["code"]} for r, s in zip(rows, scores)],
            }
            path.write_text(json.dumps(saved, indent=1), encoding="utf-8")
            log(f"[decoders] {arm['name']}: {_rate(scores):.1%} of {len(rows)} return the reference value")
    finally:
        del model, lm, layers
        gc.collect()
        mx.clear_cache()
    return saved

def score_low_bit(dataset: dict, bits, retrain: bool = False, log=print, rank: int | None = None) -> dict:
    """The target with its base weights stored at `bits` bits per weight (MLX affine quantisation, groups of
    64), scored by execution on every held-out request. With `retrain=False` the existing P_good adapter is
    used as it is; with `retrain=True` an adapter with the same settings and training rows is fitted on top
    of the low-bit weights first. Results are kept in `runs/target/low_bit.json`."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    cfg = {**json.loads((TARGET_DIR / "report.json").read_text(encoding="utf-8"))["config"], "bits": bits}
    if retrain and rank:  # a wider adapter has more room to make up for what the low-bit weights lost
        cfg.update(rank=rank, alpha=2.0 * rank)
    label = "ternary" if bits == "ternary" else f"{bits}-bit"
    path = TARGET_DIR / "low_bit.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if saved.get("dataset") != dataset["fingerprint"]:
        saved = {"dataset": dataset["fingerprint"], "runs": {}}
    name = f"{label}, " + (f"adapter retrained on it (rank {cfg['rank']})" if retrain else "existing adapter")
    if name in saved["runs"]:
        return saved
    rows = spread(dataset["target"]["dev"])
    model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"])
    try:
        if retrain:
            rng = random.Random(cfg["seed"])
            train = dataset["target"]["train"]
            chains = sorted({r["chain_id"] for r in train})
            rng.shuffle(chains)
            probe_chains = set(chains[: max(1, round(len(chains) * cfg["probe_share"]))])
            fit_rows = [r for r in train if r["chain_id"] not in probe_chains][: cfg["n_good"]]
            _fit(lm, tokenizer, fit_rows, cfg, f"P_good on {label}", log)
            mx.save_safetensors(str(TARGET_DIR / f"adapter_good_{label}_r{cfg['rank']}.safetensors"), dict(tree_flatten(lm.trainable_parameters())))
        else:
            lm.load_weights(str(TARGET_DIR / "adapter_good.safetensors"), strict=False)
        outputs = []
        for done, row in enumerate(rows, 1):
            outputs.append(_generate(lm, tokenizer, row, cfg["max_new"]))
            if done % 40 == 0 or done == len(rows):
                log(f"[low-bit] {name}: {done}/{len(rows)} calls")
    finally:
        del model, lm, layers
        gc.collect()
        mx.clear_cache()
    scores = [score_call(t, r) for t, r in zip(outputs, rows)]
    saved["runs"][name] = {
        "bits": bits, "retrained": retrain, "rank": cfg["rank"], "well_formed": _rate(scores, "well_formed"), "runs": _rate(scores, "runs"), "correct": _rate(scores),
        "rows": [{"id": r["id"], "correct": s["correct"], "code": s["code"]} for r, s in zip(rows, scores)],
    }
    path.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    log(f"[low-bit] {name}: {_rate(scores):.1%} of {len(rows)} return the reference value")
    return saved

def rebuild_weak_target(dataset: dict, log=print) -> None:
    """Write `runs/target/adapter_weak.safetensors` for a P_good that was trained before the weak snapshot was
    saved: the same rows, seed and schedule, stopped a quarter of the way through."""
    import mlx.core as mx

    cfg = json.loads((TARGET_DIR / "report.json").read_text(encoding="utf-8"))["config"]
    rng = random.Random(cfg["seed"])
    train = dataset["target"]["train"]
    chains = sorted({r["chain_id"] for r in train})
    rng.shuffle(chains)
    probe_chains = set(chains[: max(1, round(len(chains) * cfg["probe_share"]))])
    fit_rows = [r for r in train if r["chain_id"] not in probe_chains][: cfg["n_good"]]
    model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"])
    try:
        snapshots = {}
        _fit(lm, tokenizer, fit_rows, cfg, "weak P_good", log, snapshots, stop_at_weak=True)
        mx.save_safetensors(str(TARGET_DIR / "adapter_weak.safetensors"), snapshots["weak"])
    finally:
        del model, lm, layers
        gc.collect()
