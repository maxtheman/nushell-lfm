"""The target: a LoRA on LFM3 that writes the call, guided decoding, and the experiments around it."""

from collections import Counter
import gc
import json
import math
import random
import time

from .core import RUNS_DIR
from .breaks import parse_tool_call
from .environments import run_nu, same_output
from .grader import GRADER_DIR, attach_lora, grade_calls, learning_rate, set_lora

TARGET_DIR = RUNS_DIR / "target"
TARGET_SYSTEM = (
    "Make one run_nushell tool call for the request. Preserve every requested step in order. "
    "Discover files with glob, ls and open instead of assuming their contents, and bind intermediate results with let. "
    "When a path or key may be missing, recover with try, default or is-empty and return the choices. "
    "Answer a simple question with a simple command. When the request names fields or asks for structured "
    "output, return JSON with | to json --raw. Code contains only Nushell source."
)
TARGET_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_nushell",
            "description": "Execute Nushell (Nu) source in the workspace root.",
            "parameters": {"type": "object", "required": ["code"], "properties": {"code": {"type": "string"}}},
        },
    }
]

def target_prompt(tokenizer, row: dict) -> str:
    """The chat prompt up to and including the opening `<tool_call>` line; the model writes the rest."""
    refs = "\n\n".join(f"{doc['title']} (nushell)\n{doc['text']}" for doc in row.get("references", []))
    text = f"\nSyntax references:\n{refs}\n\n" if refs else "\n"
    text += f"Request: {row['speech']}\nWorkspace: {row['context']}\nFiles: " + ", ".join(row["workspace_files"])
    messages = [{"role": "system", "content": TARGET_SYSTEM}, {"role": "user", "content": [{"type": "text", "text": text}]}]
    # Thinking is switched off so the assistant turn starts directly with the tool call.
    template = tokenizer.apply_chat_template(messages, tools=TARGET_TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return template + "<tool_call>\n"

def _encode(tokenizer, row: dict):
    """Input tokens, the completion tokens the loss is taken on, and where they start."""
    prompt = target_prompt(tokenizer, row)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = tokenizer.encode(prompt + row["completion"] + "<|im_end|>", add_special_tokens=False)
    start = len(prefix)
    if tokens[:start] != prefix:  # the prompt's final newline can merge with the first completion token
        start -= 1
    if not 0 < start < len(tokens) or tokens[:start] != prefix[:start]:
        raise ValueError(f"completion is misaligned for {row['id']}")
    return tokens[:-1], tokens[start:], start - 1

def _completion_loss(lm, inputs, expected, start):
    import mlx.core as mx
    import mlx.nn as nn

    backbone = lm.model
    hidden = backbone(mx.array([inputs]))[:, start:]
    logits = backbone.embed_tokens.as_linear(hidden) if lm.config.tie_word_embeddings else lm.lm_head(hidden)
    return nn.losses.cross_entropy(logits.astype(mx.float32), mx.array([expected]), reduction="mean")

def _mean_loss(lm, tokenizer, rows) -> float | None:
    return sum(float(_completion_loss(lm, *_encode(tokenizer, r)).item()) for r in rows) / len(rows) if rows else None

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
    import mlx.core as mx
    from mlx_vlm import load as vlm_load

    mx.set_cache_limit(512 * 1024**2)
    mx.random.seed(seed)
    model, processor = vlm_load(cfg["model_path"])
    if cfg.get("bits"):  # the same weights stored with fewer bits: back to full precision, then quantised again
        import mlx.nn as nn
        from mlx_lm.utils import dequantize_model

        dequantize_model(model.language_model)
        if cfg["bits"] == "ternary":
            _ternarize(model.language_model, cfg.get("group_size", 64))
        else:
            nn.quantize(model.language_model, group_size=cfg.get("group_size", 64), bits=cfg["bits"])
        mx.eval(model.language_model.parameters())
    lm, layers, n_params = attach_lora(model, cfg["rank"], cfg["alpha"], 0.0, cfg.get("placement", "all_linear"))
    return model, getattr(processor, "tokenizer", processor), lm, layers, n_params

def _fit(lm, tokenizer, rows: list[dict], cfg: dict, name: str, log, snapshots: dict | None = None, stop_at_weak: bool = False) -> list[dict]:
    """One LoRA on `rows`: per-example gradients averaged over a batch, clipped, AdamW with warm-up and cosine
    decay. With `snapshots`, the adapter as it stood a quarter of the way through is kept under "weak"."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map

    optimizer = optim.AdamW(learning_rate=cfg["lr"], weight_decay=0.0)
    value_and_grad = nn.value_and_grad(lm, _completion_loss)
    total = math.ceil(len(rows) / cfg["batch"]) * cfg["epochs"]
    order, rows, history, step = random.Random(cfg["seed"]), list(rows), [], 0
    for _epoch in range(cfg["epochs"]):
        order.shuffle(rows)
        for offset in range(0, len(rows), cfg["batch"]):
            batch = rows[offset : offset + cfg["batch"]]
            grads, losses = None, []
            for row in batch:
                loss, grad = value_and_grad(lm, *_encode(tokenizer, row))
                grads = grad if grads is None else tree_map(lambda a, b: a + b, grads, grad)
                mx.eval(grads)
                losses.append(float(loss.item()))
            step += 1
            optimizer.learning_rate = learning_rate(step, total, cfg["lr"])
            clipped, _ = optim.clip_grad_norm(tree_map(lambda g: g / len(batch), grads), 1.0)
            optimizer.update(lm, clipped)
            mx.eval(lm.trainable_parameters(), optimizer.state)
            history.append({"adapter": name, "step": step, "loss": sum(losses) / len(losses)})
            if step == 1 or step % 10 == 0 or step == total:
                log(f"[target {name}] step {step}/{total} loss {history[-1]['loss']:.3f}")
            if snapshots is not None and step == max(1, total // 4):  # the weak version of this adapter
                snapshots["weak"] = {k: mx.array(v) for k, v in tree_flatten(lm.trainable_parameters())}
                mx.eval(snapshots["weak"])
                if stop_at_weak:
                    return history
    return history

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

def score_call(text: str, row: dict) -> dict:
    """Execute a generated tool call: is it a well-formed call, does it run, does it return the reference value?"""
    code = parse_tool_call(text.strip())
    if code is None:
        return {"well_formed": False, "runs": False, "correct": False, "code": None}
    result = run_nu(code, row["env"])
    return {"well_formed": True, "runs": result["ok"], "correct": result["ok"] and same_output(result["stdout"], row["expected"]), "code": code}

def _rate(scores: list[dict], key: str = "correct"):
    return sum(s[key] for s in scores) / len(scores) if scores else None

TARGET_GUIDANCE_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

def quick_target_run(cfg: dict, dataset: dict, log=print) -> dict:
    """Train one P_good configuration and measure it: loss on held-out reference calls, and the share of
    held-out requests whose generated call returns the reference value. Used for sweeps and ablations;
    nothing is saved."""
    import mlx.core as mx

    started = time.monotonic()
    train, dev = dataset["target"]["train"], dataset["target"]["dev"]
    spread = lambda rows: sorted(rows, key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))
    fit_rows, eval_rows = train[: cfg["n_good"]], spread(dev)[: cfg["n_eval"]]
    model, tokenizer, lm, layers, n_params = _load(cfg, cfg["seed"])
    try:
        history = _fit(lm, tokenizer, fit_rows, cfg, cfg["name"], log)
        loss = _mean_loss(lm, tokenizer, eval_rows[: cfg["n_loss"]])
        scores = [score_call(_generate(lm, tokenizer, r, cfg["max_new"]), r) for r in eval_rows]
    finally:
        del model, lm, layers
        gc.collect()
        mx.clear_cache()
    result = {
        "config": {k: v for k, v in cfg.items() if k not in ("guidance",)}, "trainable_parameters": n_params, "fit_rows": len(fit_rows),
        "dev_loss": loss, "well_formed": _rate(scores, "well_formed"), "runs": _rate(scores, "runs"), "correct": _rate(scores),
        "final_train_loss": history[-1]["loss"], "seconds": round(time.monotonic() - started, 1),
    }
    log(f"[target {cfg['name']}] dev loss {loss:.3f} · executed-correct {result['correct']:.0%} of {len(eval_rows)}")
    return result

def run_target_experiments(base: dict, dataset: dict, sizes: dict, log=print) -> dict:
    """The target's full protocol, each stage choosing by executed-correct on held-out environments (dev loss
    breaks ties):
    1. a sweep over learning rate and rank with the adapter on every linear layer;
    2. an ablation of where the adapter goes (attention, MLP or convolution layers only) at the best setting;
    3. the full pipeline at the chosen setting: P_good on the full budget, the probe, P_bad, and the
       guidance comparison (against P_bad, against a weaker P_good, against the base model).
    Everything is written to `runs/target/experiments.json`."""
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    quick = {**base, "n_good": sizes["n_sweep"], "n_eval": sizes["n_sweep_eval"], "n_loss": sizes["n_sweep_eval"], "epochs": 1, "placement": "all_linear"}
    better = lambda r: (r["correct"], -r["dev_loss"])
    sweep = [
        quick_target_run({**quick, "name": f"all_linear-r{rank}-lr{lr:g}", "rank": rank, "alpha": 2.0 * rank, "lr": lr}, dataset, log)
        for rank in sizes["ranks"] for lr in sizes["learning_rates"]
    ]
    best = max(sweep, key=better)["config"]
    ablation = [
        quick_target_run({**best, "name": f"{placement}-r{best['rank']}-lr{best['lr']:g}", "placement": placement}, dataset, log)
        for placement in ("attention_only", "mlp_only", "conv_only")
    ]
    chosen = max([r for r in sweep if r["config"]["name"] == best["name"]] + ablation, key=better)["config"]
    final = run_target_pipeline(
        {**base, **sizes["full"], "rank": chosen["rank"], "alpha": chosen["alpha"], "lr": chosen["lr"], "placement": chosen["placement"]}, dataset, log
    )
    experiments = {
        "created": time.time(), "dataset": dataset["fingerprint"], "sizes": sizes, "sweep": sweep, "sweep_best": best["name"],
        "placement_ablation": ablation, "chosen": chosen["name"],
        "final": {k: final[k] for k in ("config", "rows", "loss", "execution", "reference_logp", "probe", "seconds", "trainable_parameters")},
    }
    (TARGET_DIR / "experiments.json").write_text(json.dumps(experiments, indent=1), encoding="utf-8")
    return experiments

def run_target_pipeline(cfg: dict, dataset: dict, log=print) -> dict:
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
    spread = lambda rows: sorted(rows, key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))  # one request per chain first
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
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
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
    rows = sorted(dataset["target"]["dev"], key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))
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
    rows = sorted(dataset["target"]["dev"], key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))
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

def score_saved_target(dataset: dict, log=print) -> dict:
    """Score the saved P_good on every held-out request that took no part in choosing its settings or in the
    report's own scores, and add the result to `runs/target/report.json` as `unused_held_out`."""
    report = json.loads((TARGET_DIR / "report.json").read_text(encoding="utf-8"))
    cfg = report["config"]
    spread = lambda rows: sorted(rows, key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))
    rows = spread(dataset["target"]["dev"])[cfg["n_eval"]:]
    model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"])
    try:
        lm.load_weights(str(TARGET_DIR / "adapter_good.safetensors"), strict=False)
        outputs = []
        for done, row in enumerate(rows, 1):
            outputs.append(_generate(lm, tokenizer, row, cfg["max_new"]))
            if done % 20 == 0 or done == len(rows):
                log(f"[target] unused held-out requests: {done}/{len(rows)}")
    finally:
        del model, lm, layers
        gc.collect()
    scores = [score_call(t, r) for t, r in zip(outputs, rows)]
    by_env = {}
    for row, sc in zip(rows, scores):
        by_env.setdefault(row["env"], []).append(sc["correct"])
    report["unused_held_out"] = {
        "rows": len(rows), "well_formed": _rate(scores, "well_formed"), "runs": _rate(scores, "runs"), "correct": _rate(scores),
        "by_environment": {env: {"rows": len(v), "correct": sum(v) / len(v)} for env, v in sorted(by_env.items())},
        "wrong": [{"request": r["speech"], "reference": parse_tool_call(r["completion"]), "got": sc["code"]} for r, sc in zip(rows, scores) if not sc["correct"]],
    }
    (TARGET_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report["unused_held_out"]
