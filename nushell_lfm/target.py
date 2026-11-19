"""The target: a LoRA on LFM3 1.7B that writes one run_nushell call for a request, scored by executing the call."""

import gc
import json
import math
import random
import time

from .calls import parse_tool_call
from .core import MODELS_DIR, RUNS_DIR, model_path
from .environments import run_nu, same_output
from .lora import attach_lora, learning_rate

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

def _load(cfg: dict, seed: int, prepare=None):
    """The checkpoint with a fresh LoRA attached. `prepare(model)` runs before the LoRA goes on (the low-bit
    experiments use it to requantise the base weights)."""
    import mlx.core as mx
    from mlx_vlm import load as vlm_load

    mx.set_cache_limit(512 * 1024**2)
    mx.random.seed(seed)
    model, processor = vlm_load(model_path(cfg["model_path"]))  # a folder under $MODELS_DIR, or a path
    if prepare:
        prepare(model)
    lm, layers, n_params = attach_lora(model, cfg["rank"], cfg["alpha"], 0.0, cfg.get("placement", "all_linear"))
    return model, getattr(processor, "tokenizer", processor), lm, layers, n_params

def _fit(lm, tokenizer, rows: list[dict], cfg: dict, name: str, log, snapshots: dict | None = None, stop_at_weak: bool = False) -> list[dict]:
    """One LoRA on `rows`: per-example gradients averaged over a batch, clipped, AdamW with warm-up and cosine
    decay. With `snapshots`, the adapter as it stood a quarter of the way through is kept under "weak"
    (the guidance experiments steer away from it)."""
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

def generate(lm, tokenizer, row: dict, max_new: int) -> str:
    """Greedy decoding of one tool call, stopping at the end of the call."""
    import mlx.core as mx

    stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    tokens = mx.array([tokenizer.encode(target_prompt(tokenizer, row), add_special_tokens=False)])
    cache, written = lm.make_cache(), []
    for _ in range(max_new):
        out = lm(tokens, cache=cache)
        token = int(mx.argmax((out.logits if hasattr(out, "logits") else out)[:, -1, :], axis=-1).item())
        if token in stop:
            break
        written.append(token)
        if tokenizer.decode(written).rstrip().endswith("</tool_call>"):
            break
        tokens = mx.array([[token]])
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

def _portable(cfg: dict) -> dict:
    """The settings as they are saved: a checkpoint under $MODELS_DIR by its folder name, and no output folder."""
    out = {k: v for k, v in cfg.items() if k != "out_dir"}
    path = type(MODELS_DIR)(str(out["model_path"]))
    if path.is_absolute() and path.parent == MODELS_DIR:
        out["model_path"] = path.name
    return out

def spread(rows: list[dict]) -> list[dict]:
    """Held-out requests in a fixed order that takes one request per chain first."""
    return sorted(rows, key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))

def fit_rows_of(train: list[dict], cfg: dict) -> list[dict]:
    """The training rows the adapter is fitted on. `probe_share` of the training chains are kept out: the
    saved adapter was trained that way (the guidance experiments read its mistakes on them), so the same
    split reproduces it. Set it to 0 to fit on everything."""
    chains = sorted({r["chain_id"] for r in train})
    random.Random(cfg["seed"]).shuffle(chains)
    kept_out = set(chains[: max(1, round(len(chains) * cfg["probe_share"]))]) if cfg.get("probe_share") else set()
    return [r for r in train if r["chain_id"] not in kept_out][: cfg["n_good"]]

def _summary(rows: list[dict], scores: list[dict]) -> dict:
    by_env = {}
    for row, sc in zip(rows, scores):
        by_env.setdefault(row["env"], []).append(sc["correct"])
    return {
        "rows": len(rows), "well_formed": _rate(scores, "well_formed"), "runs": _rate(scores, "runs"), "correct": _rate(scores),
        "by_environment": {env: {"rows": len(v), "correct": sum(v) / len(v)} for env, v in sorted(by_env.items())},
        "wrong": [{"request": r["speech"], "reference": parse_tool_call(r["completion"]), "got": sc["code"]} for r, sc in zip(rows, scores) if not sc["correct"]],
    }

def quick_target_run(cfg: dict, dataset: dict, log=print) -> dict:
    """Train one P_good configuration and measure it: loss on held-out reference calls, and the share of
    held-out requests whose generated call returns the reference value. Used for sweeps and ablations;
    nothing is saved."""
    import mlx.core as mx

    started = time.monotonic()
    train, dev = dataset["target"]["train"], dataset["target"]["dev"]
    fit_rows, eval_rows = train[: cfg["n_good"]], spread(dev)[: cfg["n_eval"]]
    model, tokenizer, lm, layers, n_params = _load(cfg, cfg["seed"])
    try:
        history = _fit(lm, tokenizer, fit_rows, cfg, cfg["name"], log)
        loss = _mean_loss(lm, tokenizer, eval_rows[: cfg["n_loss"]])
        scores = [score_call(generate(lm, tokenizer, r, cfg["max_new"]), r) for r in eval_rows]
    finally:
        del model, lm, layers
        gc.collect()
        mx.clear_cache()
    result = {
        "config": _portable(cfg), "trainable_parameters": n_params, "fit_rows": len(fit_rows),
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
    3. the final adapter at the chosen setting, on the full training budget, scored on every held-out request.
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
        "final": {k: final[k] for k in ("config", "rows", "loss", "execution", "unused_held_out", "seconds", "trainable_parameters")},
    }
    (TARGET_DIR / "experiments.json").write_text(json.dumps(experiments, indent=1), encoding="utf-8")
    return experiments

def run_target_pipeline(cfg: dict, dataset: dict, log=print) -> dict:
    """Train the adapter and score it by execution on held-out environments.

    1. The base model writes the calls for the first `n_eval` held-out requests.
    2. The adapter is fitted on the training rows and saved.
    3. The adapter writes the calls for every held-out request: the first `n_eval` are compared with the
       base model, and the rest (requests used for nothing else) are reported as `unused_held_out`.
    Everything is executed; a call is correct only if it returns the reference value. The report goes to
    `runs/target/report.json` (or `cfg["out_dir"]`)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    out_dir = type(TARGET_DIR)(cfg.get("out_dir") or TARGET_DIR)  # the smoke test writes somewhere else
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    train, dev = dataset["target"]["train"], spread(dataset["target"]["dev"])
    fit_rows = fit_rows_of(train, cfg)
    eval_rows, rest_rows = dev[: cfg["n_eval"]], dev[cfg["n_eval"]:] if cfg.get("score_all", True) else []
    loss_rows = eval_rows[: cfg["n_loss"]]

    model, tokenizer, lm, layers, n_params = _load(cfg, cfg["seed"])
    try:
        loss_before = _mean_loss(lm, tokenizer, loss_rows)  # LoRA starts as the identity, so this is the base model
        log(f"[target] base model: generating {len(eval_rows)} held-out calls")
        base_out = [generate(lm, tokenizer, r, cfg["max_new"]) for r in eval_rows]
        snapshots = {}
        history = _fit(lm, tokenizer, fit_rows, cfg, "P_good", log, snapshots)
        loss_good = _mean_loss(lm, tokenizer, loss_rows)
        mx.save_safetensors(str(out_dir / "adapter_good.safetensors"), dict(tree_flatten(lm.trainable_parameters())))
        if snapshots.get("weak"):
            mx.save_safetensors(str(out_dir / "adapter_weak.safetensors"), snapshots["weak"])
        log(f"[target] with the adapter: generating {len(eval_rows) + len(rest_rows)} held-out calls")
        good_out = [generate(lm, tokenizer, r, cfg["max_new"]) for r in eval_rows + rest_rows]
    finally:
        del model, lm, layers
        gc.collect()
        mx.clear_cache()

    base_scores = [score_call(t, r) for t, r in zip(base_out, eval_rows)]
    good_scores = [score_call(t, r) for t, r in zip(good_out, eval_rows + rest_rows)]
    arms = {"base": base_scores, "P_good": good_scores[: len(eval_rows)]}
    report = {
        "created": time.time(),
        "dataset": dataset["fingerprint"],
        "config": _portable(cfg),
        "trainable_parameters": n_params,
        "rows": {"fit": len(fit_rows), "eval": len(eval_rows)},
        "history": history,
        "loss": {"good_dev_base": loss_before, "good_dev_P_good": loss_good},
        "execution": {
            name: {"well_formed": _rate(scores, "well_formed"), "runs": _rate(scores, "runs"), "correct": _rate(scores)}
            for name, scores in arms.items()
        },
        "samples": [
            {
                "request": r["speech"],
                "reference": parse_tool_call(r["completion"]),
                "base": base_out[i].strip(), "base_correct": base_scores[i]["correct"],
                "P_good": good_out[i].strip(), "P_good_correct": good_scores[i]["correct"],
            }
            for i, r in enumerate(eval_rows)
        ],
        "seconds": round(time.monotonic() - started, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / 1024**3, 2),
    }
    if rest_rows:
        report["unused_held_out"] = _summary(rest_rows, good_scores[len(eval_rows):])
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "prompt_format.json").write_text(
        json.dumps({"system": TARGET_SYSTEM, "tools": TARGET_TOOLS, "assistant_prefix": "<tool_call>\n", "stop": ["</tool_call>", "<|im_end|>"]}, indent=2),
        encoding="utf-8",
    )
    log("[target] executed-correct on held-out requests: " + ", ".join(f"{k} {v['correct']:.0%}" for k, v in report["execution"].items()))
    return report

def time_saved_target(dataset: dict, log=print) -> dict:
    """How long the saved adapter takes to write a call, over every held-out request (generation only, after one
    warm-up call; the model is already loaded). Added to `runs/target/report.json` as `timing`."""
    import mlx.core as mx

    report = json.loads((TARGET_DIR / "report.json").read_text(encoding="utf-8"))
    cfg = report["config"]
    rows = spread(dataset["target"]["dev"])
    model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"])
    try:
        lm.load_weights(str(TARGET_DIR / "adapter_good.safetensors"), strict=False)
        generate(lm, tokenizer, rows[0], cfg["max_new"])
        seconds, tokens, outputs = [], 0, []
        for done, row in enumerate(rows, 1):
            started = time.perf_counter()
            text = generate(lm, tokenizer, row, cfg["max_new"])
            seconds.append(time.perf_counter() - started)
            tokens += len(tokenizer.encode(text, add_special_tokens=False))
            outputs.append(text)
            if done % 40 == 0 or done == len(rows):
                log(f"[target] timing: {done}/{len(rows)}")
    finally:
        del model, lm, layers
        gc.collect()
        mx.clear_cache()
    ordered = sorted(seconds)
    report["timing"] = {
        "requests": len(rows), "seconds_per_request": round(sum(seconds) / len(seconds), 2), "median_seconds": round(ordered[len(ordered) // 2], 2),
        "slowest_seconds": round(ordered[-1], 2), "tokens_per_second": round(tokens / sum(seconds), 1),
        "correct": _rate([score_call(t, r) for t, r in zip(outputs, rows)]), "where": "MLX, 8-bit base + adapter, one request at a time",
    }
    (TARGET_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report["timing"]

def score_saved_target(dataset: dict, log=print) -> dict:
    """Score the saved adapter again on every held-out request that took no part in choosing its settings or
    in the report's comparison with the base model, and put the result in `runs/target/report.json` as
    `unused_held_out`."""
    import mlx.core as mx

    report = json.loads((TARGET_DIR / "report.json").read_text(encoding="utf-8"))
    cfg = report["config"]
    rows = spread(dataset["target"]["dev"])[cfg["n_eval"]:]
    model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"])
    try:
        lm.load_weights(str(TARGET_DIR / "adapter_good.safetensors"), strict=False)
        outputs = []
        for done, row in enumerate(rows, 1):
            outputs.append(generate(lm, tokenizer, row, cfg["max_new"]))
            if done % 20 == 0 or done == len(rows):
                log(f"[target] unused held-out requests: {done}/{len(rows)}")
    finally:
        del model, lm, layers
        gc.collect()
        mx.clear_cache()
    report["unused_held_out"] = _summary(rows, [score_call(t, r) for t, r in zip(outputs, rows)])
    (TARGET_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report["unused_held_out"]
