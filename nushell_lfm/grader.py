"""The grader: a LoRA on D1 that answers broken or not, which failure type, and where."""

import gc
import json
import math
import numpy as np
import random
import shutil
import time

from .core import RUNS_DIR
from .breaks import FAILURE_LABELS, FAILURE_TYPES

GRADER_DIR = RUNS_DIR / "grader"
PLACEMENTS = {
    "all_linear": ("self_attn.", "feed_forward.", "conv."),
    "attention_only": ("self_attn.",),
    "mlp_only": ("feed_forward.",),
    "conv_only": ("conv.",),
}

def attach_lora(model, rank: int, alpha: float, dropout: float, placement: str):
    """Wrap the chosen linear layers of a freshly loaded model in LoRA. Returns the language model, the
    LoRA layers, and the trainable parameter count."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.tuner import trainer
    from mlx_lm.tuner.lora import LoRALinear

    model.freeze()
    lm = model.language_model
    layers = []
    for block in lm.layers:
        replacements = []
        for name, layer in block.named_modules():
            if isinstance(layer, (nn.Linear, nn.QuantizedLinear)) and name.startswith(PLACEMENTS[placement]):
                wrapped = LoRALinear.from_base(layer, r=rank, scale=alpha / rank, dropout=dropout)
                replacements.append((name, wrapped))
                layers.append(wrapped)
        if replacements:
            block.update_modules(tree_unflatten(replacements))
    block_type = type(lm.layers[0])
    if not getattr(block_type, "_checkpointed", False):  # grad_checkpoint patches the class; do it once
        trainer.grad_checkpoint(lm.layers[0])
        block_type._checkpointed = True
    return lm, layers, sum(v.size for _, v in tree_flatten(lm.trainable_parameters()))

def set_lora(layers, scale: float | None = None, training: bool | None = None):
    """Scale 0 gives exactly the base model. LoRA dropout is on only while `training` is true."""
    for layer in layers:
        if scale is not None:
            layer.scale = scale
        if training is not None:
            layer.train(training)

def learning_rate(step: int, total: int, base: float) -> float:
    """Linear warm-up over the first tenth of training, then cosine decay to a tenth of the base rate."""
    warm = max(1, round(0.1 * total))
    return base * min(1.0, step / warm) * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - 1) / max(1, total))))

def balanced(rows: list[dict], n: int, key: str = "failure_type") -> list[dict]:
    """Up to `n` rows taken round-robin across the values of `key`."""
    buckets = {}
    for row in rows:
        buckets.setdefault(row[key], []).append(row)
    out = []
    while len(out) < n and any(buckets.values()):
        for label in sorted(buckets):
            if buckets[label] and len(out) < n:
                out.append(buckets[label].pop(0))
    return out

_QUESTIONS = {
    "is_broken": {"type": "noul", "instructions": "Is the assistant's tool call wrong for the user request?"},
    "failure_type": {"type": "choice", "instructions": "Which description fits the tool call?", "criteria": FAILURE_TYPES},
    "fault_span": {"type": "text", "instructions": "Point at the part of the tool call that is wrong, if any."},
}

def _encode(engine, row: dict) -> dict:
    """One request and tool call packed with the three questions the way the D1 engine packs them, plus
    the training targets when the row is labelled."""
    from mlx_vlm.models.d1_lfm3_vl.d1 import pack_set
    from mlx_vlm.models.d1_lfm3_vl.pointer import state_positions
    from mlx_vlm.models.d1_lfm3_vl.prompt import as_question, prefix_text, question_header, question_slots, readout_table, suffix_layout

    tokenizer = engine.tokenizer
    head = f"User request: {row['prompt']}\n\nAssistant tool call:\n"
    state = head + row["completion"]
    questions = [as_question(_QUESTIONS[name]) for name in ("is_broken", "failure_type", "fault_span")]
    prefix = prefix_text(tokenizer, state, engine.bos, engine.state_style, engine.system, "")
    layouts = [suffix_layout(tokenizer, q, engine.lead) for q in questions]
    suffixes = [text for text, _ in layouts]
    full_text = prefix + suffixes[0]
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    ids, state_len, question, ends = pack_set(
        tokenizer, prefix, suffixes, question_header(tokenizer, state), [offsets for _, offsets in layouts], encoded["input_ids"]
    )
    positions = state_positions(tokenizer, state, prefix, full_text, engine.state_style, encoded=encoded)
    span = (0, 0)  # position 0 is the engine's "nothing to point at"
    if row.get("is_broken") and row.get("span_text"):
        lo = len(head) + row["completion"].find(row["span_text"])
        hi = lo + len(row["span_text"])
        hits = [i for i, off in enumerate(positions.offsets) if off is not None and off[1] > lo and off[0] < hi]
        if hits:
            span = (hits[0], hits[-1])
    return {
        "row": row,
        "ids": ids,
        "state_len": state_len,
        "question": question,
        "ends": ends,
        "positions": positions,
        "mask": np.asarray(positions.pointer_mask, dtype=bool),
        "yes_no": readout_table(tokenizer, question_slots(questions[0], tokenizer)[0]),
        "choice": readout_table(tokenizer, question_slots(questions[1], tokenizer)[0]),
        "y_broken": 0 if row.get("is_broken") else 1,  # the yes/no readout lists yes first
        "y_type": FAILURE_LABELS.index(row.get("failure_type", "valid")),
        "y_span": span,
    }

def _readouts(model, item: dict):
    """Yes/no logits, failure-type logits, and start/end pointer logits over the state, read the same way
    the engine reads them: option tokens through the tied embedding, pointers by scaled dot product."""
    import mlx.core as mx

    hidden = model.d1_hidden(item["ids"], item["state_len"], item["question"])
    embed = model.language_model.model.embed_tokens

    def options(table, slot):
        tokens, forms, _ = table
        raw = (hidden[slot : slot + 1] @ embed(mx.array(tokens)).T).astype(mx.float32)[0]
        return mx.stack([mx.max(raw[mx.array(form)]) for form in forms])

    queries = hidden[mx.array(item["ends"][2])].astype(mx.float32)
    keys = hidden[: len(item["mask"])].astype(mx.float32)
    pointer = (queries @ keys.T) / math.sqrt(queries.shape[-1])
    pointer = mx.where(mx.array(item["mask"])[None, :], pointer, mx.array(-1e9, dtype=mx.float32))
    return options(item["yes_no"], item["ends"][0][0]), options(item["choice"], item["ends"][1][0]), pointer

def _grader_loss(model, item: dict):
    import mlx.core as mx
    import mlx.nn as nn

    yes_no, choice, pointer = _readouts(model, item)
    ce = nn.losses.cross_entropy
    span = 0.5 * (ce(pointer[0:1], mx.array([item["y_span"][0]])) + ce(pointer[1:2], mx.array([item["y_span"][1]])))
    return (ce(yes_no[None], mx.array([item["y_broken"]])) + ce(choice[None], mx.array([item["y_type"]])) + 0.5 * span).sum()

def _predict(model, layers, scale: float, item: dict, weights: list[float]) -> dict:
    """Predictions for one item from the adapter, from the base model, and from each guidance weight w:
    log p_w = log p_base + w · (log p_adapter − log p_base). w = 0 is the base model, w = 1 the adapter."""
    import mlx.core as mx
    import mlx.nn as nn

    set_lora(layers, scale=scale)
    yes_no, choice, pointer = _readouts(model, item)
    set_lora(layers, scale=0.0)
    base_yes_no, base_choice, _ = _readouts(model, item)
    set_lora(layers, scale=scale)
    adapter, base = nn.log_softmax(choice), nn.log_softmax(base_choice)
    mx.eval(adapter, base, yes_no, base_yes_no, pointer)
    start, end = int(mx.argmax(pointer[0]).item()), int(mx.argmax(pointer[1]).item())
    text = None
    if 0 < start <= end:
        try:
            text = item["positions"].read(start, end)["text"]
        except ValueError:
            text = None
    broken_only = mx.where(mx.arange(len(FAILURE_LABELS)) == 0, mx.array(-1e9), adapter)  # label 0 is "valid"
    return {
        "logp": [round(float(v), 4) for v in adapter.tolist()],
        "logp_base": [round(float(v), 4) for v in base.tolist()],
        "types": {str(w): FAILURE_LABELS[int(mx.argmax(base + w * (adapter - base)).item())] for w in weights},
        "type_if_broken": FAILURE_LABELS[int(mx.argmax(broken_only).item())],
        "p_broken": float(mx.softmax(yes_no)[0].item()),
        "p_broken_base": float(mx.softmax(base_yes_no)[0].item()),
        "type_loss": float(-adapter[item["y_type"]].item()),
        "span": (start, end),
        "span_text": text,
    }

def _score(items: list[dict], predictions: list[dict], weights: list[float]) -> dict:
    rows = [item["row"] for item in items]
    share = lambda flags: sum(flags) / len(flags) if flags else None
    spans = [
        p["span"][0] > 0 and p["span"][0] <= item["y_span"][1] and p["span"][1] >= item["y_span"][0]
        for item, p in zip(items, predictions)
        if item["y_span"] != (0, 0)
    ]
    return {
        "type_accuracy": {str(w): share([p["types"][str(w)] == r["failure_type"] for r, p in zip(rows, predictions)]) for w in weights},
        "broken_accuracy": share([(p["p_broken"] > 0.5) == bool(r["is_broken"]) for r, p in zip(rows, predictions)]),
        "broken_accuracy_base": share([(p["p_broken_base"] > 0.5) == bool(r["is_broken"]) for r, p in zip(rows, predictions)]),
        "false_fail_reference": share([p["p_broken"] > 0.5 for r, p in zip(rows, predictions) if r.get("form") == "reference"]),
        "false_fail_alternative": share([p["p_broken"] > 0.5 for r, p in zip(rows, predictions) if r.get("form") == "alternative"]),
        "span_hit": share(spans),
        "type_loss": share([p["type_loss"] for p in predictions]),
    }

def grader_subset(rows: list[dict], n: int) -> list[dict]:
    """Half correct calls (reference and alternative forms), half broken ones spread evenly over the
    failure types, so that "is it broken?" cannot be answered by always saying yes."""
    valid = balanced([r for r in rows if not r["is_broken"]], n // 2, key="form")
    broken = balanced([r for r in rows if r["is_broken"]], n - len(valid))
    return valid + broken

GUIDANCE_GRID = [0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0]

def guidance_table(records: list[dict]) -> dict:
    """Failure-type accuracy and mean log-probability of the true type on the dev rows, for two ways of
    guiding the trained adapter and a grid of weights w:
      from the base model:   log p = log p_base + w · (log p_adapter − log p_base)   (w = 1 is the adapter)
      autoguidance:          log p = log p_adapter + w · (log p_adapter − log p_weak) (w = 0 is the adapter)
    where the weak model is the same adapter a quarter of the way through its training."""
    def score(combine):
        hits, logps = [], []
        for r in records:
            mixed = np.array(combine(r), dtype=np.float64)
            mixed = mixed - (np.log(np.exp(mixed - mixed.max()).sum()) + mixed.max())
            true = FAILURE_LABELS.index(r["true"])
            hits.append(int(mixed.argmax()) == true)
            logps.append(float(mixed[true]))
        return {"accuracy": sum(hits) / len(hits), "logp": sum(logps) / len(logps)}

    table = {"from_base": {}, "autoguidance": {}}
    for w in GUIDANCE_GRID:
        table["from_base"][str(w)] = score(lambda r: np.array(r["logp_base"]) + w * (np.array(r["logp"]) - np.array(r["logp_base"])))
        if records and "logp_weak" in records[0]:
            table["autoguidance"][str(w)] = score(lambda r: np.array(r["logp"]) + w * (np.array(r["logp"]) - np.array(r["logp_weak"])))
    return table

def train_grader(cfg: dict, train_rows: list[dict], dev_rows: list[dict], adapter_path, log=print) -> dict:
    """Train one LoRA configuration on a freshly loaded grader and score it on the dev rows."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from mlx_vlm import load as vlm_load

    mx.set_cache_limit(512 * 1024**2)
    mx.random.seed(cfg["seed"])
    started = time.monotonic()
    model, _processor = vlm_load(cfg["model_path"])
    try:
        train_items = [_encode(model.engine, r) for r in train_rows]
        dev_items = [_encode(model.engine, r) for r in dev_rows]
        lm, layers, n_params = attach_lora(model, cfg["rank"], cfg["alpha"], cfg["dropout"], cfg["placement"])
        scale = cfg["alpha"] / cfg["rank"]
        optimizer = optim.AdamW(learning_rate=cfg["lr"], weight_decay=0.0)
        value_and_grad = nn.value_and_grad(lm, lambda _lm, item: _grader_loss(model, item))
        total = math.ceil(len(train_items) / cfg["batch"]) * cfg["epochs"]
        order = random.Random(cfg["seed"])
        history, step = [], 0
        for epoch in range(cfg["epochs"]):
            set_lora(layers, training=True)
            order.shuffle(train_items)
            for offset in range(0, len(train_items), cfg["batch"]):
                batch = train_items[offset : offset + cfg["batch"]]
                grads, losses = None, []
                for item in batch:
                    loss, grad = value_and_grad(lm, item)
                    grads = grad if grads is None else tree_map(lambda a, b: a + b, grads, grad)
                    mx.eval(grads)
                    losses.append(float(loss.item()))
                step += 1
                optimizer.learning_rate = learning_rate(step, total, cfg["lr"])
                clipped, _ = optim.clip_grad_norm(tree_map(lambda g: g / len(batch), grads), cfg["clip"])
                optimizer.update(lm, clipped)
                mx.eval(lm.trainable_parameters(), optimizer.state)
                history.append({"config": cfg["name"], "step": step, "loss": sum(losses) / len(losses)})
                if step == 1 or step % 10 == 0 or step == total:
                    log(f"[grader {cfg['name']}] step {step}/{total} loss {history[-1]['loss']:.3f}")
                if step == max(1, total // 4):  # the weak version of this adapter, for autoguidance
                    weak = {k: mx.array(v) for k, v in tree_flatten(lm.trainable_parameters())}
                    mx.eval(weak)
        set_lora(layers, training=False)
        predictions = [_predict(model, layers, scale, item, cfg["guidance_weights"]) for item in dev_items]
        trained = dict(tree_flatten(lm.trainable_parameters()))
        mx.save_safetensors(str(adapter_path), trained)
        lm.update(tree_unflatten(list(weak.items())))
        for item, prediction in zip(dev_items, predictions):
            prediction["logp_weak"] = _predict(model, layers, scale, item, [1.0])["logp"]
        lm.update(tree_unflatten(list(trained.items())))
    finally:
        del model, _processor
        gc.collect()
        mx.clear_cache()
    metrics = _score(dev_items, predictions, cfg["guidance_weights"])
    log(f"[grader {cfg['name']}] dev type accuracy {metrics['type_accuracy']} · broken? {metrics['broken_accuracy']:.0%}")
    return {
        "config": cfg,
        "trainable_parameters": n_params,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "history": history,
        "metrics": metrics,
        "records": (records := [
            {
                "id": item["row"]["id"], "true": item["row"]["failure_type"], "form": item["row"].get("form"), **p["types"],
                "p_broken": p["p_broken"], "logp": p["logp"], "logp_base": p["logp_base"], "logp_weak": p["logp_weak"],
            }
            for item, p in zip(dev_items, predictions)
        ]),
        "guidance": guidance_table(records),
        "seconds": round(time.monotonic() - started, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / 1024**3, 2),
    }

def run_grader_sweep(configs: list[dict], dataset: dict, n_train: int, n_dev: int, log=print, keep: bool = True) -> dict:
    """Train each configuration from a fresh model, keep the adapter with the lowest dev failure-type
    loss, and write `runs/grader/{adapter.safetensors, adapter_config.json, report.json}`. With
    `keep=False` nothing on disk is replaced: the runs are only measured."""
    GRADER_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = grader_subset(dataset["grader"]["train"], n_train)
    dev_rows = grader_subset(dataset["grader"]["dev"], n_dev)
    runs = []
    for cfg in configs:
        runs.append(train_grader(cfg, train_rows, dev_rows, GRADER_DIR / f"candidate-{cfg['name']}.safetensors", log))
    best = min(runs, key=lambda r: r["metrics"]["type_loss"])
    if keep:
        shutil.copyfile(GRADER_DIR / f"candidate-{best['config']['name']}.safetensors", GRADER_DIR / "adapter.safetensors")
        (GRADER_DIR / "adapter_config.json").write_text(json.dumps(best["config"], indent=2), encoding="utf-8")
    for cfg in configs:
        (GRADER_DIR / f"candidate-{cfg['name']}.safetensors").unlink()
    report = {"created": time.time(), "dataset": dataset["fingerprint"], "best": best["config"]["name"], "runs": runs}
    if keep:
        (GRADER_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def run_grader_experiments(model_path: str, dataset: dict, sizes: dict, log=print) -> dict:
    """The grader's full protocol, each stage choosing by dev failure-type loss on held-out environments:
    1. a sweep over learning rate and rank with the adapter on every linear layer;
    2. an ablation of where the adapter goes (attention, MLP or convolution layers only) at the best setting;
    3. the final adapter: the best setting and placement, trained on the full budget.
    Guidance is scored on the final adapter (see `guidance_table`). Everything is written to
    `runs/grader/experiments.json`."""
    base = {"model_path": model_path, "placement": "all_linear", "dropout": 0.05, "clip": 1.0, "batch": 4, "epochs": 1, "seed": 73, "guidance_weights": [0.0, 1.0, 1.5]}
    slim = lambda report: [{"config": r["config"], "metrics": r["metrics"], "seconds": r["seconds"], "trainable_parameters": r["trainable_parameters"], "train_rows": r["train_rows"]} for r in report["runs"]]
    sweep = run_grader_sweep(
        [{**base, "name": f"all_linear-r{rank}-lr{lr:g}", "rank": rank, "alpha": 2.0 * rank, "lr": lr} for rank in sizes["ranks"] for lr in sizes["learning_rates"]],
        dataset, sizes["n_sweep"], sizes["n_dev"], log, keep=False,
    )
    best = next(r["config"] for r in sweep["runs"] if r["config"]["name"] == sweep["best"])
    ablation = run_grader_sweep(
        [{**best, "name": f"{placement}-r{best['rank']}-lr{best['lr']:g}", "placement": placement} for placement in ("attention_only", "mlp_only", "conv_only")],
        dataset, sizes["n_sweep"], sizes["n_dev"], log, keep=False,
    )
    candidates = [r for r in sweep["runs"] if r["config"]["name"] == sweep["best"]] + ablation["runs"]
    chosen = min(candidates, key=lambda r: r["metrics"]["type_loss"])["config"]
    final = run_grader_sweep([{**chosen, "name": "final-" + chosen["name"], "epochs": sizes["epochs"]}], dataset, sizes["n_full"], sizes.get("n_dev_final", sizes["n_dev"]), log)
    experiments = {
        "created": time.time(), "dataset": dataset["fingerprint"], "sizes": sizes,
        "sweep": slim(sweep), "sweep_best": sweep["best"], "placement_ablation": slim(ablation), "chosen": chosen["name"],
        "final": slim(final)[0], "guidance": final["runs"][0]["guidance"],
    }
    (GRADER_DIR / "experiments.json").write_text(json.dumps(experiments, indent=1), encoding="utf-8")
    return experiments

def grade_calls(rows: list[dict], model_path: str | None = None) -> list[dict]:
    """Grade `{prompt, completion}` rows with the saved grader adapter: P(broken), failure type, the most
    likely failure type given that the call is known to be broken, and the span."""
    import mlx.core as mx
    from mlx_vlm import load as vlm_load

    cfg = json.loads((GRADER_DIR / "adapter_config.json").read_text(encoding="utf-8"))
    mx.set_cache_limit(512 * 1024**2)
    model, _processor = vlm_load(model_path or cfg["model_path"])
    try:
        lm, layers, _ = attach_lora(model, cfg["rank"], cfg["alpha"], 0.0, cfg["placement"])
        lm.load_weights(str(GRADER_DIR / "adapter.safetensors"), strict=False)
        set_lora(layers, training=False)
        out = []
        for row in rows:
            p = _predict(model, layers, cfg["alpha"] / cfg["rank"], _encode(model.engine, row), [1.0])
            out.append({"p_broken": p["p_broken"], "failure_type": p["types"]["1.0"], "type_if_broken": p["type_if_broken"], "span_text": p["span_text"]})
    finally:
        del model, _processor
        gc.collect()
        mx.clear_cache()
    return out

encode_grader_row, grader_readouts = _encode, _readouts  # also used by the activation view
