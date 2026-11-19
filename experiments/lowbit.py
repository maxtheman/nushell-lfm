"""Squeeze the target's base weights down to 2 bits or 1 bit per weight, and train it to keep the skill.

The teacher is the 8-bit base model with the P_good adapter. The student is the same network with every linear
layer of every block replaced, in four steps:

1. Hadamard fold. Each weight matrix W is replaced by W·H, with H a block Hadamard rotation of the layer's
   input, and the layer applies H to its input first. W·H·H·x = W·x, so nothing changes until the weights are
   rounded; the rotation spreads large weights out so that rounding hurts less.
2. Quantise the rotated weights (MLX affine quantisation: every group of 128 weights keeps a scale and an offset).
3. Retrain every weight, a few blocks at a time. Each block gets the teacher's input to that block and is
   trained to give the teacher's output, with the weights rounded in the forward pass and the gradient passed
   straight through the rounding. Only the blocks being trained are held at full precision, which is why this
   fits in memory.
4. Distil end to end. A LoRA and the scales and offsets of the quantised layers are trained so that the
   student's next-token distribution matches the teacher's on the training requests.

The result is scored the same way as everything else: its calls are executed on the held-out requests.

    .venv/bin/python experiments/lowbit.py --bits 2          # about two hours alone on a 24 GB M4 Pro
    .venv/bin/python experiments/lowbit.py --bits 1
    .venv/bin/python experiments/lowbit.py --bits 2 --smoke  # a few steps of every stage, to prove the code path

Inspired by the recipe of rotating, quantising and distilling against the frozen full-precision weights; scaled
down from billions of general tokens to the few hundred thousand tokens of this one task.
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
print = functools.partial(print, flush=True)


# ----------------------------------------------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------------------------------------------
def hadamard(n: int):
    """The n×n Hadamard matrix (n a power of two), scaled so that it is its own inverse."""
    import mlx.core as mx
    import numpy as np

    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return mx.array(h / math.sqrt(n))


def block_size(dim: int, largest: int = 1024) -> int:
    """The largest power of two, at most `largest`, that divides `dim`."""
    size = 1
    while size * 2 <= largest and dim % (size * 2) == 0:
        size *= 2
    return size


def rotate(x, h):
    """Apply the block rotation `h` along the last axis of `x`."""
    size = h.shape[0]
    return (x.reshape(*x.shape[:-1], x.shape[-1] // size, size) @ h).reshape(x.shape)


def rounded(w, bits: int, group: int):
    """What `w` [out, in] becomes when each group of `group` weights in a row is stored with `bits` bits:
    2**bits evenly spaced levels between the group's smallest and largest weight."""
    import mlx.core as mx

    out_dim, in_dim = w.shape
    g = w.reshape(out_dim, in_dim // group, group)
    low, high = g.min(axis=-1, keepdims=True), g.max(axis=-1, keepdims=True)
    step = mx.maximum((high - low) / (2**bits - 1), 1e-8)
    return (mx.round((g - low) / step) * step + low).reshape(out_dim, in_dim)


def make_layers():
    """The two layer classes, built on first use so that importing this file does not import MLX."""
    import mlx.core as mx
    import mlx.nn as nn

    class TrainableLowBit(nn.Module):
        """A linear layer whose full-precision weight is trained while its output is computed from the rounded
        weight (the gradient treats the rounding as the identity). The input is rotated first."""

        def __init__(self, weight, bias, bits: int, group: int):
            super().__init__()
            self._h = hadamard(block_size(weight.shape[1]))
            self.weight = rotate(weight.astype(mx.float32), self._h)
            self._bias, self._bits, self._group = bias, bits, group

        def __call__(self, x):
            w = self.weight
            w = w + mx.stop_gradient(rounded(w, self._bits, self._group) - w)
            y = rotate(x.astype(mx.float32), self._h) @ w.T
            return y if self._bias is None else y + self._bias

        def finished(self):
            """The same layer with its weight really stored at the low bit width."""
            out_dim, in_dim = self.weight.shape
            plain = nn.Linear(in_dim, out_dim, bias=self._bias is not None)
            plain.weight = self.weight
            if self._bias is not None:
                plain.bias = self._bias
            inner = nn.QuantizedLinear.from_linear(plain, group_size=self._group, bits=self._bits)
            # Scales and offsets stay in single precision while they are trained: Adam's second moment underflows in
            # half precision and the update becomes 0/0. They are counted as half precision in the size.
            inner.scales, inner.biases = inner.scales.astype(mx.float32), inner.biases.astype(mx.float32)
            return RotatedLowBit(inner, self._h)

    class RotatedLowBit(nn.Module):
        """A quantised linear layer that rotates its input first."""

        def __init__(self, inner, h):
            super().__init__()
            self.inner = inner
            self._h = h

        def __call__(self, x):
            return self.inner(rotate(x, self._h.astype(x.dtype)))

    return TrainableLowBit, RotatedLowBit


# ----------------------------------------------------------------------------------------------------------------
# Teacher and data
# ----------------------------------------------------------------------------------------------------------------
def load_teacher(defs: dict, cfg: dict):
    import mlx.core as mx
    from mlx_vlm import load

    model, processor = load(cfg["model_path"])
    lm, layers, _ = defs["attach_lora"](model, cfg["rank"], cfg["alpha"], 0.0, cfg.get("placement", "all_linear"))
    lm.load_weights(str(defs["TARGET_DIR"] / "adapter_good.safetensors"), strict=False)
    defs["set_lora"](layers, training=False)
    mx.eval(lm.parameters())
    return model, getattr(processor, "tokenizer", processor), lm


def encode(defs: dict, tokenizer, row: dict):
    """Input tokens, the completion tokens, and the position the completion starts at (as the notebook does)."""
    prompt = defs["target_prompt"](tokenizer, row)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = tokenizer.encode(prompt + row["completion"] + "<|im_end|>", add_special_tokens=False)
    start = len(prefix)
    if tokens[:start] != prefix:
        start -= 1
    return tokens[:-1], tokens[start:], start - 1


def masks_for(lm, hidden):
    """The mask each layer of the backbone gets, as its own forward pass builds them."""
    module = sys.modules[type(lm.model).__module__]
    config = lm.config
    out = []
    for layer in lm.layers:
        if layer.is_attention_layer:
            window = config.sliding_window if layer.layer_type == "sliding_attention" else None
            out.append(module.create_attention_mask(hidden, None, window_size=window))
        else:
            out.append(module.create_ssm_mask(hidden, None))
    return out


def teacher_states(lm, ids: list[int], upto: int):
    """The teacher's residual stream before each of its first `upto` layers and after the last of them."""
    import mlx.core as mx

    h = lm.model.embed_tokens(mx.array([ids]))
    masks = masks_for(lm, h)
    states = [h]
    for layer, mask in zip(lm.layers[:upto], masks):
        h = layer(h, mask, None)
        states.append(h)
    mx.eval(states)
    return states, masks


def linear_names(block) -> list[str]:
    import mlx.nn as nn

    return [name for name, m in block.named_modules() if isinstance(m, (nn.Linear, nn.QuantizedLinear)) and name.startswith(("self_attn.", "feed_forward.", "conv."))]


def full_weight(layer):
    import mlx.core as mx
    import mlx.nn as nn

    if isinstance(layer, nn.QuantizedLinear):
        return mx.dequantize(layer.weight, layer.scales, layer.biases, layer.group_size, layer.bits)
    return layer.weight


# ----------------------------------------------------------------------------------------------------------------
# Step 3: retrain every weight, a few blocks at a time
# ----------------------------------------------------------------------------------------------------------------
def retrain_blocks(defs, teacher, student, tokenizer, rows, bits: int, group: int, steps: int, chunk: int, lr: float, log) -> list[dict]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_unflatten

    trainable, _ = make_layers()
    order = random.Random(73)
    record = []
    n_layers = len(student.layers)
    for first in range(0, n_layers, chunk):
        span = list(range(first, min(n_layers, first + chunk)))
        for i in span:  # steps 1 and 2: fold the rotation in and switch to rounded weights
            block = student.layers[i]
            swaps = []
            for name in linear_names(block):
                layer = functools.reduce(getattr, name.split("."), block)
                weight = full_weight(layer)
                if weight.shape[1] % group:
                    continue
                swaps.append((name, trainable(weight, getattr(layer, "bias", None) if "bias" in layer else None, bits, group)))
            block.update_modules(tree_unflatten(swaps))
            block.freeze()
            for name, _ in swaps:
                functools.reduce(getattr, name.split("."), block).unfreeze(keys=["weight"])
        mx.eval(student.parameters())

        def loss_of(block, h_in, mask, target):
            out = block(h_in.astype(mx.float32), mask, None)
            return ((out - target) ** 2).mean() / (target**2).mean()

        grads_of = {i: nn.value_and_grad(student.layers[i], loss_of) for i in span}
        optimizers = {i: optim.Adam(learning_rate=lr) for i in span}
        seen = {i: [] for i in span}
        pool = list(rows)
        for step in range(1, steps + 1):
            if (step - 1) % len(pool) == 0:
                order.shuffle(pool)
            ids, _, _ = encode(defs, tokenizer, pool[(step - 1) % len(pool)])
            states, masks = teacher_states(teacher, ids, span[-1] + 1)
            rate = lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - 1) / steps)))
            for i in span:
                target = states[i + 1].astype(mx.float32)
                loss, grads = grads_of[i](student.layers[i], states[i], masks[i], target)
                optimizers[i].learning_rate = rate
                optimizers[i].update(student.layers[i], grads)
                mx.eval(student.layers[i].trainable_parameters(), optimizers[i].state, loss)
                seen[i].append(float(loss.item()))
            del states
            if step == 1 or step % 25 == 0 or step == steps:
                tail = lambda xs: sum(xs[-25:]) / len(xs[-25:])
                log(f"[low-bit {bits}] blocks {span[0] + 1}-{span[-1] + 1} step {step}/{steps} · output error " + " ".join(f"{tail(seen[i]):.3f}" for i in span))
        for i in span:  # store the trained weights at the low bit width and drop the full-precision copies
            block = student.layers[i]
            done = [(name, functools.reduce(getattr, name.split("."), block).finished()) for name in linear_names_trainable(block)]
            block.update_modules(tree_unflatten(done))
            block.freeze()
            record.append({"block": i + 1, "type": block.layer_type, "error_first": round(sum(seen[i][:10]) / len(seen[i][:10]), 4), "error_last": round(sum(seen[i][-25:]) / len(seen[i][-25:]), 4)})
        del grads_of, optimizers
        gc.collect()
        mx.eval(student.parameters())
        mx.clear_cache()
    return record


def linear_names_trainable(block) -> list[str]:
    return [name for name, m in block.named_modules() if type(m).__name__ == "TrainableLowBit"]


# ----------------------------------------------------------------------------------------------------------------
# Step 4: distil end to end
# ----------------------------------------------------------------------------------------------------------------
def attach_student_lora(student, rank: int):
    """A LoRA beside every rotated low-bit layer, and the layer's scales and offsets made trainable."""
    from mlx.utils import tree_unflatten
    from mlx_lm.tuner.lora import LoRALinear

    student.freeze()
    for block in student.layers:
        swaps = []
        for name, module in block.named_modules():
            if type(module).__name__ == "RotatedLowBit":
                lora = LoRALinear.from_base(module.inner, r=rank, scale=2.0, dropout=0.0)
                lora.linear = module
                module.inner.unfreeze(keys=["scales", "biases"])
                swaps.append((name, lora))
        block.update_modules(tree_unflatten(swaps))


def distil(defs, teacher, student, tokenizer, rows, epochs: int, batch: int, lr: float, log, name: str) -> list[dict]:
    """Train the student's adapter, scales and offsets so that its next-token distribution over each call matches
    the teacher's (KL divergence), plus the ordinary loss on the reference call, half and half."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map

    def logits_of(lm, ids, start):
        hidden = lm.model(mx.array([ids]))[:, start:]
        return (lm.model.embed_tokens.as_linear(hidden) if lm.config.tie_word_embeddings else lm.lm_head(hidden)).astype(mx.float32)

    def loss_of(lm, ids, expected, start, soft):
        log_p = nn.log_softmax(logits_of(lm, ids, start), axis=-1)
        kl = (mx.exp(soft) * (soft - log_p)).sum(axis=-1).mean()
        ce = nn.losses.cross_entropy(log_p, mx.array([expected]), reduction="mean")
        return 0.5 * kl + 0.5 * ce, kl

    grads_of = nn.value_and_grad(student, lambda lm, *a: loss_of(lm, *a)[0])
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.0)
    total = math.ceil(len(rows) / batch) * epochs
    order, rows, history, step = random.Random(73), list(rows), [], 0
    for _ in range(epochs):
        order.shuffle(rows)
        for offset in range(0, len(rows), batch):
            grads, losses = None, []
            for row in rows[offset : offset + batch]:
                ids, expected, start = encode(defs, tokenizer, row)
                soft = mx.stop_gradient(nn.log_softmax(logits_of(teacher, ids, start), axis=-1))
                mx.eval(soft)
                loss, grad = grads_of(student, ids, expected, start, soft)
                grads = grad if grads is None else tree_map(lambda a, b: a + b, grads, grad)
                mx.eval(grads, loss)
                losses.append(float(loss.item()))
            step += 1
            optimizer.learning_rate = defs["learning_rate"](step, total, lr)
            clipped, _ = optim.clip_grad_norm(tree_map(lambda g: g / len(losses), grads), 1.0)
            optimizer.update(student, clipped)
            mx.eval(student.trainable_parameters(), optimizer.state)
            history.append({"step": step, "loss": sum(losses) / len(losses)})
            if not math.isfinite(history[-1]["loss"]):
                raise FloatingPointError(f"distillation loss is not a number at step {step}")
            if step == 1 or step % 10 == 0 or step == total:
                log(f"[low-bit {name}] distil step {step}/{total} loss {history[-1]['loss']:.3f}")
    return history


# ----------------------------------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------------------------------
def write_call(defs, lm, tokenizer, row, max_new: int) -> str:
    import mlx.core as mx

    stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    tokens = mx.array([tokenizer.encode(defs["target_prompt"](tokenizer, row), add_special_tokens=False)])
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


def agreement(defs, teacher, student, tokenizer, rows) -> dict:
    """On the reference calls of `rows`: how often the student's first-choice token is the teacher's, and the
    student's mean loss on the reference tokens. A quick reading that needs no execution."""
    import mlx.core as mx
    import mlx.nn as nn

    same = total = 0
    loss = 0.0
    for row in rows:
        ids, expected, start = encode(defs, tokenizer, row)
        outs = []
        for lm in (teacher, student):
            hidden = lm.model(mx.array([ids]))[:, start:]
            outs.append((lm.model.embed_tokens.as_linear(hidden) if lm.config.tie_word_embeddings else lm.lm_head(hidden)).astype(mx.float32))
        same += int((outs[0].argmax(axis=-1) == outs[1].argmax(axis=-1)).sum().item())
        total += len(expected)
        loss += float(nn.losses.cross_entropy(outs[1], mx.array([expected]), reduction="mean").item())
    return {"same_first_choice": round(same / total, 4), "reference_loss": round(loss / len(rows), 4)}


def run(bits: int, *, group: int = 128, chunk: int = 4, block_steps: int = 300, block_lr: float = 2e-5, rank: int = 32, epochs: int = 2,
        distil_lr: float = 1e-4, limit: int | None = None, smoke: bool = False, log=print) -> dict:
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_vlm import load

    from defs import load_defs

    started = time.monotonic()
    defs = load_defs()
    dataset = defs["dataset"]
    target_dir = defs["TARGET_DIR"]
    cfg = json.loads((target_dir / "report.json").read_text(encoding="utf-8"))["config"]
    mx.set_cache_limit(512 * 1024**2)
    mx.random.seed(cfg["seed"])
    if smoke:
        block_steps, epochs, limit = 4, 1, 4

    rng = random.Random(cfg["seed"])
    train = dataset["target"]["train"]
    chains = sorted({r["chain_id"] for r in train})
    rng.shuffle(chains)
    probe = set(chains[: max(1, round(len(chains) * cfg["probe_share"]))])
    fit_rows = [r for r in train if r["chain_id"] not in probe]
    dev = sorted(dataset["target"]["dev"], key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))
    eval_rows = dev[:limit] if limit else dev
    quick_rows = dev[:8] if smoke else dev[:24]
    if smoke:
        fit_rows = fit_rows[:8]

    _teacher_model, tokenizer, teacher = load_teacher(defs, cfg)
    student_model, _ = load(cfg["model_path"])
    student = student_model.language_model
    student.freeze()
    out = {"bits": bits, "group": group, "dataset": dataset["fingerprint"], "stages": {}}
    name = f"{bits}-bit"

    saved_blocks = target_dir / f"low_bit_{bits}_blocks.safetensors"
    if saved_blocks.exists() and not smoke:  # the block stage is the slow one: reuse it when it is already on disk
        trainable, _ = make_layers()
        for block in student.layers:
            swaps = []
            for layer_name in linear_names(block):
                layer = functools.reduce(getattr, layer_name.split("."), block)
                weight = full_weight(layer)
                if weight.shape[1] % group == 0:
                    swaps.append((layer_name, trainable(weight, None, bits, group).finished()))
            block.update_modules(tree_unflatten(swaps))
        student.load_weights(str(saved_blocks), strict=False)
        student.freeze()
        mx.eval(student.parameters())
        blocks = json.loads(saved_blocks.with_suffix(".json").read_text(encoding="utf-8"))
        log(f"[low-bit {name}] retrained blocks loaded from {saved_blocks.name}")
    else:
        blocks = retrain_blocks(defs, teacher, student, tokenizer, fit_rows, bits, group, block_steps, chunk, block_lr, log)
        if not smoke:
            mx.save_safetensors(str(saved_blocks), {k: v for k, v in tree_flatten(student.parameters()) if ".layers." in k})
            saved_blocks.with_suffix(".json").write_text(json.dumps(blocks), encoding="utf-8")
    out["blocks"] = blocks
    out["stages"]["after retraining the blocks"] = agreement(defs, teacher, student, tokenizer, quick_rows)
    log(f"[low-bit {name}] after retraining the blocks: {out['stages']['after retraining the blocks']}")

    attach_student_lora(student, rank)
    out["history"] = distil(defs, teacher, student, tokenizer, fit_rows, epochs, cfg["batch"], distil_lr, log, name)
    out["stages"]["after distilling"] = agreement(defs, teacher, student, tokenizer, quick_rows)
    log(f"[low-bit {name}] after distilling: {out['stages']['after distilling']}")
    # the finished model's blocks (low-bit weights, their scales and offsets, and the distilled LoRA)
    mx.save_safetensors(str(target_dir / f"low_bit_{bits}_final.safetensors"), {k: v for k, v in tree_flatten(student.parameters()) if ".layers." in k})

    n_weights = sum(v.size * 32 // bits for k, v in tree_flatten(student.parameters()) if k.endswith(".inner.weight"))
    stored = sum(v.nbytes if k.endswith(".weight") else v.size * 2 for k, v in tree_flatten(student.parameters()) if ".inner." in k and k.rsplit(".", 1)[-1] in ("weight", "scales", "biases"))
    out["size"] = {"block_weights": n_weights, "megabytes": round(stored / 1e6, 1), "bits_per_weight": round(stored * 8 / max(1, n_weights), 3)}
    log(f"[low-bit {name}] block weights: {n_weights / 1e6:.0f} M stored in {stored / 1e6:.0f} MB ({out['size']['bits_per_weight']} bits each)")

    calls = []
    for done, row in enumerate(eval_rows, 1):
        calls.append(write_call(defs, student, tokenizer, row, cfg["max_new"]))
        if done % 40 == 0 or done == len(eval_rows):
            log(f"[low-bit {name}] wrote {done}/{len(eval_rows)} calls")
    scores = [defs["score_call"](text, row) for text, row in zip(calls, eval_rows)]
    rate = lambda key: sum(s[key] for s in scores) / len(scores)
    out["execution"] = {"requests": len(scores), "well_formed": rate("well_formed"), "runs": rate("runs"), "correct": rate("correct")}
    out["rows"] = [{"id": r["id"], "correct": s["correct"], "code": s["code"]} for r, s in zip(eval_rows, scores)]
    out["seconds"] = round(time.monotonic() - started)
    log(f"[low-bit {name}] executed: {out['execution']}")
    if not smoke:
        path = target_dir / f"low_bit_{bits}.json"
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        log(f"saved {path.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--block-steps", type=int, default=300)
    parser.add_argument("--chunk", type=int, default=4, help="blocks retrained together")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit", type=int, help="score only the first N held-out requests")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.bits, chunk=args.chunk, block_steps=args.block_steps, rank=args.rank, epochs=args.epochs, limit=args.limit, smoke=args.smoke)
