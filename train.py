#!/usr/bin/env python3
"""Train a completion-only Nushell LoRA on a local MLX LFM3-Omni checkpoint."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_vlm import load

HERE = Path(__file__).resolve().parent
SYSTEM = (
    "Make one run_nushell tool call for the request. Preserve every requested step in order. "
    "Use the user's values and workspace context. Use references for syntax. "
    "Code contains only Nushell source. Use | to json --raw for requested JSON."
)
TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_nushell",
        "description": (
            "Execute Nushell (Nu) source in /workspace. Include definitions, pipelines and invocations. "
            "For a requested interactive Nu UI only, supply ui_keys such as Down and Enter."
        ),
        "parameters": {
            "type": "object",
            "required": ["code"],
            "properties": {
                "code": {"type": "string"},
                "ui_keys": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}]


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_dataset(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def encode(tokenizer, row):
    refs = "\n\n".join(f"{doc['title']} (nushell)\n{doc['text']}" for doc in row["references"])
    text = f"\nSyntax references:\n{refs}\n\n" if refs else "\n"
    text += f"Spoken request transcript: {row['speech']}"
    if row["context"] or row["workspace_files"]:
        text += "\nTerminal fixture: " + row["context"]
        if row["workspace_files"]:
            text += "\nWorkspace files: " + ", ".join(row["workspace_files"])
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ], tools=TOOLS, tokenize=False, add_generation_prompt=True) + "<tool_call>\n"
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = tokenizer.encode(prompt + row["completion"] + "<|im_end|>", add_special_tokens=False)
    start = len(prefix)
    # The final prefix newline can merge with the first completion token.
    if tokens[:start] != prefix:
        start -= 1
    if not 0 < start < len(tokens) or tokens[:start] != prefix[:start]:
        raise ValueError(f"Assistant completion is misaligned for {row['id']}")
    return tokens[:-1], tokens[start:], start - 1


def attach(model, rank, alpha):
    model.freeze()
    lm = model.language_model
    projections = Counter()
    for block in lm.layers:
        replacements = []
        for name, layer in block.named_modules():
            if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
                replacements.append((name, LoRALinear.from_base(layer, r=rank, scale=alpha / rank)))
                projections[name] += 1
        block.update_modules(tree_unflatten(replacements))
    weights = tree_flatten(lm.trainable_parameters())
    if not weights or any(not key.endswith(("lora_a", "lora_b")) for key, _ in weights):
        raise ValueError("Unexpected trainable parameter scope")
    return lm, projections, sum(value.size for _, value in weights)


def completion_loss(lm, inputs, expected, start):
    backbone = lm.model
    hidden = backbone(mx.array([inputs]))[:, start:]
    if lm.config.tie_word_embeddings:
        logits = backbone.embed_tokens.as_linear(hidden)
    else:
        logits = lm.lm_head(hidden)
    return nn.losses.cross_entropy(logits.astype(mx.float32), mx.array([expected]), reduction="mean")


def save_checkpoint(out, lm, report):
    adapter = out / "adapter.safetensors"
    pending = out / ".adapter.pending.safetensors"
    mx.save_safetensors(str(pending), dict(tree_flatten(lm.trainable_parameters())))
    pending.replace(adapter)
    report_path = out / "training-report.json"
    pending_report = out / ".training-report.pending.json"
    pending_report.write_text(json.dumps({**report, "adapter_sha256": sha256(adapter)}, indent=2) + "\n")
    pending_report.replace(report_path)


def train(args, rows, development, manifest):
    mx.set_cache_limit(512 * 1024**2)
    mx.random.seed(args.seed)
    model, processor = load(args.model)
    lm, projections, parameters = attach(model, args.rank, args.alpha)
    grad_checkpoint(lm.layers[0])
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.0)
    value_and_grad = nn.value_and_grad(lm, completion_loss)
    dev = [encode(processor.tokenizer, row) for row in development]
    before = sum(completion_loss(lm, *example).item() for example in dev) / len(dev)
    total_steps = math.ceil(len(rows) / args.batch_size) * args.epochs
    print(f"{len(rows)} training rows, {len(dev)} dev rows, {parameters:,} trainable weights", flush=True)
    step = input_tokens = completion_tokens = 0
    start_time = time.monotonic()
    for _ in range(args.epochs):
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            grads = None
            losses = []
            for row in batch:
                example = encode(processor.tokenizer, row)
                input_tokens += len(example[0])
                completion_tokens += len(example[1])
                loss, grad = value_and_grad(lm, *example)
                grads = grad if grads is None else tree_map(lambda a, b: a + b, grads, grad)
                mx.eval(grads)
                losses.append(loss.item())
            step += 1
            decay = 0.2 + 0.4 * (1 + math.cos(math.pi * (step - 1) / total_steps))
            optimizer.learning_rate = args.learning_rate * min(1, step / 8) * decay
            average = tree_map(lambda g: g / len(batch), grads)
            clipped, _ = optim.clip_grad_norm(average, 1.0)
            optimizer.update(lm, clipped)
            mx.eval(lm.trainable_parameters(), optimizer.state)
            if step == 1 or step % 10 == 0 or step == total_steps:
                print(f"{step}/{total_steps}  loss {sum(losses)/len(losses):.4f}", flush=True)
    after = sum(completion_loss(lm, *example).item() for example in dev) / len(dev)
    save_checkpoint(args.output, lm, {
        "base_weight_sha256": manifest["base_weight_sha256"], "training_data_sha256": manifest["files"],
        "train_ids": [row["id"] for row in rows], "dev_ids": [row["id"] for row in development],
        "rank": args.rank, "alpha": args.alpha, "scale": args.alpha / args.rank,
        "learning_rate": args.learning_rate, "batch_size": args.batch_size, "epochs": args.epochs,
        "seed": args.seed, "steps": step, "projections": dict(projections), "trainable_parameters": parameters,
        "dev_loss_before": before, "dev_loss_after": after, "input_tokens": input_tokens,
        "completion_tokens": completion_tokens, "seconds": time.monotonic() - start_time,
        "peak_mlx_bytes": mx.get_peak_memory(),
    })
    print(f"Adapter saved to {args.output}; dev loss {before:.4f} → {after:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local LFM3-Omni-2B MLX 8-bit model folder.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this run.")
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--limit", type=int, help="Optional prefix of the training split for a smoke test.")
    args = parser.parse_args()
    if min(args.rank, args.alpha, args.learning_rate, args.batch_size, args.epochs) <= 0:
        parser.error("Rank, alpha, learning rate, batch size and epochs must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    args.model = args.model.expanduser().resolve()
    manifest = json.loads((args.data / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(args.data / name) != expected:
            raise ValueError(f"Unexpected training data contents: {name}")
    if sha256(args.model / "model.safetensors") != manifest["base_weight_sha256"]:
        raise ValueError("Base weights differ from the pinned LFM3-Omni-2B MLX checkpoint")
    rows = read_dataset(args.data / "train.jsonl")
    dev = read_dataset(args.data / "dev.jsonl")
    if len(rows) != manifest["training_examples"] or len(dev) != manifest["development_examples"]:
        raise ValueError("Dataset row count differs from manifest")
    args.output.mkdir(parents=True)
    train(args, rows[:args.limit], dev, manifest)


if __name__ == "__main__":
    main()
