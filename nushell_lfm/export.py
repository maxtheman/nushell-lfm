"""Export for the browser: the adapter added to the 4-bit ONNX build, and the check with ONNX Runtime."""

from pathlib import Path
import gc
import json
import shutil
import time

from .core import ROOT
from .calls import parse_tool_call
from .target import TARGET_DIR

BROWSER_DIR = ROOT / "webgpu-demo"
BASE_ONNX_DIR = BROWSER_DIR / "models" / "onnx-community" / "LFM3-1.7B-ONNX"
EXPORT_ONNX_DIR = BROWSER_DIR / "models" / "nushell-lfm3-1.7b"
# (name in the ONNX graph, name in the MLX model); the graph keeps the attention gate as the last rows of q_proj
_LAYERS = [
    ("conv_in_proj", "conv.in_proj"), ("conv_out_proj", "conv.out_proj"),
    ("mlp_gate_proj", "feed_forward.w1"), ("mlp_up_proj", "feed_forward.w3"), ("mlp_down_proj", "feed_forward.w2"),
    ("attn_q_proj", "self_attn.q_proj"), ("attn_k_proj", "self_attn.k_proj"),
    ("attn_v_proj", "self_attn.v_proj"), ("attn_o_proj", "self_attn.o_proj"),
]

def export_for_browser(model_path: str, adapter_path, rank: int, alpha: float, log=print) -> dict:
    """Put a LoRA adapter into the 4-bit ONNX graph without touching the quantised weights: beside each
    4-bit matrix product y = x·W, add the low-rank branch (x·A)·B and sum the two. The adapter stays in
    half precision and exact; re-quantising W + A·B to 4 bits instead loses much of the small update.
    Runs on the MLX thread."""
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    import onnx
    from mlx_vlm import load as vlm_load
    from onnx import helper, numpy_helper

    started = time.monotonic()
    graph = onnx.load(str(BASE_ONNX_DIR / "onnx" / "model_q4f16.onnx"))
    tensors = {t.name: t for t in graph.graph.initializer}
    producers = {n.input[1]: n for n in graph.graph.node if n.op_type == "MatMulNBits"}
    adapter = {k: np.array(v.astype(mx.float32)) for k, v in mx.load(str(adapter_path)).items()}

    def nibbles(packed):  # two 4-bit values per byte, low half first
        return np.stack([packed & 15, packed >> 4], -1).reshape(*packed.shape[:-1], -1)

    def read(prefix):
        """The quantised weight as floats, [outputs, inputs]. Only used to line rows up with the MLX model."""
        packed = numpy_helper.to_array(tensors[prefix + "_quant"])
        scales = numpy_helper.to_array(tensors[prefix + "_scales"]).astype(np.float32)
        rows, blocks, _ = packed.shape
        zero = nibbles(numpy_helper.to_array(tensors[prefix + "_zp"]))[:, :blocks] if prefix + "_zp" in tensors else np.full((rows, blocks), 8)
        return ((nibbles(packed).astype(np.float32) - zero[..., None]) * scales[..., None]).reshape(rows, -1)

    def factors(layer: int, name: str):
        """A [inputs, rank] and B [rank, outputs], with the LoRA scale folded into B."""
        return adapter[f"model.layers.{layer}.{name}.lora_a"], (alpha / rank) * adapter[f"model.layers.{layer}.{name}.lora_b"]

    def row_order(onnx_rows, mlx_rows, head: int):
        """The ONNX build stores the outputs of q_proj and k_proj in a different order inside each attention
        head. Recover it from the first head by matching rows, and check it on the whole weight."""
        unit = lambda m: (m - m.mean(1, keepdims=True)) / (np.linalg.norm(m - m.mean(1, keepdims=True), axis=1, keepdims=True) + 1e-9)
        inside = (unit(onnx_rows[:head]) @ unit(mlx_rows[:head]).T).argmax(1)
        order = np.concatenate([inside + start for start in range(0, mlx_rows.shape[0], head)])
        match = float(np.mean(np.sum(unit(onnx_rows) * unit(mlx_rows[order]), axis=1)))
        if sorted(inside.tolist()) != list(range(head)) or match < 0.95:
            raise ValueError(f"could not align attention rows (match {match:.2f})")
        return order

    model, _processor = vlm_load(model_path)
    try:
        head = getattr(model.language_model.config, "head_dim", 128)
        added, extra = 0, []
        for index, block in enumerate(model.language_model.layers):
            modules = dict(block.named_modules())
            for onnx_name, mlx_name in _LAYERS:
                prefix = f"model_layers_{index}_{onnx_name}_MatMul_weight"
                node = producers.get(prefix + "_quant")
                if node is None or f"model.layers.{index}.{mlx_name}.lora_a" not in adapter:
                    continue
                a, b = factors(index, mlx_name)
                if onnx_name in ("attn_q_proj", "attn_k_proj"):
                    layer = modules[mlx_name]
                    base = np.array(mx.dequantize(layer.weight, layer.scales, layer.biases, layer.group_size, layer.bits).astype(mx.float32)) if isinstance(layer, nn.QuantizedLinear) else np.array(layer.weight.astype(mx.float32))
                    weight = read(prefix)
                    b = b[:, row_order(weight[: base.shape[0]], base, head)]
                    if onnx_name == "attn_q_proj" and weight.shape[0] > b.shape[1] and f"model.layers.{index}.self_attn.gate_proj.lora_a" in adapter:
                        # the graph computes the attention gate as extra outputs of q_proj: stack the two adapters
                        gate_a, gate_b = factors(index, "self_attn.gate_proj")
                        a = np.concatenate([a, gate_a], axis=1)
                        b = np.block([[b, np.zeros((b.shape[0], gate_b.shape[1]), np.float32)], [np.zeros((gate_b.shape[0], b.shape[1]), np.float32), gate_b]])
                name = f"lora_{index}_{onnx_name}"
                graph.graph.initializer.extend(
                    [numpy_helper.from_array(a.astype(np.float16), name + "_A"), numpy_helper.from_array(b.astype(np.float16), name + "_B")]
                )
                output = node.output[0]
                node.output[0] = output + "_base"
                extra.append(
                    (
                        node.name,
                        [
                            helper.make_node("MatMul", [node.input[0], name + "_A"], [name + "_down"], name=name + "/down"),
                            helper.make_node("MatMul", [name + "_down", name + "_B"], [name + "_up"], name=name + "/up"),
                            helper.make_node("Add", [output + "_base", name + "_up"], [output], name=name + "/add"),
                        ],
                    )
                )
                added += 1
    finally:
        del model, _processor
        gc.collect()
        mx.clear_cache()

    after = dict(extra)  # keep the graph in execution order: each branch right after the product it joins
    nodes = []
    for node in graph.graph.node:
        nodes.append(node)
        nodes.extend(after.get(node.name, []))
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)

    if EXPORT_ONNX_DIR.exists():
        shutil.rmtree(EXPORT_ONNX_DIR)
    (EXPORT_ONNX_DIR / "onnx").mkdir(parents=True)
    onnx.save(
        graph, str(EXPORT_ONNX_DIR / "onnx" / "model_q4f16.onnx"), save_as_external_data=True,
        all_tensors_to_one_file=True, location="model_q4f16.onnx_data", size_threshold=1024,
    )
    for item in BASE_ONNX_DIR.iterdir():
        if item.is_file():
            shutil.copyfile(item, EXPORT_ONNX_DIR / item.name)
    shutil.copyfile(TARGET_DIR / "prompt_format.json", BROWSER_DIR / "prompt_format.json")
    summary = {
        "adapted_weights": added,
        "seconds": round(time.monotonic() - started, 1),
        "size_mb": round(sum(f.stat().st_size for f in (EXPORT_ONNX_DIR / "onnx").iterdir()) / 1e6),
    }
    log(f"[export] added the adapter beside {added} weights, {summary['size_mb']} MB")
    return summary

def check_browser_export(rows: list[dict], score_call, target_prompt, max_new: int = 220, model_dir=None, log=print) -> dict:
    """Generate with the exported ONNX model on ONNX Runtime, using the tokenizer and chat template that
    ship in the export, and execute each call. This is the same graph and prompt the browser runs."""
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_dir = Path(model_dir or EXPORT_ONNX_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    session = ort.InferenceSession(str(model_dir / "onnx" / "model_q4f16.onnx"), providers=["CPUExecutionProvider"])
    past_names = [i.name for i in session.get_inputs() if i.name.startswith("past")]
    shapes = {i.name: i.shape for i in session.get_inputs()}
    outputs = [o.name for o in session.get_outputs()]
    stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    results = []
    for row in rows:
        ids = tokenizer.encode(target_prompt(tokenizer, row), add_special_tokens=False)
        past = {name: np.zeros([1 if isinstance(d, str) and "batch" in d else 0 if isinstance(d, str) else d for d in shapes[name]], np.float16) for name in past_names}
        feed, written, seen = np.array([ids], np.int64), [], len(ids)
        for _ in range(max_new):
            out = session.run(outputs, {"input_ids": feed, "attention_mask": np.ones((1, seen), np.int64), "num_logits_to_keep": np.array(1, np.int64), **past})
            token = int(out[0][0, -1].argmax())
            past = {name.replace("present_conv", "past_conv").replace("present.", "past_key_values."): value for name, value in zip(outputs[1:], out[1:])}
            if token in stop:
                break
            written.append(token)
            if tokenizer.decode(written).rstrip().endswith("</tool_call>"):
                break
            feed, seen = np.array([[token]], np.int64), seen + 1
        text = tokenizer.decode(written)
        results.append({"request": row["speech"], "output": text.strip(), **score_call(text, row)})
    rate = lambda key: sum(r[key] for r in results) / max(1, len(results))
    summary = {"requests": len(results), "well_formed": rate("well_formed"), "runs": rate("runs"), "correct": rate("correct"), "samples": results}
    log(f"[export] ONNX Runtime on {len(results)} held-out requests: {summary['correct']:.0%} return the reference value")
    return summary

def write_browser_requests(rows: list[dict], environments: list[dict]) -> None:
    """What the demo page needs besides the model: each held-out environment's files and requests."""
    projects = []
    for environment in environments:
        own = [r for r in rows if r["env"] == environment["id"]]
        if own:
            projects.append(
                {
                    "id": environment["id"],
                    "title": environment["title"],
                    "context": own[0]["context"],
                    "files": environment["files"],
                    # `code` is the reference call: the page runs it in its own Nushell to get the value to compare with,
                    # because directory order and directory sizes differ between a real disk and the browser.
                    "requests": [
                        {"request": r["speech"], "references": r["references"], "expected": r["expected"], "code": parse_tool_call(r["completion"])}
                        for r in own
                    ],
                }
            )
    (BROWSER_DIR / "environments.json").write_text(json.dumps(projects, indent=1), encoding="utf-8")
