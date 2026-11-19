"""LoRA on the linear layers of an MLX model: where it goes, how it is switched, and the learning-rate schedule."""

import math

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
