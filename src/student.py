"""Pruned-from-teacher student. This is what makes distillation feasible on 8 GB."""
import gc
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM


def pick_layer_indices(total_layers: int, keep: int) -> list[int]:
    """Evenly spaced layer selection, always including first and last."""
    if keep >= total_layers:
        return list(range(total_layers))
    idx = np.linspace(0, total_layers - 1, keep).round().astype(int).tolist()
    return idx


def build_pruned_student(teacher_model, keep_layers: int) -> LlamaForCausalLM:
    """Create a smaller Llama by copying a subset of teacher layers."""
    tcfg = teacher_model.config
    keep = pick_layer_indices(tcfg.num_hidden_layers, keep_layers)

    scfg = LlamaConfig(**tcfg.to_dict())
    scfg.num_hidden_layers = keep_layers
    student = LlamaForCausalLM(scfg)

    t_sd = teacher_model.state_dict()
    s_sd = student.state_dict()
    new_sd = {}
    for k in s_sd.keys():
        if ".layers." in k:
            # k looks like "model.layers.<i>.<rest>"
            head, tail = k.split(".layers.", 1)
            local_i = int(tail.split(".", 1)[0])
            rest = tail.split(".", 1)[1]
            src = keep[local_i]
            new_sd[k] = t_sd[f"{head}.layers.{src}.{rest}"]
        else:
            new_sd[k] = t_sd[k]  # embed_tokens, norm, lm_head, rotary_emb
    student.load_state_dict(new_sd)
    del t_sd, s_sd, new_sd
    gc.collect()
    return student


def parameter_report(student):
    total = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    return total, trainable


def estimate_training_memory_gb(trainable_params: int, frozen_params: int,
                                batch: int, seq: int, hidden: int,
                                layers: int, grad_ckpt: bool) -> float:
    """Conservative peak. Accounts for AdamW state, grads, activations."""
    train_bytes = trainable_params * (4 + 4 + 8)   # weight + grad + AdamW(m,v)
    frozen_bytes = frozen_params * 4
    # Activations scale with layers (no ckpt) or ~1 layer (with ckpt)
    act_units = 2 if grad_ckpt else layers
    act_bytes = batch * seq * hidden * act_units * 4 * 8   # ~8 tensors/layer
    # Logits buffer (B, seq, V=32000) fp32
    logits_bytes = batch * seq * 32000 * 4
    return (train_bytes + frozen_bytes + act_bytes + logits_bytes) / (1024 ** 3)