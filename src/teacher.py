"""Load TinyLlama frozen in float32. Never used during student training."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_teacher(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok