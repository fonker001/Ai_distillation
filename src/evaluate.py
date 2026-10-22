"""Held-out KD loss + CE + sample generations. No teacher needed if you
already have the cache; teacher is only used for prompt-side comparison."""
import json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, AutoTokenizer
from .dataset import TopKCacheDataset, collate
from .loss import distillation_loss


def evaluate(cfg):
    device = "cpu"
    student = LlamaForCausalLM.from_pretrained(cfg["student_path"],
                                               torch_dtype=torch.float32).to(device).eval()
    tok = AutoTokenizer.from_pretrained(cfg["student_path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- 1. Held-out KD loss + CE ---
    val_ds = TopKCacheDataset(cfg["cache_dir"], "val", 0.05)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)
    kd_losses, ce_losses = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = student(input_ids=batch["input_ids"], use_cache=False).logits
            kd = distillation_loss(
                logits[:, :-1, :], batch["targets"],
                batch["topk_logits"], batch["topk_ids"],
                temperature=2.0, kd_alpha=1.0)          # pure KD
            ce = distillation_loss(
                logits[:, :-1, :], batch["targets"],
                batch["topk_logits"], batch["topk_ids"],
                temperature=2.0, kd_alpha=0.0)          # pure CE
            kd_losses.append(kd.item()); ce_losses.append(ce.item())
    kd_mean = sum(kd_losses) / max(len(kd_losses), 1)
    ce_mean = sum(ce_losses) / max(len(ce_losses), 1)
    ppl = float(torch.exp(torch.tensor(ce_mean)))

    # --- 2. Sample generations ---
    prompts = []
    with open(cfg["prompts_path"]) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["prompt"])

    samples = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        t0 = time.time()
        with torch.no_grad():
            out = student.generate(
                input_ids=ids, max_new_tokens=cfg["max_new_tokens"],
                do_sample=True, temperature=cfg["temperature"], top_p=cfg["top_p"],
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        latency = time.time() - t0
        reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        samples.append({"prompt": p, "student": reply, "latency_s": round(latency, 2)})

    report = {
        "student_path": cfg["student_path"],
        "heldout_kd_loss": kd_mean,
        "heldout_ce_loss": ce_mean,
        "heldout_perplexity": ppl,
        "samples": samples,
        "caveat": ("Lower loss does NOT mean the student matches the teacher. "
                   "Review the samples; the KD loss only measures agreement on "
                   "the top-k teacher distribution."),
    }
    Path(cfg["output_report"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["output_report"]).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in
                      ("heldout_kd_loss", "heldout_ce_loss", "heldout_perplexity")}, indent=2))
    for s in samples:
        print(f"\n[prompt] {s['prompt']}\n[student] {s['student']}")
    return report