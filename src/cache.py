"""Top-k teacher logit cache via llama.cpp. Teacher loaded once per run."""
import json, time
from pathlib import Path
import numpy as np
import torch
from llama_cpp import Llama
from transformers import AutoTokenizer

VERSION = 2


def load_llm(path, n_ctx=2048, n_threads=4, n_batch=512):
    return Llama(
        model_path=path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        logits_all=True,          # required to expose .scores for every token
        verbose=False,
    )


def build_cache(cfg):
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"

    start_shard = 0
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        if m.get("version") != VERSION:
            raise RuntimeError(
                f"Cache version mismatch (found {m.get('version')}, need {VERSION}). "
                f"Delete {out_dir} and rebuild.")
        start_shard = m.get("completed_shards", 0)
        print(f"[cache] resuming after shard {start_shard}")

    llm = load_llm(
        cfg["teacher_gguf_path"],
        n_ctx=cfg["sequence_length"] + 64,
        n_threads=cfg.get("n_threads", 4),
        n_batch=cfg.get("n_batch", 512),
    )
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_model_id"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    top_k = cfg["top_k"]
    seq_len = cfg["sequence_length"]
    vocab = llm.n_vocab()

    samples, shard_idx, total = [], start_shard, 0

    def flush():
        nonlocal samples, shard_idx
        if not samples:
            return
        torch.save(samples, out_dir / f"shard_{shard_idx:05d}.pt")
        samples = []; shard_idx += 1
        meta_path.write_text(json.dumps({
            "version": VERSION, "completed_shards": shard_idx,
            "top_k": top_k, "sequence_length": seq_len,
            "teacher_model": "tinyllama-1.1b-chat-q4_k_m",
            "vocab_size": vocab,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }))

    with open(cfg["conversations_path"], encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = tok.apply_chat_template(rec["messages"], tokenize=False)
            tokens = tok.encode(text)
            if len(tokens) < 4:
                continue
            tokens = tokens[:seq_len]

            llm.reset()
            llm.eval(tokens)
            scores = np.array(llm.scores, dtype=np.float32)   # (len(tokens), vocab)

            # Next-token alignment: position i predicts token i+1
            pred_scores = scores[:-1]                          # (T-1, vocab)
            targets = np.array(tokens[1:], dtype=np.int32)     # (T-1,)

            # top-k
            topk_idx = np.argpartition(pred_scores, -top_k, axis=-1)[:, -top_k:]
            topk_vals = np.take_along_axis(pred_scores, topk_idx, axis=-1)
            # sort descending for consistency
            order = np.argsort(-topk_vals, axis=-1)
            topk_idx = np.take_along_axis(topk_idx, order, axis=-1)
            topk_vals = np.take_along_axis(topk_vals, order, axis=-1)

            samples.append({
                "input_ids":  torch.tensor(tokens, dtype=torch.int32),
                "targets":    torch.tensor(targets, dtype=torch.int32),
                "topk_logits": torch.tensor(topk_vals, dtype=torch.float32),
                "topk_ids":   torch.tensor(topk_idx, dtype=torch.int32),
            })
            total += 1
            if len(samples) >= cfg["shard_size"]:
                flush()
                print(f"[cache] wrote shard {shard_idx - 1}, total {total}")

    flush()
    meta_path.write_text(json.dumps({
        "version": VERSION, "completed_shards": shard_idx, "total_samples": total,
        "top_k": top_k, "sequence_length": seq_len,
        "teacher_model": "tinyllama-1.1b-chat-q4_k_m",
        "vocab_size": vocab,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))
    print(f"[cache] done: {total} samples, {shard_idx} shards -> {out_dir}")