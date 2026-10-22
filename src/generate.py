"""Teacher generation via llama.cpp (GGUF). Much faster than PyTorch on CPU."""
import json, hashlib, random, time
from pathlib import Path
from llama_cpp import Llama
from transformers import AutoTokenizer

SYSTEM = "You are a friendly, helpful conversational assistant. Keep replies short and natural."


def _hash(msgs):
    return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()[:16]


def load_llm(path, n_ctx=2048, n_threads=4, n_batch=512, logits_all=False):
    return Llama(
        model_path=path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        logits_all=logits_all,
        verbose=False,
    )


def load_tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _gen_one(llm, tok, messages, max_new, temp, top_p, sample):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = llm.create_completion(
        prompt=prompt,
        max_tokens=max_new,
        temperature=temp if sample else 0.0,
        top_p=top_p if sample else 1.0,
        echo=False,
    )
    return out["choices"][0]["text"].strip()


def generate_all(cfg):
    out_path = Path(cfg["output_path"])
    ckpt_path = Path(cfg["checkpoint_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start, seen = 0, set()
    if ckpt_path.exists():
        ck = json.loads(ckpt_path.read_text())
        start, seen = ck["next"], set(ck["seen"])
        print(f"[gen] resuming at {start}")

    llm = load_llm(
        cfg["teacher_gguf_path"],
        n_ctx=cfg.get("n_ctx", 2048),
        n_threads=cfg.get("n_threads", 4),
        n_batch=cfg.get("n_batch", 512),
    )
    tok = load_tokenizer(cfg["tokenizer_model_id"])
    rng = random.Random(cfg["seed"])
    mode = "a" if start > 0 else "w"

    with open(out_path, mode, encoding="utf-8") as f:
        for i in range(start, cfg["num_conversations"]):
            n = rng.randint(*cfg["turns_per_conversation"])
            messages = [{"role": "system", "content": SYSTEM}]
            messages.append({"role": "user", "content": rng.choice(
                ["Hi there!", "Hello!", "Hey, how's it going?", "Good morning!"])})
            ok = True
            for t in range(n):
                reply = _gen_one(
                    llm, tok, messages,
                    cfg["max_new_tokens_per_turn"],
                    cfg["temperature"], cfg["top_p"], cfg["do_sample"],
                )
                if len(reply) < cfg["min_response_chars"]:
                    ok = False; break
                messages.append({"role": "assistant", "content": reply})
                if t == n - 1:
                    break
                follow = _gen_one(
                    llm, tok,
                    messages + [{"role": "user",
                                 "content": "Ask a short, natural follow-up question."}],
                    48, cfg["temperature"], cfg["top_p"], cfg["do_sample"],
                )
                if len(follow) < 5:
                    follow = rng.choice(["What about you?", "Anything else?",
                                         "How has your day been?"])
                messages.append({"role": "user", "content": follow})

            if not ok:
                continue
            full = tok.apply_chat_template(messages, tokenize=False)
            if len(tok(full).input_ids) > cfg["max_sequence_tokens"]:
                continue
            h = _hash(messages)
            if cfg["duplicate_check"] and h in seen:
                continue
            seen.add(h)
            f.write(json.dumps({
                "id": f"c{i:06d}_{h}",
                "messages": messages,
                "source": "tinyllama_generated",
                "teacher_model": "tinyllama-1.1b-chat-q4_k_m",
                "seed": cfg["seed"],
            }, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % cfg["checkpoint_every"] == 0:
                ckpt_path.write_text(json.dumps({"next": i + 1, "seen": list(seen)}))
                print(f"[gen] {i+1}/{cfg['num_conversations']}")

    ckpt_path.write_text(json.dumps({"next": cfg["num_conversations"], "seen": list(seen)}))
    print(f"[gen] done -> {out_path}")