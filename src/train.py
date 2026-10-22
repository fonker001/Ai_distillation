"""Student distillation training. Teacher is never loaded here."""
import gc, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from .dataset import TopKCacheDataset, collate
from .loss import distillation_loss
from .student import (build_pruned_student, parameter_report,
                      estimate_training_memory_gb)


def _sample_prompts(tokenizer, prompts, max_new=64):
    """Helper used only for periodic sanity generation (during train)."""
    from transformers import LlamaForCausalLM  # noqa
    return prompts  # not used in training loop; kept for eval hook


def train(cfg):
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)

    # --- Load student from on-disk pruned init (created in step 3) ---
    from transformers import LlamaForCausalLM
    student = LlamaForCausalLM.from_pretrained(cfg["student_init_path"],
                                               torch_dtype=torch.float32)
    if cfg["freeze_embeddings"]:
        for p in student.get_input_embeddings().parameters():
            p.requires_grad_(False)
    if cfg["gradient_checkpointing"]:
        student.gradient_checkpointing_enable()

    total, trainable = parameter_report(student)
    est_gb = estimate_training_memory_gb(
        trainable_params=trainable, frozen_params=total - trainable,
        batch=cfg["batch_size"], seq=384,
        hidden=student.config.hidden_size, layers=student.config.num_hidden_layers,
        grad_ckpt=cfg["gradient_checkpointing"],
    )
    print(f"[train] params total={total:,} trainable={trainable:,}")
    print(f"[train] conservative peak-memory estimate: {est_gb:.2f} GB")
    import psutil
    avail = psutil.virtual_memory().available / 1e9
    if avail < est_gb + 1.5:
        raise MemoryError(
            f"Need ~{est_gb:.1f} GB + 1.5 GB margin, only {avail:.1f} GB free. "
            "Lower num_layers in configs/train.yaml (try 4) or use more RAM."
        )

    # --- Data ---
    train_ds = TopKCacheDataset(cfg["cache_dir"], "train", cfg["val_fraction"])
    val_ds   = TopKCacheDataset(cfg["cache_dir"], "val",   cfg["val_fraction"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, collate_fn=collate, num_workers=0)

    # --- Optimizer / scheduler ---
    # SGD+momentum: 12 bytes/param vs AdamW's 16. On an 8 GB machine
    # with a ~200M trainable student, that saves ~800 MB — enough to fit.
    # For a smoke run this is quality-equivalent to AdamW.
    params = [p for p in student.parameters() if p.requires_grad]
    optim = torch.optim.SGD(
        params,
        lr=cfg["learning_rate"] * 10,   # SGD needs ~10× larger LR than AdamW
        momentum=0.9,
        weight_decay=cfg["weight_decay"],
    )

    total_steps = max(1, (len(train_loader) * cfg["epochs"]) // cfg["gradient_accumulation_steps"])
    warmup = int(total_steps * cfg["warmup_ratio"])
    sched = get_linear_schedule_with_warmup(optim, warmup, total_steps)

    start_epoch, gstep = 0, 0
    if cfg["resume_from"]:
        ck = torch.load(cfg["resume_from"], map_location="cpu")
        student.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        start_epoch, gstep = ck["epoch"], ck["step"]
        print(f"[train] resumed at epoch {start_epoch}, step {gstep}")

    history_path = out / "history.jsonl"
    history_path.write_text("") if start_epoch == 0 else None
    f_hist = open(history_path, "a", encoding="utf-8")

    student.train()
    for epoch in range(start_epoch, cfg["epochs"]):
        ep_loss, t0 = 0.0, time.time()
        optim.zero_grad()
        for step, batch in enumerate(train_loader):
            # ---- Forward ----
            logits = student(input_ids=batch["input_ids"], use_cache=False).logits  # (B,T,V)
            pred = logits[:, :-1, :].contiguous()                # (B,T-1,V)
            tgt  = batch["targets"].contiguous()                 # (B,T-1)  <-- already aligned
            tk_logits = batch["topk_logits"].contiguous()        # (B,T-1,K)
            tk_ids    = batch["topk_ids"].contiguous()           # (B,T-1,K)

            loss = distillation_loss(
                pred, tgt, tk_logits, tk_ids,
                temperature=cfg["temperature"], kd_alpha=cfg["kd_alpha"],
            )
            ep_loss += loss.item()
            (loss / cfg["gradient_accumulation_steps"]).backward()

            if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                optim.step(); sched.step(); optim.zero_grad()
                gstep += 1
                if gstep % cfg["log_every"] == 0:
                    rec = {"epoch": epoch + 1, "step": gstep,
                           "train_loss": loss.item(),
                           "elapsed_s": round(time.time() - t0, 1)}
                    print(f"[train] epoch {epoch+1} step {gstep} "
                          f"loss {loss.item():.4f}")
                    f_hist.write(json.dumps(rec) + "\n"); f_hist.flush()
                if gstep % cfg["save_every"] == 0:
                    ck = out / f"ckpt_step_{gstep}.pt"
                    torch.save({"model": student.state_dict(),
                                "optim": optim.state_dict(),
                                "sched": sched.state_dict(),
                                "epoch": epoch, "step": gstep}, ck)
                    print(f"[train] saved {ck}")

        # ---- Validation ----
        student.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                logits = student(input_ids=batch["input_ids"], use_cache=False).logits
                loss = distillation_loss(
                    logits[:, :-1, :], batch["targets"],
                    batch["topk_logits"], batch["topk_ids"],
                    temperature=cfg["temperature"], kd_alpha=cfg["kd_alpha"])
                val_loss += loss.item(); n_val += 1
        avg_train = ep_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(n_val, 1)
        rec = {"epoch": epoch + 1, "train_epoch_loss": avg_train, "val_loss": avg_val}
        print(f"[train] epoch {epoch+1} done: train {avg_train:.4f} val {avg_val:.4f}")
        f_hist.write(json.dumps(rec) + "\n"); f_hist.flush()
        student.train()
        gc.collect()

    student.save_pretrained(out / "final")
    # Save tokenizer alongside (copy from init dir)
    from shutil import copyfile
    for f in (Path(cfg["student_init_path"]).glob("*token*")):
        copyfile(f, out / "final" / f.name)
    (out / "TRAINING_COMPLETE").write_text(
        json.dumps({"epochs": cfg["epochs"], "total_steps": gstep}))
    f_hist.close()
    print(f"[train] final student -> {out / 'final'}")