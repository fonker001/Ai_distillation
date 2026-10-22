"""Prune TinyLlama and save a small student to disk. Teacher loaded once."""
import argparse, yaml, torch
from pathlib import Path
from src.teacher import load_teacher
from src.student import build_pruned_student, parameter_report

ap = argparse.ArgumentParser()
ap.add_argument("--num_layers", type=int, default=6)
ap.add_argument("--out", default="models/student_init")
args = ap.parse_args()

teacher, tok = load_teacher("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
student = build_pruned_student(teacher, args.num_layers)
total, trainable = parameter_report(student)
print(f"[init] pruned to {args.num_layers} layers: "
      f"total={total:,} trainable={trainable:,}")
assert total < 500_000_000, f"student too large: {total:,}"

Path(args.out).mkdir(parents=True, exist_ok=True)
student.save_pretrained(args.out)
tok.save_pretrained(args.out)
del teacher, student; import gc; gc.collect()
print(f"[init] saved -> {args.out}")