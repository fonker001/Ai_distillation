import argparse, yaml
from src.evaluate import evaluate
ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/eval.yaml")
cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8")); evaluate(cfg)