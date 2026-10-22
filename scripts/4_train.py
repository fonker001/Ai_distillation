import argparse, yaml
from src.train import train
ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/train.yaml")
cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8")); train(cfg)