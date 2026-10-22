import argparse, yaml
from src.generate import generate_all
ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/generate.yaml")
cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8")); generate_all(cfg)