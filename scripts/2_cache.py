import argparse, yaml
from src.cache import build_cache
ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/cache.yaml")
cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8")); build_cache(cfg)