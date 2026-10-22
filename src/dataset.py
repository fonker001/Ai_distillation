import json, random
from pathlib import Path
import torch
from torch.utils.data import Dataset


class TopKCacheDataset(Dataset):
    """Loads shards lazily; never holds more than one shard in RAM."""
    def __init__(self, cache_dir, split="train", val_fraction=0.05, seed=42):
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "metadata.json").read_text())
        if meta["version"] != 2:
            raise RuntimeError(f"Cache version {meta['version']} incompatible")
        shards = sorted(self.cache_dir.glob("shard_*.pt"))
        rng = random.Random(seed); rng.shuffle(shards)
        n_val = max(1, int(len(shards) * val_fraction)) if len(shards) > 1 else 0
        self.shards = shards[:n_val] if split == "val" else shards[n_val:] or shards[:1]
        self.index = []
        for si, sp in enumerate(self.shards):
            n = len(torch.load(sp, map_location="cpu"))
            self.index.extend((si, j) for j in range(n))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, j = self.index[i]
        return torch.load(self.shards[si], map_location="cpu")[j]


def collate(batch, pad_id=0):
    """Pad input_ids to max_len; pad targets/topk to max_len - 1.
    This is the critical fix: targets and topk are already length T-1."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    B = len(batch)
    K = batch[0]["topk_ids"].size(-1)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    targets   = torch.full((B, max_len - 1), -100, dtype=torch.long)
    topk_logits = torch.zeros((B, max_len - 1, K), dtype=torch.float32)
    topk_ids    = torch.zeros((B, max_len - 1, K), dtype=torch.long)

    for b, item in enumerate(batch):
        L = item["input_ids"].size(0)
        input_ids[b, :L] = item["input_ids"]
        targets[b, :L-1] = item["targets"]         # already length L-1
        topk_logits[b, :L-1] = item["topk_logits"]
        topk_ids[b, :L-1] = item["topk_ids"]

    return {"input_ids": input_ids, "targets": targets,
            "topk_logits": topk_logits, "topk_ids": topk_ids}