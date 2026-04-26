import pathlib
import random
from bisect import bisect_right

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ChessDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = pathlib.Path(data_dir)
        self.shards = []
        self.cumulative_lengths = []
        running = 0

        feature_files = sorted(self.data_dir.glob("*_features.npy"))
        if not feature_files:
            fallback_dir = pathlib.Path.home() / "chess-nn-data" / "processed"
            feature_files = sorted(fallback_dir.glob("*_features.npy"))
            if feature_files:
                self.data_dir = fallback_dir
                print(f"Using fallback processed directory: {self.data_dir}")
        if not feature_files:
            raise FileNotFoundError(f"No *_features.npy files found in {self.data_dir}")

        for feat_path in feature_files:
            pol_path = pathlib.Path(str(feat_path).replace("_features.npy", "_policies.npy"))
            val_path = pathlib.Path(str(feat_path).replace("_features.npy", "_values.npy"))
            if not pol_path.exists() or not val_path.exists():
                raise FileNotFoundError(f"Missing policy/value pair for {feat_path.name}")

            # Load a temporary mmap just to read the length, then discard it.
            # The real mmap is opened lazily per worker so the dataset stays picklable.
            tmp_mmap = np.load(feat_path, mmap_mode="r")
            length = int(tmp_mmap.shape[0])
            del tmp_mmap

            policies = np.load(pol_path)
            values = np.load(val_path)

            if policies.shape[0] != length or values.shape[0] != length:
                raise ValueError(f"Mismatched shard lengths for {feat_path.name}")

            # Store path as str (picklable); mmap opened lazily in _get_features
            self.shards.append((str(feat_path), policies, values, length))
            running += length
            self.cumulative_lengths.append(running)

        # Populated lazily per worker process — never pickled with non-empty contents
        self._feature_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    def _get_features(self, shard_idx: int) -> np.ndarray:
        if shard_idx not in self._feature_cache:
            feat_path = self.shards[shard_idx][0]
            self._feature_cache[shard_idx] = np.load(feat_path, mmap_mode="r")
        return self._feature_cache[shard_idx]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx}")

        shard_idx = bisect_right(self.cumulative_lengths, idx)
        prev_cum = 0 if shard_idx == 0 else self.cumulative_lengths[shard_idx - 1]
        local_idx = idx - prev_cum

        _, policies, values, _ = self.shards[shard_idx]
        features_mmap = self._get_features(shard_idx)
        x = torch.tensor(np.asarray(features_mmap[local_idx], dtype=np.float32), dtype=torch.float32)
        p = torch.tensor(int(policies[local_idx]), dtype=torch.long)
        v = torch.tensor(float(values[local_idx]), dtype=torch.float32)
        return x, p, v


if __name__ == "__main__":
    processed_dir = pathlib.Path(__file__).resolve().parent / "processed"
    ds = ChessDataset(str(processed_dir))
    print(f"Total dataset length: {len(ds)}")

    for _ in range(5):
        i = random.randrange(len(ds))
        x, p, v = ds[i]
        print(f"Sample idx={i} | x={tuple(x.shape)} p={p.item()} v={v.item()}")

    loader = DataLoader(ds, batch_size=32, shuffle=True)
    xb, pb, vb = next(iter(loader))
    print(f"Batch features shape: {tuple(xb.shape)}")
    print(f"Batch policies shape: {tuple(pb.shape)}")
    print(f"Batch values shape: {tuple(vb.shape)}")
    assert tuple(xb.shape) == (32, 18, 8, 8)
    assert tuple(pb.shape) == (32,)
    assert tuple(vb.shape) == (32,)
    print("dataset.py: All tests passed")
