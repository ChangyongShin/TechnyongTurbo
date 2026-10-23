"""Small-subset dataset; keep complete episodes in disjoint train/validation lists."""
from pathlib import Path

import numpy as np

from turbovla.data.preprocessing import CAMERAS, Normalizer, action_chunk, finite_array, preprocess_views


def load_episode(path):
    path = Path(path)
    if path.is_dir():
        import json
        episode = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        for key in ("rgb", "states", "actions"):
            episode[key] = np.load(path / f"{key}.npy", mmap_mode="r", allow_pickle=False)
        if episode.get("modality") == "rgbd":
            episode["depth"] = np.load(path / "depth.npy", mmap_mode="r", allow_pickle=False)
        episode["cameras"] = np.asarray(episode["cameras"])
        episode["instruction"] = np.asarray(episode["instruction"])
    else:
        with np.load(path, allow_pickle=False) as data:
            episode = {key: data[key] for key in data.files}
    count = len(episode["states"])
    finite_array(episode["states"], (count, 61), "states")
    finite_array(episode["actions"], (count, 23), "actions")
    rgb = episode["rgb"]
    if count == 0 or rgb.ndim != 5 or rgb.shape[:2] != (count, 3) or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected nonempty RGB [T,3,H,W,3] uint8")
    if tuple(episode["cameras"].tolist()) != CAMERAS or int(episode["fps"]) != 30:
        raise ValueError("Unexpected camera order or fps")
    instruction = str(episode["instruction"].item())
    if not instruction.strip():
        raise ValueError("Instruction cannot be empty")
    episode["instruction"] = instruction
    modality = str(np.asarray(episode.get("modality", "rgb")).item())
    if modality not in ("rgb", "rgbd"):
        raise ValueError("Unknown episode modality")
    episode["modality"] = modality
    if modality == "rgbd":
        depth = episode["depth"]
        encoding = str(np.asarray(episode["depth_encoding"]).item())
        if encoding != "m" or depth.dtype != np.float32 or depth.shape != rgb.shape[:4]:
            raise ValueError("Expected aligned float32 metric depth [T,3,H,W]")
        # Validate in chunks to avoid a dataset-sized boolean allocation.
        for start in range(0, count, 64):
            block = depth[start:start + 64]
            if not np.isfinite(block).all() or np.any((block < 0.01 - 1e-6) | (block > 10.0 + 1e-6)):
                raise ValueError("Depth cache contains invalid metric values")
    elif "depth" in episode:
        raise ValueError("Depth array present without RGB-D modality metadata")
    return episode


class BehaviorDataset:
    def __init__(self, paths, normalizer, preprocessing, horizon=12, modality="rgb"):
        self.episodes = [load_episode(Path(p)) for p in paths]
        if any(episode["modality"] != modality for episode in self.episodes):
            raise ValueError(f"All caches must match the requested {modality} modality")
        self.ends = np.cumsum([len(data["states"]) for data in self.episodes])
        if not len(self.ends) or self.ends[-1] == 0:
            raise ValueError("Empty dataset")
        self.normalizer = normalizer
        self.preprocessing = preprocessing
        self.horizon = horizon
        self.modality = modality

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        e = int(np.searchsorted(self.ends, index, side="right"))
        t = index - (int(self.ends[e - 1]) if e else 0)
        data = self.episodes[e]
        chunk, mask = action_chunk(data["actions"], t, self.horizon)
        depth = {"depth": data["depth"][t], "depth_encoding": "m"} if self.modality == "rgbd" else {}
        return {
            "images": preprocess_views(data["rgb"][t], **self.preprocessing, **depth),
            "state": self.normalizer.state(data["states"][t]),
            "actions": self.normalizer.action(chunk), "mask": mask,
            "instruction": data["instruction"],
        }


def training_normalizer(paths):
    if not paths:
        raise ValueError("No training episodes")
    count, mean, m2 = 0, np.zeros(61, dtype=np.float64), np.zeros(61, dtype=np.float64)
    low, high = np.full(23, np.inf), np.full(23, -np.inf)
    for path in paths:
        episode = load_episode(path)
        states = np.asarray(episode["states"], dtype=np.float64)
        n = len(states)
        batch_mean = states.mean(0)
        delta = batch_mean - mean
        m2 += ((states - batch_mean) ** 2).sum(0) + delta ** 2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
        low = np.minimum(low, episode["actions"].min(0))
        high = np.maximum(high, episode["actions"].max(0))
    return Normalizer(mean, np.maximum(np.sqrt(m2 / count), 1e-6), low, high)
