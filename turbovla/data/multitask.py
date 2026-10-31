"""Task-uniform, frame-uniform sampling and fixed-scale action evaluation."""
import numpy as np


class TaskBalancedSampler:
    """Sample a task uniformly, then a frame uniformly within that task.

    Draws with replacement; a sampler pass is not a dataset epoch. Task balance
    is in expectation, rather than an exact task quota in each mini-batch.
    """

    def __init__(self, episodes, seed=42):
        self.seed = seed
        self.iteration = 0
        self.ranges = {}
        offset = 0
        for episode in episodes:
            task = int(episode.get("task_id", -1))
            if task < 0:
                raise ValueError("Task-balanced sampling requires cached task_id metadata")
            length = len(episode["states"])
            if length < 1:
                raise ValueError("Empty episode")
            self.ranges.setdefault(task, []).append((offset, length))
            offset += length
        if not offset:
            raise ValueError("Empty training dataset")
        self.size = offset
        self.tasks = sorted(self.ranges)

    def __len__(self):
        return self.size

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.iteration]))
        self.iteration += 1
        for start in range(0, self.size, 4096):
            choices = rng.integers(len(self.tasks), size=min(4096, self.size - start))
            indices = np.empty(len(choices), dtype=np.int64)
            for i, task in enumerate(self.tasks):
                positions = np.flatnonzero(choices == i)
                ranges = self.ranges[task]
                ends = np.cumsum([length for _, length in ranges])
                draws = rng.integers(int(ends[-1]), size=len(positions))
                episode = np.searchsorted(ends, draws, side="right")
                begins = np.r_[0, ends[:-1]]
                offsets = np.asarray([offset for offset, _ in ranges])
                indices[positions] = offsets[episode] + draws - begins[episode]
            yield from indices.tolist()


def reference_action_l1(prediction, raw_target, mask, current, reference):
    """Per-sample masked errors in a fixed reference normalization.

    Predictions use the unbounded affine transform, matching the original
    trainer's unclipped normalized prediction loss. Targets are clipped just
    like Normalizer.action. Reference-constant axes contribute zero; physical
    errors on these axes must be checked separately.
    """
    import torch

    def vector(values):
        return torch.as_tensor(values, device=prediction.device, dtype=prediction.dtype)

    raw_prediction = vector(current.action_min) + (prediction + 1) * .5 * vector(current.action_max - current.action_min)
    low, span = vector(reference.action_min), vector(reference.action_max - reference.action_min)
    active = span > 1e-6
    pred = torch.where(active, 2 * (raw_prediction - low) / span.clamp_min(1e-6) - 1, 0)
    target = torch.where(active, (2 * (raw_target - low) / span.clamp_min(1e-6) - 1).clamp(-1, 1), 0)
    return ((pred - target).abs() * mask).sum(dim=1)
