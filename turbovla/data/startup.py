"""Training-only emphasis on real demonstration starts; no forced search actions."""
from itertools import islice

import numpy as np

from turbovla.data.stages import PHASES


def startup_frames(seconds):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("startup seconds must be finite and positive")
    return max(1, int(np.ceil(seconds * 30)))


def startup_base_l1(prediction, target, mask, frame_index, limit):
    """Extra supervised error on the first base action of early observations."""
    chosen = (frame_index < limit).to(prediction.dtype) * mask[:, 0].to(prediction.dtype)
    error = (prediction[:, 0, :3] - target[:, 0, :3]).abs().mean(-1)
    return (error * chosen).sum() / chosen.sum().clamp_min(1)


def startup_report(dataset, seconds):
    limit = startup_frames(seconds)
    rows = []
    for episode, labels in zip(dataset.episodes, dataset.labels):
        first = labels[:limit]
        moving = np.flatnonzero((first == 0) | (first == 1))
        rows.append({"episode_id": int(episode["episode_id"]), "frames": len(first),
                     "phase_counts": dict(zip(PHASES, np.bincount(first, minlength=4).tolist())),
                     "first_motion_seconds": float(moving[0] / 30) if len(moving) else None})
    return {"seconds": seconds, "episodes": rows}


class StartupSampler:
    """Mix the existing sampler with episode-uniform early-phase-uniform draws.

    The early branch chooses an available phase from that episode's start.
    An episode beginning with manipulation/hold keeps those expert actions.
    Inference never receives episode time or an injected motion command.
    """
    def __init__(self, dataset, base_sampler, seconds=10, fraction=.25, seed=42, task_balanced=False):
        if not np.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError("startup fraction must be finite and in [0,1)")
        self.size, self.base_sampler = len(dataset), base_sampler
        self.fraction, self.seed, self.iteration = fraction, seed, 0
        self.pools = {}
        offset, limit = 0, startup_frames(seconds)
        for episode, labels in zip(dataset.episodes, dataset.labels):
            task = int(episode.get("task_id", -1)) if task_balanced else 0
            if task_balanced and task < 0:
                raise ValueError("Task metadata required")
            first = labels[:limit]
            if not len(first):
                raise ValueError("Empty episode")
            phases = [np.flatnonzero(first == phase) + offset for phase in np.unique(first)]
            self.pools.setdefault(task, []).append(phases)
            offset += len(labels)
        if not self.pools or offset != self.size:
            raise ValueError("Startup labels do not match dataset length")

    def __len__(self):
        return self.size

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.iteration, 917]))
        self.iteration += 1
        background = iter(self.base_sampler) if self.base_sampler is not None else iter(rng.permutation(self.size))
        tasks = sorted(self.pools)
        for start in range(0, self.size, 4096):
            count = min(4096, self.size - start)
            draws = np.asarray(list(islice(background, count)), dtype=np.int64)
            if len(draws) != count:
                raise ValueError("Base sampler length mismatch")
            for position in np.flatnonzero(rng.random(count) < self.fraction):
                episodes = self.pools[tasks[int(rng.integers(len(tasks)))]]
                phases = episodes[int(rng.integers(len(episodes)))]
                pool = phases[int(rng.integers(len(phases)))]
                draws[position] = pool[int(rng.integers(len(pool)))]
            yield from draws.tolist()


class StartupMetrics:
    """First-action raw base errors and motion recall on teacher observations."""
    def __init__(self):
        self.count = 0
        self.error = np.zeros(3)
        self.hold_error = np.zeros(3)
        self.confusion = np.zeros((2, 2), dtype=np.int64)

    def update(self, prediction, target):
        prediction, target = np.asarray(prediction), np.asarray(target)
        if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 23:
            raise ValueError("Expected aligned [B,23] first actions")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("Nonfinite startup actions")
        self.count += len(target)
        self.error += np.abs(prediction[:, :3] - target[:, :3]).sum(0)
        self.hold_error += np.abs(target[:, :3]).sum(0)
        def moving(action):
            return (np.linalg.norm(action[:, :2], axis=1) >= .01) | (np.abs(action[:, 2]) >= .02)
        truth, guessed = moving(target).astype(int), moving(prediction).astype(int)
        self.confusion += np.bincount(2 * truth + guessed, minlength=4).reshape(2, 2)

    def result(self):
        if not self.count:
            return {"samples": 0}
        tn, fp, fn, tp = self.confusion.ravel().tolist()
        return {"samples": self.count, "base_raw_l1_per_axis": (self.error / self.count).tolist(),
                "hold_base_raw_l1_per_axis": (self.hold_error / self.count).tolist(),
                "motion_recall": tp / (tp + fn) if tp + fn else None,
                "motion_precision": tp / (tp + fp) if tp + fp else None,
                "false_motion_rate": fp / (tn + fp) if tn + fp else None,
                "motion_confusion": self.confusion.tolist()}
