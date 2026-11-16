"""Observable motion-phase supervision. Never claims target visibility or success."""
import argparse
import json
from pathlib import Path

import numpy as np

from turbovla.data.dataset import BehaviorDataset, load_episode

PHASES = ("navigate", "turn", "manipulate", "hold")
PHASE_SPEC = {"version": 1, "priority": list(PHASES), "base_xy_command_threshold": .01,
              "base_yaw_command_threshold": .02, "joint_target_current_threshold": .01,
              "gripper_command_change_threshold": .01,
              "note": "Motion-derived proxy labels; not search/detection/contact/success annotations"}


def phase_labels(episode):
    action, state = np.asarray(episode["actions"]), np.asarray(episode["states"])
    joint_delta = np.maximum(np.abs(action[:, 3:7] - state[:, 53:57]).max(axis=1),
        np.maximum(np.abs(action[:, 7:14] - state[:, 3:10]).max(axis=1),
                   np.abs(action[:, 15:22] - state[:, 28:35]).max(axis=1)))
    grip_delta = np.max(np.abs(np.diff(action[:, [14, 22]], axis=0, prepend=action[:1, [14, 22]])), axis=1)
    labels = np.full(len(action), 3, dtype=np.int64)
    labels[(joint_delta >= .01) | (grip_delta >= .01)] = 2
    labels[np.abs(action[:, 2]) >= .02] = 1
    labels[np.linalg.norm(action[:, :2], axis=1) >= .01] = 0
    return labels


class PhaseDataset(BehaviorDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = [phase_labels(e) for e in self.episodes]

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        e = int(np.searchsorted(self.ends, index, side="right"))
        t = index - (int(self.ends[e-1]) if e else 0)
        sample["phase"] = int(self.labels[e][t])
        return sample


class PhaseBalancedSampler:
    """Uniform task (optional), then uniform available phase, then uniform frame."""
    def __init__(self, dataset, seed=42, task_balanced=False):
        self.seed, self.iteration, self.size = seed, 0, len(dataset)
        self.buckets = {}
        offset = 0
        for episode, labels in zip(dataset.episodes, dataset.labels):
            task = int(episode.get("task_id", -1)) if task_balanced else 0
            if task_balanced and task < 0:
                raise ValueError("Task metadata required for joint task/phase balancing")
            for phase in np.unique(labels):
                self.buckets.setdefault(task, {}).setdefault(int(phase), []).append(np.flatnonzero(labels == phase) + offset)
            offset += len(labels)
        self.buckets = {task: {phase: np.concatenate(parts) for phase, parts in phases.items()}
                        for task, phases in self.buckets.items()}

    def __len__(self):
        return self.size

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.iteration]))
        self.iteration += 1
        tasks = sorted(self.buckets)
        for start in range(0, self.size, 4096):
            task_draws = rng.integers(len(tasks), size=min(4096, self.size-start))
            result = np.empty(len(task_draws), dtype=np.int64)
            for i, task in enumerate(tasks):
                slots = np.flatnonzero(task_draws == i)
                phases = sorted(self.buckets[task])
                choices = rng.integers(len(phases), size=len(slots))
                for j, phase in enumerate(phases):
                    positions = slots[choices == j]
                    pool = self.buckets[task][phase]
                    result[positions] = pool[rng.integers(len(pool), size=len(positions))]
            yield from result.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--episodes", required=True, nargs="+", type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--startup-seconds", type=float, default=10.)
    args = parser.parse_args()
    from turbovla.data.startup import startup_frames
    early_limit = startup_frames(args.startup_seconds)
    if args.out.exists():
        raise FileExistsError("Use a new phase report path")
    rows = []
    for eid in args.episodes:
        episode = load_episode(args.cache / f"episode_{eid:06d}")
        labels = phase_labels(episode)
        early = labels[:early_limit]
        moving = np.flatnonzero((early == 0) | (early == 1))
        rows.append({"episode_id": eid, "frames": len(labels),
                     "phase_counts": dict(zip(PHASES, np.bincount(labels, minlength=4).tolist())),
                     "startup_seconds": args.startup_seconds,
                     "startup_phase_counts": dict(zip(PHASES, np.bincount(early, minlength=4).tolist())),
                     "first_startup_motion_seconds": float(moving[0] / 30) if len(moving) else None,
                     "transitions": np.flatnonzero(np.diff(labels) != 0).tolist()})
        for key in ("rgb", "states", "actions", "depth"):
            array = episode.get(key)
            if isinstance(array, np.memmap):
                array._mmap.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"spec": PHASE_SPEC, "episodes": rows}, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
