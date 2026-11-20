"""Optional TRAIN-only interventions for departure failures; no simulator dependency.

Velocity masking is input regularization, not a physically valid recovery demo.
Transitions are command-derived events, not radio/search/success annotations.
"""
import numpy as np


def mask_base_velocity(state, probability, rng=None):
    """Mask all three RAW base velocities together to physical zero before scaling."""
    if not 0 <= probability <= 1:
        raise ValueError("Velocity mask probability must be in [0, 1]")
    value = np.array(state, dtype=np.float32, copy=True)
    if value.shape != (61,) or not np.isfinite(value).all():
        raise ValueError("Expected a finite 61-dimensional raw state")
    rng = np.random if rng is None else rng
    if probability and rng.random() < probability:
        value[:3] = 0
    return value


def mild_photometric(rgb, rng=None):
    """Shared mild exposure/contrast/saturation; preserve all pixel coordinates.

    No mirror, crop, camera permutation, or modification of depth/action targets.
    Half the draws are unchanged. Identical views receive identical transforms.
    """
    value = np.asarray(rgb)
    if value.dtype != np.uint8 or value.ndim != 4 or value.shape[0] != 3 or value.shape[-1] != 3:
        raise ValueError("Expected uint8 RGB [3,H,W,3]")
    rng = np.random if rng is None else rng
    if rng.random() >= .5:
        return value.copy()
    brightness, contrast, saturation = rng.uniform(.85, 1.15), rng.uniform(.85, 1.15), rng.uniform(.9, 1.1)
    work = value.astype(np.float32) / 255
    gray = (work * np.array([.299, .587, .114], dtype=np.float32)).sum(-1, keepdims=True)
    work = gray + saturation * (work - gray)
    work = ((work - .5) * contrast + .5) * brightness
    return np.rint(np.clip(work, 0, 1) * 255).astype(np.uint8)


def departure_events(actions, *, quiet_frames=15, window=15):
    """Find rest-to-command onsets, including slow ramps that soon cross .05.

    Rest: XY norm <= .01 and abs(yaw) <= .02 for quiet_frames (30 Hz).
    Onset: first non-rest command whose following window reaches XY > .05
    or abs(yaw) > .05. Initial moving frames have no preceding-rest evidence.
    Windows are inclusive, clipped within this episode, never across episodes.
    """
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 23 or not np.isfinite(actions).all():
        raise ValueError("Expected finite [T,23] actions")
    if quiet_frames < 1 or window < 1:
        raise ValueError("quiet_frames and window must be positive")
    xy = np.linalg.norm(actions[:, :2], axis=1)
    yaw = np.abs(actions[:, 2])
    rest = (xy <= .01) & (yaw <= .02)
    quiet = 0
    events = []
    for t in range(len(actions)):
        if rest[t]:
            quiet += 1
            continue
        if quiet >= quiet_frames:
            end = min(len(actions), t + window + 1)
            move, turn = bool((xy[t:end] > .05).any()), bool((yaw[t:end] > .05).any())
            if move or turn:
                events.append({"frame": t, "kind": "move_and_turn" if move and turn else "move" if move else "turn",
                               "start": max(0, t-window), "stop": min(len(actions), t+window+1)})
        quiet = 0
    return events


class TransitionMixtureSampler:
    """Replace a fraction of original draws with episode/event-balanced windows."""
    def __init__(self, dataset, base_sampler, fraction, seed=42, window=15, quiet_frames=15):
        if not 0 <= fraction <= 1:
            raise ValueError("Transition fraction must be in [0, 1]")
        self.size, self.base_sampler, self.fraction = len(dataset), base_sampler, fraction
        self.seed, self.iteration = seed, 0
        self.buckets, self.events = [], []
        offset = 0
        for episode in dataset.episodes:
            events = departure_events(episode["actions"], quiet_frames=quiet_frames, window=window)
            self.events.append(events)
            if events:
                self.buckets.append([(offset+event["start"], offset+event["stop"]) for event in events])
            offset += len(episode["actions"])
        if fraction and not self.buckets:
            raise ValueError("No departure events in TRAIN; no sampling fallback was applied")

    def __len__(self):
        return self.size

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.iteration, 743]))
        self.iteration += 1
        base = iter(self.base_sampler) if self.base_sampler is not None else iter(rng.permutation(self.size))
        for _ in range(self.size):
            index = next(base)
            if self.fraction and rng.random() < self.fraction:
                episode = self.buckets[int(rng.integers(len(self.buckets)))]
                low, high = episode[int(rng.integers(len(episode)))]
                index = int(rng.integers(low, high))
            yield int(index)
