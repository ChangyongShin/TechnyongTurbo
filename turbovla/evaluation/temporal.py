"""Align overlapping chunks to the current control step before averaging joints."""
from collections import deque

import numpy as np


class JointTemporalEnsemble:
    def __init__(self, horizon, window=1, decay=1.0):
        if not 1 <= window <= horizon or not np.isfinite(decay) or decay < 0:
            raise ValueError("ensemble window must be in [1, horizon]; decay must be finite and nonnegative")
        self.window, self.decay = window, decay
        self.chunks = deque(maxlen=window)

    def reset(self):
        self.chunks.clear()

    def update(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 23 or len(chunk) < self.window or not np.isfinite(chunk).all():
            raise ValueError("Expected a finite [horizon,23] action chunk")
        self.chunks.append(chunk.copy())
        # A chunk predicted age steps ago contributes its age-th future action,
        # not its first action (which belongs to a past control step).
        aligned = np.stack([old[age] for age, old in enumerate(reversed(self.chunks))])
        weights = np.exp(-self.decay * np.arange(len(aligned)))
        result = chunk[0].copy()
        # Preserve current base velocity and gripper switches. Average only
        # trunk and arm joint targets in the existing controller input space.
        for start, end in ((3, 14), (15, 22)):
            result[start:end] = np.average(aligned[:, start:end], axis=0, weights=weights)
        return result
