"""Shared train/inference contract. No simulator or Torch dependency."""
from dataclasses import dataclass

import numpy as np

CAMERAS = ("zed_link_camera_0", "left_realsense_link_camera_0", "right_realsense_link_camera_0")
ACTION_DIM = 23
STATE_DIM = 61


def finite_array(value, shape, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name}: expected finite {shape}, got {result.shape}")
    return result


def depth_metres(value, encoding):
    """Encoding is explicit: never infer it from channel count or value range."""
    depth = np.asarray(value, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError("depth must be a finite HxW array")
    if encoding == "log12":
        if np.any((depth < 0) | (depth > 4095)):
            raise ValueError("log12 codes must be in [0, 4095]")
        depth = np.exp(depth / 4095.0 * np.log(13.5 / 3.51) + np.log(3.51)) - 3.5
    elif encoding == "mm":
        depth = depth / 1000.0
    elif encoding != "m":
        raise ValueError("depth encoding must be log12, mm or m")
    if np.any(depth < 0):
        raise ValueError("negative depth")
    return np.clip(depth, 0.01, 10.0)


def preprocess_views(rgb, *, size, mean, std, depth=None, depth_encoding=None):
    """HWC uint8 RGB/RGBA; optional separate depth. Returns VCHW float32.

    RGB normalization parameters must come from the chosen DINOv3 processor.
    Depth is nearest-resized and scaled to [0.001, 1]; it is not RGB-normalized.
    """
    from PIL import Image

    if len(rgb) != len(CAMERAS) or size < 16 or size % 16:
        raise ValueError("expected three views and a positive size divisible by 16")
    mean = finite_array(mean, (3,), "RGB mean")[:, None, None]
    std = finite_array(std, (3,), "RGB std")[:, None, None]
    if np.any(std <= 0):
        raise ValueError("RGB std must be positive")
    if depth is not None and (len(depth) != 3 or depth_encoding is None):
        raise ValueError("three separate depth images and an explicit encoding are required")
    views = []
    for i, value in enumerate(rgb):
        arr = np.asarray(value)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            raise ValueError("RGB must be HWC uint8, with 3 channels or RGBA")
        # The fourth channel is alpha, NEVER depth.
        image = Image.fromarray(arr[..., :3]).resize((size, size), Image.Resampling.BICUBIC)
        channels = (np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0 - mean) / std
        if depth is not None:
            d = depth_metres(depth[i], depth_encoding)
            if d.shape != arr.shape[:2]:
                raise ValueError("RGB and depth must have matching original spatial dimensions")
            d = np.asarray(Image.fromarray(d).resize((size, size), Image.Resampling.NEAREST))
            channels = np.concatenate([channels, d[None] / 10.0], axis=0)
        views.append(channels)
    return np.stack(views).astype(np.float32)


@dataclass
class Normalizer:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray

    def __post_init__(self):
        for name, dim in (("state_mean", 61), ("state_std", 61), ("action_min", 23), ("action_max", 23)):
            setattr(self, name, finite_array(getattr(self, name), (dim,), name))
        if np.any(self.state_std <= 0) or np.any(self.action_max < self.action_min):
            raise ValueError("invalid normalization ranges")

    @classmethod
    def fit(cls, states, actions):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if states.ndim != 2 or actions.ndim != 2 or len(states) == 0 or len(states) != len(actions):
            raise ValueError("nonempty aligned state/action matrices required")
        finite_array(states, (len(states), 61), "states")
        finite_array(actions, (len(actions), 23), "actions")
        return cls(states.mean(0), np.maximum(states.std(0), 1e-6), actions.min(0), actions.max(0))

    def state(self, value):
        return (finite_array(value, (61,), "state") - self.state_mean) / self.state_std

    def action(self, value):
        value = np.asarray(value, dtype=np.float32)
        if value.shape[-1] != 23 or not np.isfinite(value).all():
            raise ValueError("expected finite actions ending in 23 dimensions")
        span = self.action_max - self.action_min
        # Constant dimensions have target zero and decode to the training constant.
        return np.where(span > 1e-6, np.clip(2 * (value - self.action_min) / np.maximum(span, 1e-6) - 1, -1, 1), 0)

    def inverse_action(self, value):
        value = np.asarray(value, dtype=np.float32)
        if value.shape[-1] != 23 or not np.isfinite(value).all():
            raise ValueError("expected finite actions ending in 23 dimensions")
        return self.action_min + (np.clip(value, -1, 1) + 1) * 0.5 * (self.action_max - self.action_min)

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in self.__dataclass_fields__}


def action_chunk(actions, index, horizon):
    if horizon < 1 or not 0 <= index < len(actions):
        raise ValueError("invalid chunk index/horizon")
    valid = min(horizon, len(actions) - index)
    chunk = np.asarray(actions[index:index + valid], dtype=np.float32)
    finite_array(chunk, (valid, 23), "chunk")
    chunk = np.concatenate([chunk, np.repeat(chunk[-1:], horizon - valid, axis=0)])
    return chunk, (np.arange(horizon) < valid).astype(np.float32)
