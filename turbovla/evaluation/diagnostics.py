"""Read-only deployment identity and bounded inference trace helpers."""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class TraceWriter:
    """Create new files in a dedicated run directory; never overwrite a trace."""

    def __init__(self, directory, metadata, limit=64):
        if limit < 1:
            raise ValueError("Trace limit must be positive")
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        self.limit, self.count, self.episode = limit, 0, 0
        (self.root / "server-metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    def reset(self):
        self.episode += 1

    def write(self, trace, *, instruction, task_id, camera_keys):
        if self.count >= self.limit:
            return
        import numpy as np

        values = {key: np.asarray(value) for key, value in trace.items()}
        values.update(instruction=np.asarray(instruction), task_id=np.asarray(task_id),
                      camera_keys=np.asarray(camera_keys), rollout=np.asarray(self.episode))
        if any(value.dtype.hasobject for value in values.values()):
            raise ValueError("Object dtype is not allowed in inference traces")
        with (self.root / f"request_{self.count:06d}.npz").open("xb") as stream:
            np.savez_compressed(stream, **values)
        self.count += 1
