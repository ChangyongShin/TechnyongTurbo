"""Refresh the packaged sources' hashes, excluding local test/install artifacts."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    manifest = root / "PACKAGE_SHA256.json"
    paths = set(json.loads(manifest.read_text(encoding="utf-8")))
    for directory in ("turbovla/data", "turbovla/training", "turbovla/evaluation", "turbovla/models", "turbovla/behavior", "experiments/behavior_2026", "scripts/behavior_2026", "tests"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                paths.add(path.relative_to(root).as_posix())
    paths.update(("README.md", "APPLIED_BEHAVIOR_FIXES_KO.md", ".gitignore", "pyproject.toml",
                  "turbovla/models/vision_encoder.py"))
    hashes = {}
    for relative in sorted(paths):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path == manifest:
            raise ValueError(f"Invalid manifest path: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            hashes[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest.write_text(json.dumps(hashes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated SHA256 manifest: {len(hashes)} packaged source files")


if __name__ == "__main__":
    main()
