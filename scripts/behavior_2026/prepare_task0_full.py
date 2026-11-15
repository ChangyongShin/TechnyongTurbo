"""Reuse complete RGB caches and prepare the missing task-0 baseline split."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from turbovla.data.dataset import load_episode
from turbovla.data.preprocessing import CAMERAS


def baseline_split(config):
    train, val = config["train_episode_ids"], config["validation_episode_ids"]
    if (len(train) != 180 or len(val) != 20 or len(set(train + val)) != 200 or
            set(train) & set(val) or set(train + val) != set(range(200))):
        raise ValueError("Expected the existing task0 180/20 split covering episodes 0..199")
    return sorted(train), sorted(val)


def check_cache(path, eid, length, revision):
    if not path.exists():
        return False
    if not (path / "metadata.json").exists():
        raise ValueError(f"Incomplete cache preserved; repair before retrying: {path}")
    episode = load_episode(path)
    try:
        if (int(episode["episode_id"]) != eid or int(episode["task_id"]) != 0 or
                episode["modality"] != "rgb" or len(episode["states"]) != length or
                tuple(episode["rgb"].shape[2:4]) != (224, 224) or
                (episode.get("source") or {}).get("revision") != revision):
            raise ValueError(f"Cache identity/schema/revision mismatch: {path}")
    finally:
        # Release mapped files even when identity validation fails.
        for name in ("rgb", "states", "actions"):
            if hasattr(episode[name], "_mmap"):
                episode[name]._mmap.close()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--baseline-config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.baseline_config.read_text(encoding="utf-8"))
    train, val = baseline_split(config)
    source = json.loads((args.root / "turbovla_source.json").read_text(encoding="utf-8"))
    baseline_revisions = {s["revision"] for s in config.get("data_sources", []) if s and "revision" in s}
    if not baseline_revisions or baseline_revisions != {source["revision"]}:
        raise ValueError("Source revision does not match baseline provenance")
    import pyarrow.dataset as ds
    rows = ds.dataset(str(args.root / "meta/episodes/chunk-000"), format="parquet").to_table().to_pylist()
    if len(rows) != 200:
        raise ValueError("Expected exactly 200 task0 episode metadata rows")
    rows = {int(r["episode_index"]): r for r in rows}
    if set(rows) != set(train + val):
        raise ValueError("Missing or unexpected task0 episode metadata")
    args.cache.mkdir(parents=True, exist_ok=True)
    missing = [eid for eid in sorted(rows) if not check_cache(
        args.cache / f"episode_{eid:06d}", eid, int(rows[eid]["length"]), source["revision"])]
    info = json.loads((args.root / "meta/info.json").read_text(encoding="utf-8"))
    required_videos = set()
    for eid in missing:
        for camera in CAMERAS:
            key = f"observation.rgb.{camera}"
            prefix = f"videos/{key}"
            required_videos.add(args.root / info["video_path"].format(video_key=key,
                chunk_index=int(rows[eid][f"{prefix}/chunk_index"]),
                file_index=int(rows[eid][f"{prefix}/file_index"])))
    absent = sorted(str(p) for p in required_videos if not p.is_file())
    if absent:
        raise FileNotFoundError("Original video shards missing; no training started:\n" + "\n".join(absent))
    if missing and not list((args.root / "data/chunk-000").glob("*.parquet")):
        raise FileNotFoundError("Task0 parquet data is missing")
    required_bytes = sum(int(rows[eid]["length"]) for eid in missing) * 451584
    print(json.dumps({"train_episodes": len(train), "validation_episodes": len(val),
        "reused_episodes": 200-len(missing), "to_prepare": len(missing),
        "additional_rgb_cache_gib": required_bytes / 1024**3,
        "train_frames": sum(int(rows[eid]["length"]) for eid in train),
        "validation_frames": sum(int(rows[eid]["length"]) for eid in val),
        "revision": source["revision"]}), flush=True)
    if shutil.disk_usage(args.cache).free < required_bytes * 1.05:
        raise OSError("Insufficient reported free disk space for remaining RGB caches plus headroom")
    if missing:
        subprocess.run([sys.executable, "-u", "-m", "turbovla.data.prepare", "--root", str(args.root),
            "--episodes", *map(str, missing), "--out", str(args.cache), "--image-size", "224", "--format", "mmap"],
            cwd=ROOT, check=True)
    for eid in sorted(rows):
        if not check_cache(args.cache / f"episode_{eid:06d}", eid, int(rows[eid]["length"]), source["revision"]):
            raise FileNotFoundError(f"Episode {eid} is not complete; no training started")
    (args.cache / "task0-full-ready.json").write_text(json.dumps({"train": train, "validation": val,
        "revision": source["revision"], "image_size": 224}, indent=2), encoding="utf-8")
    print("READY: all 200 task0 caches checked; original 180/20 split preserved", flush=True)


if __name__ == "__main__":
    main()
