"""Plan/download/prepare task 0,66,77,89,92 using a pinned dataset revision."""
import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

TASKS = (0, 66, 77, 89, 92)


def split_episodes(ids, seed=42):
    ids = sorted(ids)
    if len(ids) != len(set(ids)) or len(ids) < 2:
        raise ValueError("Expected at least two unique episodes per task")
    random.Random(seed).shuffle(ids)
    count = min(len(ids) - 1, max(1, math.ceil(len(ids) * .1)))
    return sorted(ids[count:]), sorted(ids[:count])


def validate_task0_split(split, ids):
    train, val = split["train"], split["validation"]
    if (len(set(train)) != len(train) or len(set(val)) != len(val) or
            set(train) & set(val) or set(train + val) != set(ids) or not train or not val):
        raise ValueError("Existing task0 split does not partition the downloaded metadata")
    return train, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["plan", "download", "prepare", "all"], default="plan")
    parser.add_argument("--root", type=Path, default=Path("/workspace/data/behavior-2026-rgb"))
    parser.add_argument("--cache", type=Path, default=Path("/workspace/data/behavior-2026-toggle-cache"))
    parser.add_argument("--task0-cache", type=Path, default=Path("/workspace/data/behavior-2026-task0-cache"))
    parser.add_argument("--task0-split", type=Path, default=Path("experiments/behavior_2026/task0_split.json"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/behavior_2026/toggle5_split.json"))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    import pyarrow.dataset as ds

    source = json.loads((args.root / "turbovla_source.json").read_text(encoding="utf-8"))
    old_split = json.loads(args.task0_split.read_text(encoding="utf-8"))
    if old_split["revision"] != source["revision"]:
        raise ValueError("Task0 split and dataset revisions differ")
    if args.phase in ("plan", "all"):
        snapshot_download(source["repo_id"], repo_type="dataset", revision=source["revision"],
                          local_dir=str(args.root), allow_patterns=[
                              f"meta/episodes/chunk-{task:03d}/*.parquet" for task in TASKS])
        manifest = {"revision": source["revision"], "seed": 42, "task_ids": list(TASKS),
                    "train_paths": [], "validation_paths": [], "tasks": {}}
        seen = set()
        for task in TASKS:
            rows = ds.dataset(str(args.root / f"meta/episodes/chunk-{task:03d}"), format="parquet").to_table(
                columns=["episode_index", "length"]).to_pylist()
            lengths = {int(row["episode_index"]): int(row["length"]) for row in rows}
            if len(lengths) != len(rows) or seen & set(lengths) or any(n < 1 for n in lengths.values()):
                raise ValueError("Invalid/duplicate episode metadata")
            seen.update(lengths)
            train, val = validate_task0_split(old_split, lengths) if task == 0 else split_episodes(lengths)
            cache = args.task0_cache if task == 0 else args.cache
            entry = {"train": train, "validation": val, "cache": str(cache.resolve()),
                     "train_frames": sum(lengths[i] for i in train),
                     "validation_frames": sum(lengths[i] for i in val),
                     "lengths": {str(k): v for k, v in lengths.items()}}
            manifest["tasks"][str(task)] = entry
            for key, ids in (("train_paths", train), ("validation_paths", val)):
                manifest[key].extend(str((cache / f"episode_{i:06d}").resolve()) for i in ids)
            print(json.dumps({"task": task, "train_episodes": len(train), "validation_episodes": len(val),
                              "train_frames": entry["train_frames"],
                              "rgb_cache_gib": sum(lengths.values()) * 451584 / 1024**3}), flush=True)
        manifest["equal_task0_exposure_steps"] = math.ceil(manifest["tasks"]["0"]["train_frames"] * len(TASKS) / 32)
        manifest["equal_compute_steps"] = 12048
        if args.manifest.exists():
            if json.loads(args.manifest.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Manifest changed; choose a new manifest filename")
        else:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest: {args.manifest}", flush=True)
        print("No training started. Run download and prepare phases next.", flush=True)
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest["revision"] != source["revision"]:
            raise ValueError("Manifest and dataset revisions differ")
    for task in TASKS:
        entry = manifest["tasks"][str(task)]
        ids = sorted(entry["train"] + entry["validation"])
        if args.phase in ("download", "all"):
            subprocess.run([sys.executable, "-m", "turbovla.data.download", "--root", str(args.root),
                            "--revision", manifest["revision"], "--task-id", str(task),
                            "--episodes", *map(str, ids)], check=True)
        if args.phase in ("prepare", "all"):
            cache = Path(entry["cache"])
            missing = []
            for eid in ids:
                dest = cache / f"episode_{eid:06d}"
                if not dest.exists():
                    missing.append(eid)
                    continue
                marker = dest / "metadata.json"
                if not marker.exists():
                    raise ValueError(f"Incomplete cache (preserved): {dest}; repair it before retrying")
                from turbovla.data.dataset import load_episode
                episode = load_episode(dest)
                if (int(episode["episode_id"]) != eid or int(episode["task_id"]) != task or
                        episode["modality"] != "rgb" or episode["rgb"].shape[2:4] != (224, 224) or
                        len(episode["states"]) != entry["lengths"][str(eid)] or
                        (episode.get("source") or {}).get("revision") != manifest["revision"]):
                    raise ValueError(f"Existing cache contract mismatch: {dest}")
                del episode
            if missing:
                subprocess.run([sys.executable, "-m", "turbovla.data.prepare", "--root", str(args.root),
                                "--out", str(cache), "--image-size", "224", "--format", "mmap",
                                "--episodes", *map(str, missing)], check=True)
            print(f"Task {task}: caches checked/prepared; {len(ids)} episodes", flush=True)


if __name__ == "__main__":
    main()
