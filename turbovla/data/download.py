"""Download selected complete RGB/RGB-D episodes at a reproducible HF revision."""
import argparse
import json
from pathlib import Path

from turbovla.data.preprocessing import CAMERAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--rgbd", action="store_true")
    args = parser.parse_args()
    import pyarrow.dataset as ds
    from huggingface_hub import HfApi, snapshot_download

    if not 0 <= args.task_id < 100 or len(set(args.episodes)) != len(args.episodes):
        raise ValueError("Expected task-id 0..99 and unique episode IDs")
    repo_id = "behavior-1k/2026-challenge-demos"
    revision = HfApi().dataset_info(repo_id, revision=args.revision).sha
    provenance_path = args.root / "turbovla_source.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if previous["revision"] != revision:
            raise ValueError("Dataset revision changed; use a separate root to prevent mixed shards")
    elif args.root.exists() and any(args.root.iterdir()):
        raise ValueError("Use a new dataset root; existing files have no recorded revision")
    provenance = {"repo_id": repo_id, "revision": revision}
    args.root.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    chunk = f"chunk-{args.task_id:03d}"
    patterns = ["meta/info.json", "meta/tasks.parquet", "meta/stats.json", f"meta/episodes/{chunk}/*.parquet"]
    snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=str(args.root), allow_patterns=patterns)
    info = json.loads((args.root / "meta/info.json").read_text(encoding="utf-8"))
    episodes = ds.dataset(str(args.root / "meta/episodes" / chunk), format="parquet").to_table(
        filter=ds.field("episode_index").isin(args.episodes)).to_pylist()
    if {int(e["episode_index"]) for e in episodes} != set(args.episodes):
        raise ValueError("Selected episodes are missing from this task chunk; inspect episode metadata first")
    files = {f"data/{chunk}/*.parquet"}
    for episode in episodes:
        for camera in CAMERAS:
            for modality in (("rgb", "depth_linear") if args.rgbd else ("rgb",)):
                key = f"observation.{modality}.{camera}"
                prefix = f"videos/{key}"
                files.add(info["video_path"].format(video_key=key,
                    chunk_index=int(episode[f"{prefix}/chunk_index"]), file_index=int(episode[f"{prefix}/file_index"])))
    snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=str(args.root),
                      allow_patterns=sorted(files))
    print(json.dumps({**provenance, "episodes": args.episodes, "video_shards": len(files) - 1,
                      "modality": "rgbd" if args.rgbd else "rgb"}))


if __name__ == "__main__":
    main()
