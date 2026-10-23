"""Check A100 training assets before allocating a long training job."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--dinov3", required=True)
    parser.add_argument("--bert", required=True)
    parser.add_argument("--rgbd", action="store_true")
    args = parser.parse_args()
    import torch
    import pyarrow.dataset as ds
    from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer
    from turbovla.data.preprocessing import CAMERAS
    from turbovla.data.tasks import load_tasks
    from turbovla.data.depth import validate_depth_feature

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA/BF16-compatible training GPU is required")
    info = json.loads((args.root / "meta/info.json").read_text(encoding="utf-8"))
    if info["codebase_version"] != "v3.0" or info["fps"] != 30:
        raise ValueError("Expected released LeRobot v3, 30 Hz")
    for key, shape in (("action", [23]), ("observation.state", [61])):
        if info["features"][key]["shape"] != shape:
            raise ValueError(f"Schema mismatch for {key}")
    tasks = load_tasks(args.root)
    episodes = ds.dataset(str(args.root / "meta/episodes"), format="parquet").to_table(
        filter=ds.field("episode_index").isin(args.episodes)).to_pylist()
    if {int(e["episode_index"]) for e in episodes} != set(args.episodes):
        raise ValueError("Missing selected episode metadata")
    for episode in episodes:
        for camera in CAMERAS:
            for modality in (("rgb", "depth_linear") if args.rgbd else ("rgb",)):
                key = f"observation.{modality}.{camera}"
                if modality == "depth_linear":
                    validate_depth_feature(info["features"][key])
                prefix = f"videos/{key}"
                video = args.root / info["video_path"].format(video_key=key,
                    chunk_index=int(episode[f"{prefix}/chunk_index"]), file_index=int(episode[f"{prefix}/file_index"]))
                if not video.is_file():
                    raise FileNotFoundError(video)
    vision = AutoConfig.from_pretrained(args.dinov3, local_files_only=True)
    text = AutoConfig.from_pretrained(args.bert, local_files_only=True)
    if vision.model_type != "dinov3_vit" or text.model_type != "bert":
        raise ValueError("Use HF DINOv3 ViT and BERT models")
    AutoImageProcessor.from_pretrained(args.dinov3, local_files_only=True)
    AutoTokenizer.from_pretrained(args.bert, local_files_only=True)
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "episodes": args.episodes,
                      "tasks": len(tasks), "status": "metadata preflight passed; run prepare and 1-step training next"}))


if __name__ == "__main__":
    main()
