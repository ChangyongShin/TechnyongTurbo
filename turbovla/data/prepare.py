"""Prepare an explicit small LeRobot-v3 episode subset for initial training.

Reads episode offsets (including video timestamps) rather than guessing file names.
RGB is the baseline. Depth decoding is intentionally not delegated to an RGB codec.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from turbovla.data.preprocessing import CAMERAS, finite_array
from turbovla.data.tasks import load_tasks
from turbovla.data.depth import decode_depth, validate_depth_feature


def decode_rgb(path, timestamps, fps, size=None, destination=None):
    import av

    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    images = []
    i = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        start = max(0, float(timestamps[0]) - 1.0)
        container.seek(int(start / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            time = float(frame.pts * stream.time_base)
            target = float(timestamps[i])
            if time < target - 0.51 / fps:
                continue
            if abs(time - target) > 0.51 / fps:
                raise ValueError(f"Missing frame in {path}: requested {target}, decoded {time}")
            image = frame.to_ndarray(format="rgb24")
            if size is not None:
                from PIL import Image
                image = np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BICUBIC))
            if destination is None:
                images.append(image)
            else:
                destination[i] = image
            i += 1
            if i == len(timestamps):
                break
    if i != len(timestamps):
        raise ValueError(f"Video ended before all timestamps were decoded: {path}")
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--format", choices=["mmap", "npz"], default="mmap")
    parser.add_argument("--rgbd", action="store_true", help="Decode aligned released log12 depth to metric cache")
    args = parser.parse_args()
    import pyarrow.dataset as ds

    if args.image_size < 16 or args.image_size % 16:
        raise ValueError("image size must be divisible by 16")
    info = json.loads((args.root / "meta/info.json").read_text())
    for key, dimension in (("action", 23), ("observation.state", 61)):
        if info["features"][key]["shape"] != [dimension]:
            raise ValueError(f"Unexpected {key} schema; explicit remapping required")
    if info["codebase_version"] != "v3.0" or info["fps"] != 30:
        raise ValueError("This adapter expects BEHAVIOR LeRobot v3 at 30 Hz")
    if args.rgbd:
        for camera in CAMERAS:
            depth_feature = info["features"][f"observation.depth_linear.{camera}"]
            validate_depth_feature(depth_feature)
            if depth_feature["shape"][:2] != info["features"][f"observation.rgb.{camera}"]["shape"][:2]:
                raise ValueError("Released RGB/depth resolutions are not aligned")
    episodes = ds.dataset(str(args.root / "meta/episodes"), format="parquet").to_table(
        filter=ds.field("episode_index").isin(args.episodes)
    ).to_pylist()
    if {int(e["episode_index"]) for e in episodes} != set(args.episodes):
        raise ValueError("Selected episode metadata is missing")
    # Official tasks.jsonl supplies task-level instructions, never future skill labels.
    tasks = load_tasks(args.root)
    source_path = args.root / "turbovla_source.json"
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else None
    args.out.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        eid = int(episode["episode_index"])
        destination = args.out / (f"episode_{eid:06d}" + (".npz" if args.format == "npz" else ""))
        if destination.exists():
            raise FileExistsError(destination)
        # An episode can span data files: filter the dataset, not one guessed shard.
        rows = ds.dataset(str(args.root / "data"), format="parquet").to_table(
            filter=ds.field("episode_index") == eid,
            columns=["index", "frame_index", "timestamp", "task_index", "observation.state", "action"],
        ).to_pylist()
        rows.sort(key=lambda row: row["frame_index"])
        if not rows or [int(r["frame_index"]) for r in rows] != list(range(len(rows))):
            raise ValueError(f"Incomplete or noncontiguous episode {eid}")
        if int(episode["length"]) != len(rows):
            raise ValueError(f"Episode {eid} is not fully downloaded")
        timestamps = np.asarray([r["timestamp"] for r in rows], dtype=np.float64)
        if abs(timestamps[0]) > 1e-4:
            raise ValueError("Expected episode-relative timestamps starting at zero")
        if not np.allclose(np.diff(timestamps), 1 / info["fps"], atol=1e-4):
            raise ValueError("Unexpected frame timing")
        states = finite_array([r["observation.state"] for r in rows], (len(rows), 61), "states")
        actions = finite_array([r["action"] for r in rows], (len(rows), 23), "actions")
        task_ids = {int(r["task_index"]) for r in rows}
        if len(task_ids) != 1:
            raise ValueError("Expected one task per episode")
        task_id = task_ids.pop()
        instruction = tasks[task_id]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Missing task instruction")
        rgb, depths = [], []
        if args.format == "mmap":
            destination.mkdir()
            rgb_cache = np.lib.format.open_memmap(destination / "rgb.npy", mode="w+", dtype=np.uint8,
                                                  shape=(len(rows), 3, args.image_size, args.image_size, 3))
            if args.rgbd:
                depth_cache = np.lib.format.open_memmap(destination / "depth.npy", mode="w+", dtype=np.float32,
                                                        shape=(len(rows), 3, args.image_size, args.image_size))
        for view_index, camera in enumerate(CAMERAS):
            key = f"observation.rgb.{camera}"
            prefix = f"videos/{key}"
            video_path = args.root / info["video_path"].format(
                video_key=key, chunk_index=int(episode[f"{prefix}/chunk_index"]),
                file_index=int(episode[f"{prefix}/file_index"]),
            )
            offset = float(episode[f"{prefix}/from_timestamp"])
            frames = decode_rgb(video_path, timestamps + offset, info["fps"], args.image_size,
                                destination=rgb_cache[:, view_index] if args.format == "mmap" else None)
            if args.format == "npz":
                rgb.append(np.stack(frames))
            if args.rgbd:
                depth_key = f"observation.depth_linear.{camera}"
                depth_prefix = f"videos/{depth_key}"
                depth_path = args.root / info["video_path"].format(video_key=depth_key,
                    chunk_index=int(episode[f"{depth_prefix}/chunk_index"]),
                    file_index=int(episode[f"{depth_prefix}/file_index"]))
                depth_timestamps = timestamps + float(episode[f"{depth_prefix}/from_timestamp"])
                depth_frames = decode_depth(depth_path, depth_timestamps, info["fps"], args.image_size,
                    destination=depth_cache[:, view_index] if args.format == "mmap" else None,
                    expected_shape=info["features"][depth_key]["shape"][:2])
                if args.format == "npz":
                    depths.append(np.stack(depth_frames))
        metadata = dict(instruction=instruction, episode_id=eid, task_id=task_id, fps=info["fps"],
                        cameras=list(CAMERAS), source_root=str(args.root.resolve()))
        metadata["modality"] = "rgbd" if args.rgbd else "rgb"
        if args.rgbd:
            metadata["depth_encoding"] = "m"
        if args.format == "mmap":
            metadata["source"] = source
        if args.format == "mmap":
            rgb_cache.flush()
            del rgb_cache
            if args.rgbd:
                depth_cache.flush()
                del depth_cache
            np.save(destination / "states.npy", states)
            np.save(destination / "actions.npy", actions)
            # Completion marker: incomplete caches never load as valid episodes.
            (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        else:
            extra = {"depth": np.stack(depths, axis=1)} if args.rgbd else {}
            np.savez_compressed(destination, rgb=np.stack(rgb, axis=1), states=states, actions=actions, **metadata, **extra)
        print(f"Prepared {destination}: {len(rows)} frames", flush=True)


if __name__ == "__main__":
    main()
