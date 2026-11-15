"""Audit a saved policy on explicit TRAIN caches without executing a robot."""
import argparse
import json
from pathlib import Path

import numpy as np

from turbovla.data.dataset import BehaviorDataset
from turbovla.data.preprocessing import CAMERAS
from turbovla.evaluation.observations import ACTION_ORDER
from turbovla.evaluation.policy import BehaviorPolicy


def pose_hold_action(state, gripper_fallback):
    """Declared state61 mapping; grippers use a mean, not an invented qpos inverse."""
    action = np.asarray(gripper_fallback, dtype=np.float32).copy()
    action[:3] = 0
    action[3:7] = state[53:57]
    action[7:14] = state[3:10]
    action[15:22] = state[28:35]
    return action


def grouped(values):
    return {group: float(np.mean(values[a:b])) for group, (a, b) in ACTION_ORDER.items()}


def audit(policy, paths, samples_per_episode=16):
    normalizer = policy.normalizer
    dataset = BehaviorDataset(paths, normalizer, policy.preprocessing, policy.model.chunk_size,
                              "rgbd" if policy.input_channels == 4 else "rgb")
    sums = np.zeros(23, dtype=np.float64)
    frames = 0
    roundtrip = np.zeros(23)
    outside = np.zeros(23, dtype=np.int64)
    base_slow = 0
    for episode in dataset.episodes:
        actions = np.asarray(episode["actions"])
        frames += len(actions)
        sums += actions.sum(axis=0, dtype=np.float64)
        restored = normalizer.inverse_action(normalizer.action(actions))
        roundtrip = np.maximum(roundtrip, np.abs(restored - actions).max(axis=0))
        outside += ((actions < normalizer.action_min - 1e-6) |
                    (actions > normalizer.action_max + 1e-6)).sum(axis=0)
        base_slow += int((np.linalg.norm(actions[:, :2], axis=1) < .01).sum())
    mean_action = (sums / frames).astype(np.float32)
    error = {name: np.zeros(23) for name in ("model_first", "model_chunk", "mean_first", "pose_hold_first")}
    raw_error = {name: np.zeros(23) for name in ("model_first", "mean_first", "pose_hold_first")}
    first_count, chunk_count, start, input_diff = 0, 0, 0, 0.
    sampled = []
    policy.trace_enabled = True
    for episode in dataset.episodes:
        length = len(episode["states"])
        indices = np.unique(np.linspace(0, length - 1, min(length, samples_per_episode), dtype=int))
        for t in indices:
            sample = dataset[start + int(t)]
            extra = {"depth": episode["depth"][t], "depth_encoding": "m"} if policy.input_channels == 4 else {}
            policy.reset()
            predicted_action = policy.infer(episode["rgb"][t], episode["states"][t],
                                            episode["instruction"], **extra)["action"]
            trace = policy.last_trace
            input_diff = max(input_diff, float(np.abs(trace["processed_images"] - sample["images"]).max()),
                             float(np.abs(trace["inference_state_normalized"] - sample["state"]).max()))
            mask = sample["mask"][:, None]
            normalized_chunk = trace["normalized_chunk"]
            error["model_chunk"] += (np.abs(normalized_chunk - sample["actions"]) * mask).sum(axis=0)
            chunk_count += float(mask.sum())
            error["model_first"] += np.abs(normalized_chunk[0] - sample["actions"][0])
            baselines = {"model_first": predicted_action,
                         "mean_first": mean_action,
                         "pose_hold_first": pose_hold_action(episode["states"][t], mean_action)}
            target = episode["actions"][t]
            for name, prediction in baselines.items():
                raw_error[name] += np.abs(prediction - target)
                if name != "model_first":
                    error[name] += np.abs(normalizer.action(prediction) - sample["actions"][0])
            first_count += 1
            sampled.append({"episode_id": int(episode["episode_id"]), "frame": int(t)})
        start += length
    for name in error:
        error[name] /= chunk_count if name == "model_chunk" else first_count
    for name in raw_error:
        raw_error[name] /= first_count
    return {
        "sampled_train_frames": sampled, "sample_count": first_count,
        "roundtrip_max_abs_per_axis": roundtrip.tolist(),
        "action_outside_training_range_fraction_per_axis": (outside / frames).tolist(),
        "constant_action_axes": np.flatnonzero(normalizer.action_max - normalizer.action_min <= 1e-6).tolist(),
        "camera_order": list(CAMERAS), "train_vs_server_processed_tensor_max_abs_diff": input_diff,
        "base_xy_label_norm_below_0_01_fraction": base_slow / frames,
        "base_threshold_note": "Command-input units, not proven m/s. Descriptive threshold only.",
        "normalized_l1": {name: {"per_axis": v.tolist(), "groups": grouped(v), "mean": float(v.mean())}
                          for name, v in error.items()},
        "raw_controller_unit_l1": {name: {"per_axis": v.tolist(), "groups": grouped(v)}
                                    for name, v in raw_error.items()},
        "baseline_note": "Mean uses only selected TRAIN episodes. Hold assumes declared state ordering; "
                         "grippers use mean action because finger qpos-to-command inverse is not established. "
                         "Model normalized L1 is unclipped, baseline commands are normalized/clipped; "
                         "use raw_controller_unit_l1 as the like-for-like comparison. "
                         "Groups contain mixed units; compare matching axes and groups only.",
        "limitations": ["These TRAIN caches are not all TRAIN episodes; means/motion ratios refer to selected episodes.",
                        "Semantic controller/frame correctness needs expert replay on a matching simulator.",
                        "Identical preprocessing does not prove raw cache timing or camera source is correct.",
                        "Offline first-step/chunk error does not establish closed-loop success."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--episodes", required=True, type=int, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dinov3")
    parser.add_argument("--bert")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples-per-episode", type=int, default=16)
    args = parser.parse_args()
    if args.samples_per_episode < 1:
        raise ValueError("samples-per-episode must be positive")
    if args.out.exists():
        raise FileExistsError("Choose a new audit output directory")
    policy = BehaviorPolicy.from_checkpoint(args.checkpoint, args.device, 1, dinov3=args.dinov3, bert=args.bert)
    metadata = policy.deployment_metadata
    if not set(args.episodes).issubset(set(metadata.get("train_episode_ids") or [])):
        raise ValueError("Use episodes from this checkpoint's TRAIN split only")
    paths = [args.cache / f"episode_{eid:06d}" for eid in args.episodes]
    result = audit(policy, paths, args.samples_per_episode)
    args.out.mkdir(parents=True)
    (args.out / "checkpoint-identity.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.out / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "step": metadata["step"],
                      "checkpoint_sha256": metadata["checkpoint_sha256"],
                      "preprocessing_max_abs_diff": result["train_vs_server_processed_tensor_max_abs_diff"],
                      "roundtrip_max_abs": max(result["roundtrip_max_abs_per_axis"]),
                      "raw_group_l1": {name: values["groups"] for name, values in result["raw_controller_unit_l1"].items()}}, indent=2))


if __name__ == "__main__":
    main()
