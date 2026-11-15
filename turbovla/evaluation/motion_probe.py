"""Compare expert commands and image counterfactuals on explicit TRAIN caches."""
import argparse
import json
from pathlib import Path

import numpy as np

from turbovla.data.dataset import load_episode
from turbovla.evaluation.audit_behavior import grouped, pose_hold_action
from turbovla.evaluation.policy import BehaviorPolicy


def select_phases(episode, limit=16, xy_threshold=.01, yaw_threshold=.02, joint_threshold=.01):
    if limit < 1 or min(xy_threshold, yaw_threshold, joint_threshold) <= 0:
        raise ValueError("Sample count and descriptive thresholds must be positive")
    action, state = episode["actions"], episode["states"]
    moving = np.linalg.norm(action[:, :2], axis=1) >= xy_threshold
    turning = np.abs(action[:, 2]) >= yaw_threshold
    # Target-current difference, not proof of actual motion or contact phase.
    delta = np.maximum(np.abs(action[:, 7:14] - state[:, 3:10]).max(axis=1),
                       np.abs(action[:, 15:22] - state[:, 28:35]).max(axis=1))
    arms = delta >= joint_threshold
    masks = {"moving_xy": moving, "turning_yaw": turning,
             "arm_target_change": arms, "quiet_commands": ~(moving | turning | arms)}
    selected, populations = {}, {}
    for name, mask in masks.items():
        indices = np.flatnonzero(mask)
        populations[name] = len(indices)
        selected[name] = indices[np.unique(np.linspace(0, len(indices)-1, min(limit, len(indices)), dtype=int))].tolist() if len(indices) else []
    return selected, populations


def summarize(records, phase):
    selected = [r for r in records if phase in r["phases"]]
    if not selected:
        return {"samples": 0, "note": "No selected frames in this phase"}
    target = np.asarray([r["expert_action"] for r in selected])
    predictions = {name: np.asarray([r["predictions"][name] for r in selected])
                   for name in selected[0]["predictions"]}
    result = {"samples": len(selected), "raw_controller_l1": {},
              "expert_xy_command_norm_mean": float(np.linalg.norm(target[:, :2], axis=1).mean()),
              "expert_yaw_command_abs_mean": float(np.abs(target[:, 2]).mean())}
    for name, pred in predictions.items():
        error = np.abs(pred - target).mean(axis=0)
        norms = np.linalg.norm(pred[:, :2], axis=1)
        true_norms = np.linalg.norm(target[:, :2], axis=1)
        nonzero = true_norms >= .01
        entry = {"per_axis": error.tolist(), "groups": grouped(error),
                 "predicted_xy_command_norm_mean": float(norms.mean()),
                 "predicted_yaw_command_abs_mean": float(np.abs(pred[:, 2]).mean())}
        if nonzero.any():
            entry["xy_norm_ratio_on_moving_frames"] = float((norms[nonzero] / true_norms[nonzero]).mean())
            usable = nonzero & (norms > 1e-6)
            entry["xy_direction_cosine_mean"] = float((np.sum(pred[usable, :2] * target[usable, :2], axis=1) /
                (norms[usable] * true_norms[usable])).mean()) if usable.any() else None
            entry["moving_frames_predicted_below_0_01_fraction"] = float((norms[nonzero] < .01).mean())
        result["raw_controller_l1"][name] = entry
    return result


def probe(policy, episodes, samples_per_phase=16):
    records, populations = [], []
    mean = sum(np.asarray(e["actions"]).sum(axis=0, dtype=np.float64) for e in episodes) / sum(len(e["actions"]) for e in episodes)
    for episode in episodes:
        if episode.get("modality", "rgb") != "rgb":
            raise ValueError("This initial visual probe is for the RGB baseline only")
        selected, counts = select_phases(episode, samples_per_phase)
        populations.append({"episode_id": int(episode["episode_id"]), "frames": len(episode["actions"]),
                            "phase_population_frames": counts})
        indices = sorted({t for values in selected.values() for t in values})
        for t in indices:
            state, rgb = episode["states"][t], episode["rgb"][t]
            other_t = (t + max(1, len(episode["states"]) // 2)) % len(episode["states"])
            inputs = {"normal_rgb": rgb, "black_rgb": np.zeros_like(rgb),
                      "other_time_rgb": episode["rgb"][other_t]}
            predictions = {}
            for name, images in inputs.items():
                policy.reset()  # No stale queue across counterfactuals.
                predictions[name] = policy.infer(images, state, episode["instruction"])["action"].tolist()
            predictions["pose_hold_mean_grippers"] = pose_hold_action(state, mean).tolist()
            predictions["selected_train_mean"] = mean.tolist()
            records.append({"episode_id": int(episode["episode_id"]), "task_id": int(episode.get("task_id", -1)),
                            "frame": t, "other_image_frame": other_t,
                            "phases": [name for name, values in selected.items() if t in values],
                            "expert_action": episode["actions"][t].tolist(), "predictions": predictions})
        print(f"Episode {episode['episode_id']}: tested {len(indices)} frames with three image conditions", flush=True)
    return {"episode_populations": populations, "records": records,
            "phase_summary": {name: summarize(records, name) for name in
                              ("moving_xy", "turning_yaw", "arm_target_change", "quiet_commands")},
            "interpretation": [
                "This is offline TRAIN analysis, not simulator replay or closed-loop success.",
                "Phase sets overlap; phase means are on deterministic selected frames, not all phase frames.",
                "Thresholds use raw controller-input units and are descriptive, not success thresholds.",
                "Arm phase assumes the declared state/action ordering and uses target-current deviation.",
                "State and instruction are held fixed for image conditions; all camera roles stay in order.",
                "Black images are out of distribution. Other-time images can also be inconsistent with state.",
                "Image sensitivity alone does not prove target-object recognition. Compare normal-image accuracy too.",
                "Pose hold uses mean gripper commands; mean is from selected TRAIN episodes only."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--samples-per-phase", type=int, default=16)
    parser.add_argument("--dinov3")
    parser.add_argument("--bert")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new probe output directory")
    policy = BehaviorPolicy.from_checkpoint(args.checkpoint, args.device, 1, dinov3=args.dinov3, bert=args.bert)
    if policy.input_channels != 3:
        raise ValueError("Use an RGB checkpoint")
    if not set(args.episodes).issubset(set(policy.deployment_metadata.get("train_episode_ids") or [])):
        raise ValueError("Select this checkpoint's TRAIN episodes only")
    episodes = [load_episode(args.cache / f"episode_{eid:06d}") for eid in args.episodes]
    for eid, episode in zip(args.episodes, episodes):
        if int(episode["episode_id"]) != eid:
            raise ValueError("Cache episode identity mismatch")
    result = probe(policy, episodes, args.samples_per_phase)
    args.out.mkdir(parents=True)
    result["checkpoint_identity"] = policy.deployment_metadata
    (args.out / "motion-probe.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "phase_summary": result["phase_summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
