"""Launch a separate staged task-0 experiment using the existing baseline split."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dinov3", default="/workspace/models/dinov3-vitb")
    parser.add_argument("--bert", default="/workspace/models/bert-base-uncased")
    parser.add_argument("--smoke", action="store_true", help="Use two TRAIN episodes and one held-out episode")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decision-pretrain-steps", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()
    baseline = json.loads(args.baseline_config.read_text(encoding="utf-8"))
    train_ids, val_ids = sorted(baseline["train_episode_ids"]), sorted(baseline["validation_episode_ids"])
    if args.smoke:
        train_ids, val_ids = train_ids[:2], val_ids[:1]
    if not train_ids or not val_ids or set(train_ids) & set(val_ids):
        raise ValueError("Invalid baseline split")
    train = [args.cache / f"episode_{eid:06d}" for eid in train_ids]
    val = [args.cache / f"episode_{eid:06d}" for eid in val_ids]
    revisions = {s["revision"] for s in baseline.get("data_sources", []) if s and "revision" in s}
    for eid, path in zip(train_ids + val_ids, train + val):
        marker = path / "metadata.json"
        if not marker.exists():
            raise FileNotFoundError(f"Prepare complete episode {eid} first: {path}")
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if int(metadata["episode_id"]) != eid or int(metadata["task_id"]) != 0:
            raise ValueError(f"Expected task0 episode {eid}: {path}")
        if revisions and (metadata.get("source") or {}).get("revision") not in revisions:
            raise ValueError(f"Cache revision differs from baseline: {path}")
    warmup = args.decision_pretrain_steps if args.decision_pretrain_steps is not None else (0 if args.smoke else min(100, args.steps // 10))
    root = Path(__file__).resolve().parents[2]
    print(json.dumps({"fresh_initialization": True, "train_ids": train_ids, "validation_ids": val_ids,
                      "decision_pretrain_steps": warmup, "out": str(args.out)}), flush=True)
    command = [sys.executable, "-u", "-m", "turbovla.training.trainer",
               "--train", *map(str, train), "--validation", *map(str, val),
               "--dinov3", args.dinov3, "--bert", args.bert, "--out", str(args.out),
               "--reference-config", str(args.baseline_config), "--staged", "--phase-balanced",
               "--group-balanced-loss", "--temporal-weight", "0.1", "--decision-weight", "0.1",
               "--decision-pretrain-steps", str(warmup), "--steps", str(args.steps),
               "--save-every", str(args.save_every), "--batch-size", str(args.batch_size),
               "--grad-accumulation", "1", "--device", args.device,
               "--vision-precision", "auto", "--wandb-mode", args.wandb_mode]
    subprocess.run(command, cwd=root, check=True,
                   env={**os.environ, "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")})


if __name__ == "__main__":
    main()
