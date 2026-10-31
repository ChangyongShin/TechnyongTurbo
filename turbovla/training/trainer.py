"""Single-device RGB/RGB-D training: python -m turbovla.training.trainer --help."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from turbovla.data.dataset import BehaviorDataset, load_episode, training_normalizer
from turbovla.evaluation.observations import ACTION_ORDER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+")
    parser.add_argument("--validation", type=Path, nargs="+")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--task-balanced", action="store_true")
    parser.add_argument("--reference-config", type=Path,
                        help="Baseline config.json defining a fixed action evaluation scale")
    parser.add_argument("--dinov3", required=True)
    parser.add_argument("--bert", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vision-precision", choices=["auto", "fp32", "bf16_autocast"], default="auto")
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--rgbd", action="store_true")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                        default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()
    manifest = None
    if args.split_manifest:
        if args.train or args.validation:
            raise ValueError("Use --split-manifest or explicit episode paths, not both")
        manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        args.train = [Path(p) for p in manifest["train_paths"]]
        args.validation = [Path(p) for p in manifest["validation_paths"]]
    if not args.train or not args.validation:
        raise ValueError("Nonempty training and validation paths are required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This entry point is single-GPU; launch with python, not multi-process torchrun")
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor
    from turbovla.models import build_turbovla, TurboVLAConfig

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch and an NVIDIA GPU are required for --device cuda")
    supports_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    precision = args.vision_precision
    if precision == "auto":
        precision = "bf16_autocast" if supports_bf16 else "fp32"
    if precision == "bf16_autocast" and not supports_bf16:
        raise ValueError("BF16 requires a compatible CUDA GPU; use --vision-precision fp32")

    if args.image_size < 16 or args.image_size % 16:
        raise ValueError("image size must be positive and divisible by 16")
    if min(args.steps, args.batch_size, args.horizon, args.grad_accumulation, args.save_every) < 1 or args.lr <= 0 or args.workers < 0:
        raise ValueError("steps, batch size, horizon and lr must be positive")
    if args.resume and args.initial_weights:
        raise ValueError("Use either --resume or --initial-weights")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a new output directory; checkpoints are not overwritten")
    train_ids = [int(load_episode(p)["episode_id"]) for p in args.train]
    val_ids = [int(load_episode(p)["episode_id"]) for p in args.validation]
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids) or set(train_ids) & set(val_ids):
        raise ValueError("Duplicate or overlapping training/validation episode IDs")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    normalizer = training_normalizer(args.train)
    processor = AutoImageProcessor.from_pretrained(args.dinov3, local_files_only=True)
    preprocessing = dict(size=args.image_size, mean=list(processor.image_mean), std=list(processor.image_std))
    config = TurboVLAConfig.from_mapping({
        "text": {"model_name_or_path": args.bert, "local_files_only": True},
        "vision": {"model_name_or_path": args.dinov3, "num_views": 3, "input_channels": 4 if args.rgbd else 3,
                   "image_size": args.image_size, "frozen": True, "local_files_only": True,
                   "compute_precision": precision},
        "action": {"action_dim": 23, "state_dim": 61, "horizon": args.horizon},
    })
    model = build_turbovla(config)
    if args.initial_weights:
        # Only explicit core-model keys and matching shapes are transferred.
        source = torch.load(args.initial_weights, map_location="cpu", weights_only=True)
        source = source.get("model", source.get("state_dict", source))
        target = model.state_dict()
        compatible = {k: v for k, v in source.items() if k in target and hasattr(v, "shape") and v.shape == target[k].shape}
        if not compatible:
            raise ValueError("No matching core TurboVLA tensors; checkpoint conversion required")
        result = model.load_state_dict(compatible, strict=False)
        print(json.dumps({"loaded_tensors": len(compatible), "missing_keys": result.missing_keys}), flush=True)
    model.to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    modality = "rgbd" if args.rgbd else "rgb"
    train_dataset = BehaviorDataset(args.train, normalizer, preprocessing, args.horizon, modality)
    validation_dataset = BehaviorDataset(args.validation, normalizer, preprocessing, args.horizon, modality)
    from turbovla.data.multitask import TaskBalancedSampler, reference_action_l1
    from turbovla.data.preprocessing import Normalizer
    sampler = TaskBalancedSampler(train_dataset.episodes, args.seed) if args.task_balanced else None
    if args.task_balanced:
        training_tasks = {int(e["task_id"]) for e in train_dataset.episodes}
        validation_tasks = {int(e.get("task_id", -1)) for e in validation_dataset.episodes}
        if training_tasks != validation_tasks:
            raise ValueError("Task-balanced runs require the same tasks in train and validation")
    if manifest:
        for paths, episodes, split_key in ((args.train, train_dataset.episodes, "train"),
                                           (args.validation, validation_dataset.episodes, "validation")):
            for path, episode in zip(paths, episodes):
                task = str(int(episode["task_id"]))
                entry = manifest["tasks"][task]
                if int(episode["episode_id"]) not in entry[split_key]:
                    raise ValueError(f"Cache does not match task manifest: {path}")
                if (episode.get("source") or {}).get("revision") != manifest["revision"]:
                    raise ValueError(f"Cache dataset revision mismatch: {path}")
    reference = None
    reference_metadata = None
    if args.reference_config:
        reference_metadata = json.loads(args.reference_config.read_text(encoding="utf-8"))
        reference = Normalizer(**reference_metadata["normalization"])
    train = DataLoader(train_dataset, batch_size=args.batch_size,
                       shuffle=sampler is None, sampler=sampler, num_workers=args.workers, pin_memory=device.type == "cuda")
    validation = DataLoader(validation_dataset, batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {"model_config": config.to_dict(), "normalization": normalizer.to_dict(),
                "preprocessing": preprocessing, "train_episode_ids": train_ids, "validation_episode_ids": val_ids,
                "data_sources": [load_episode(p).get("source") for p in [*args.train, *args.validation]],
                "modality": modality, "depth_preprocessing": {"encoding": "m", "clip": [0.01, 10.0],
                                                            "scale": 10.0} if args.rgbd else None,
                "action_order": ACTION_ORDER, "seed": args.seed,
                "training": {"batch_size": args.batch_size, "grad_accumulation": args.grad_accumulation,
                             "lr": args.lr, "workers": args.workers,
                             "task_balanced": args.task_balanced}}
    if manifest:
        metadata["split_manifest"] = manifest
    if reference is not None:
        metadata["reference_normalization"] = reference.to_dict()
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        for key in ("normalization", "preprocessing", "train_episode_ids", "validation_episode_ids"):
            if checkpoint[key] != metadata[key]:
                raise ValueError(f"Resume contract changed: {key}")
        if "data_sources" in checkpoint and checkpoint["data_sources"] != metadata["data_sources"]:
            raise ValueError("Resume dataset provenance changed")
        if checkpoint.get("training", {}).get("task_balanced", False) != args.task_balanced:
            raise ValueError("Resume sampling strategy changed")
        previous_config = checkpoint["model_config"]
        # Backbone paths may change when a run moves between servers.
        for section in ("vision", "text"):
            previous_config[section]["model_name_or_path"] = metadata["model_config"][section]["model_name_or_path"]
        if previous_config != metadata["model_config"]:
            raise ValueError("Resume model configuration changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        if start_step >= args.steps:
            raise ValueError("--steps must exceed the resumed optimizer step")
    (args.out / "config.json").write_text(json.dumps(metadata, indent=2))

    def loss_for(batch, details=False):
        prediction = model(batch["instruction"], batch["images"].to(args.device), batch["state"].to(args.device))
        target = batch["actions"].to(args.device)
        mask = batch["mask"].to(args.device)[..., None]
        if prediction.shape != target.shape:
            raise ValueError(f"Action shape mismatch: {prediction.shape} vs {target.shape}")
        error = (prediction - target).abs() * mask
        loss = error.sum() / (mask.sum().clamp_min(1) * 23)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        if details:
            reference_error = None
            if reference is not None:
                reference_error = reference_action_l1(
                    prediction, batch["raw_actions"].to(args.device), mask, normalizer, reference)
            return error.detach().sum(1), mask.sum(1).squeeze(-1), reference_error
        return loss, error.detach().sum((0, 1)), mask.sum().detach()

    from contextlib import nullcontext
    tracking = nullcontext(None)
    if args.wandb_mode != "disabled":
        import wandb
        tracking = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "TechnyongTurbo"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=os.environ.get("WANDB_NAME") or args.out.name,
            mode=args.wandb_mode,
            config={**{k: str(v) if isinstance(v, Path) else
                       [str(p) for p in v] if isinstance(v, list) else v
                       for k, v in vars(args).items()},
                    "train_episode_ids": train_ids, "validation_episode_ids": val_ids,
                    "start_step": start_step, "resolved_precision": precision,
                    "effective_batch_size": args.batch_size * args.grad_accumulation},
        )
    with tracking as run:
        if run is not None:
            run.define_metric("train_step")
            run.define_metric("train/*", step_metric="train_step")
            run.define_metric("validation/*", step_metric="train_step")
        iterator = iter(train)
        from collections import Counter
        sampled_tasks = Counter()
        for step in range(start_step + 1, args.steps + 1):
            model.train()
            # Keep the frozen visual backbone deterministic.
            model.vision_encoder.backbone.eval()
            optimizer.zero_grad(set_to_none=True)
            logged_loss = 0.0
            for _ in range(args.grad_accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train)
                    batch = next(iterator)
                loss, _, _ = loss_for(batch)
                sampled_tasks.update(batch["task_id"].tolist())
                (loss / args.grad_accumulation).backward()
                logged_loss += loss.item() / args.grad_accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if run is not None:
                run.log({"train_step": step, "train/l1": logged_loss,
                         "train/lr": optimizer.param_groups[0]["lr"],
                         **{f"train/task_{task:03d}_sample_fraction": n / sum(sampled_tasks.values())
                            for task, n in sampled_tasks.items()}})
            if step == 1 or step % 50 == 0:
                print(json.dumps({"step": step, "train_l1": logged_loss}), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                model.eval()
                total = torch.zeros(23, device=args.device)
                count = torch.zeros((), device=args.device)
                by_task = {}
                with torch.inference_mode():
                    for batch in validation:
                        error, valid, reference_error = loss_for(batch, details=True)
                        total += error.sum(0)
                        count += valid.sum()
                        for task in batch["task_id"].unique().tolist():
                            chosen = (batch["task_id"] == task).to(args.device)
                            if task not in by_task:
                                by_task[task] = [torch.zeros(23, device=args.device),
                                                 torch.zeros((), device=args.device),
                                                 torch.zeros(23, device=args.device)]
                            by_task[task][0] += error[chosen].sum(0)
                            by_task[task][1] += valid[chosen].sum()
                            if reference_error is not None:
                                by_task[task][2] += reference_error[chosen].sum(0)
                values = (total / count).cpu().tolist()
                print(json.dumps({"step": step, "validation_l1_per_action": values}), flush=True)
                task_metrics = {}
                task_means = []
                for task, (task_error, task_count, ref_error) in sorted(by_task.items()):
                    task_values = (task_error / task_count).cpu().tolist()
                    task_mean = sum(task_values) / len(task_values)
                    task_means.append(task_mean)
                    result = {"step": step, "task_id": task, "validation_l1_per_action": task_values}
                    task_metrics[f"validation/task_{task:03d}/l1_mean"] = task_mean
                    for i, value in enumerate(task_values):
                        task_metrics[f"validation/task_{task:03d}/action_{i:02d}_l1"] = value
                    if reference is not None:
                        ref_values = (ref_error / task_count).cpu().tolist()
                        result["reference_l1_per_action"] = ref_values
                        task_metrics[f"validation/task_{task:03d}/reference_l1_mean"] = sum(ref_values) / 23
                    print(json.dumps(result), flush=True)
                task_metrics["validation/task_macro_l1_mean"] = sum(task_means) / len(task_means)
                if run is not None:
                    run.log({"train_step": step, "validation/l1_mean": sum(values) / len(values),
                             **{f"validation/action_{i:02d}_l1": v for i, v in enumerate(values)}, **task_metrics})
                temporary = args.out / f"step_{step}.pt.tmp"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, **metadata}, temporary)
                temporary.replace(args.out / f"step_{step}.pt")


train_model = main


if __name__ == "__main__":
    main()
