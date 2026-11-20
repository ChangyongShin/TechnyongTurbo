"""Verified same-policy fine-tuning with fresh optimizer and unchanged units."""
from copy import deepcopy


def verify_finetune_contract(checkpoint, metadata):
    for key in ("normalization", "preprocessing", "train_episode_ids", "validation_episode_ids",
                "data_sources", "action_order"):
        if key not in checkpoint or checkpoint[key] != metadata[key]:
            raise ValueError(f"Fine-tune contract changed: {key}")
    if checkpoint.get("modality", "rgb") != metadata["modality"]:
        raise ValueError("Fine-tune modality changed")
    original, current = deepcopy(checkpoint["model_config"]), deepcopy(metadata["model_config"])
    for section in ("vision", "text"):
        original[section]["model_name_or_path"] = current[section]["model_name_or_path"]
    if original != current:
        raise ValueError("Fine-tune architecture changed; full strict transfer required")
    if checkpoint.get("phase_spec") != metadata.get("phase_spec"):
        raise ValueError("Fine-tune phase specification changed")
