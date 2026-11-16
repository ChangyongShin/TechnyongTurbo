"""Validated policy weight transfer; does not resume optimizer state."""
from copy import deepcopy


def load_policy_warm_start(model, checkpoint, metadata):
    from turbovla.models.configuration import TurboVLAConfig

    for key in ("normalization", "preprocessing", "action_order", "train_episode_ids", "validation_episode_ids", "data_sources"):
        if key not in checkpoint or checkpoint[key] != metadata[key]:
            raise ValueError(f"Warm-start contract changed or absent: {key}")
    source_config = TurboVLAConfig.from_mapping(checkpoint["model_config"]).to_dict()
    target_config = deepcopy(metadata["model_config"])
    for section in ("text", "vision"):
        source_config[section]["model_name_or_path"] = target_config[section]["model_name_or_path"]
    source_config["vision"]["compute_precision"] = target_config["vision"]["compute_precision"]
    source_config["action"]["decision_conditioning"] = target_config["action"]["decision_conditioning"]
    if source_config != target_config:
        raise ValueError("Warm-start architecture differs beyond decision conditioning")
    source, target = checkpoint["model"], model.state_dict()
    is_decision = lambda key: key.startswith(("decision_head.", "decision_embedding."))
    missing = [key for key in target if key not in source and not is_decision(key)]
    unexpected = [key for key in source if key not in target and not is_decision(key)]
    mismatched = [key for key in target if key in source and source[key].shape != target[key].shape]
    if missing or unexpected or mismatched:
        raise ValueError(f"Incomplete policy transfer: missing={missing}, unexpected={unexpected}, shapes={mismatched}")
    transferred = {key: value for key, value in source.items() if key in target}
    result = model.load_state_dict(transferred, strict=False)
    return {"loaded_tensors": len(transferred), "initialized_keys": result.missing_keys,
            "source_step": checkpoint.get("step"), "optimizer_resumed": False,
            "decision_conditioning_changed": checkpoint["model_config"].get("action", {}).get("decision_conditioning", False)
                != target_config["action"]["decision_conditioning"]}
