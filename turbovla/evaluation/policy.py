"""Model-to-controller adapter; the caller maps official observations explicitly.

No world-pose access, wheel kinematics, gripper binarization or extra base scaling.
The output stays in demonstration/controller-input order and units.
"""
from collections import deque
from pathlib import Path

import numpy as np

from turbovla.data.preprocessing import Normalizer, preprocess_views


class BehaviorPolicy:
    def __init__(self, model, normalizer, preprocessing, execute_steps=1):
        if not 1 <= execute_steps <= model.chunk_size:
            raise ValueError("execute_steps must be between one and the predicted horizon")
        if model.action_dim != 23 or model.state_dim != 61 or model.num_views != 3:
            raise ValueError("Expected the BEHAVIOR 23-action/61-state/3-view model")
        self.model = model.eval()
        self.normalizer = normalizer
        self.preprocessing = preprocessing
        self.input_channels = model.config.vision.input_channels
        self.execute_steps = execute_steps
        self.queue = deque()
        self.trace_enabled = False
        self.last_trace = None
        self._trace_chunk = None
        self._chunk_index = 0

    def reset(self):
        self.queue.clear()
        self.last_trace = None
        self._trace_chunk = None
        self._chunk_index = 0

    @classmethod
    def from_checkpoint(cls, path, device="cuda", execute_steps=1, *, dinov3=None, bert=None):
        import torch
        from copy import deepcopy
        from turbovla.models import build_turbovla
        from turbovla.evaluation.diagnostics import file_sha256, object_sha256

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = deepcopy(checkpoint["model_config"])
        if dinov3 is not None:
            config["vision"]["model_name_or_path"] = str(dinov3)
        if bert is not None:
            config["text"]["model_name_or_path"] = str(bert)
        if not str(device).startswith("cuda") or not torch.cuda.is_bf16_supported():
            config["vision"]["compute_precision"] = "fp32"
        model = build_turbovla(config)
        loaded = model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device)
        policy = cls(model, Normalizer(**checkpoint["normalization"]), checkpoint["preprocessing"], execute_steps)
        policy.deployment_metadata = {
            "checkpoint_name": Path(path).name,
            "checkpoint_sha256": file_sha256(path), "step": checkpoint.get("step"),
            "normalization_sha256": object_sha256(checkpoint["normalization"]),
            "preprocessing_sha256": object_sha256(checkpoint["preprocessing"]),
            "saved_model_config_sha256": object_sha256(checkpoint["model_config"]),
            "effective_model_config": model.config.to_dict(),
            "action_order": checkpoint.get("action_order"),
            "train_episode_ids": checkpoint.get("train_episode_ids"),
            "validation_episode_ids": checkpoint.get("validation_episode_ids"),
            "horizon": model.chunk_size, "execute_steps": execute_steps,
            "eval_mode": not model.training, "weights_key": "model", "ema_selected": False,
            "missing_keys": list(loaded.missing_keys), "unexpected_keys": list(loaded.unexpected_keys),
        }
        return policy

    def infer(self, rgb, state, instruction, *, depth=None, depth_encoding=None):
        import torch

        if (self.input_channels == 4) != (depth is not None):
            raise ValueError("Depth input must match the checkpoint RGB/RGB-D modality")
        replanned = not self.queue
        if replanned:
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("A task instruction is required")
            images = preprocess_views(rgb, **self.preprocessing, depth=depth, depth_encoding=depth_encoding)
            normalized_state = self.normalizer.state(state)
            device = next(self.model.parameters()).device
            with torch.inference_mode():
                inputs = (
                    [instruction], torch.from_numpy(images[None]).to(device),
                    torch.from_numpy(normalized_state[None]).to(device),
                )
                decision_probabilities = None
                if getattr(getattr(self.model.config, "action", None), "decision_conditioning", False):
                    output, decision_logits = self.model.forward_with_decision(*inputs)
                    decision_probabilities = decision_logits.softmax(-1)[0].float().cpu().numpy()
                else:
                    output = self.model(*inputs)
                predicted = output[0].float().cpu().numpy()
            if predicted.shape != (self.model.chunk_size, 23):
                raise ValueError("Invalid predicted action chunk")
            actions = self.normalizer.inverse_action(predicted)
            self.queue.extend(actions[:self.execute_steps].copy())
            self._chunk_index = 0
            if self.trace_enabled:
                self._trace_chunk = {"inference_state_raw": np.asarray(state).copy(),
                                     "inference_state_normalized": normalized_state.copy(),
                                     "processed_images": images.copy(),
                                     "normalized_chunk": predicted.copy(),
                                     "denormalized_chunk": actions.copy()}
                self._trace_chunk.update({f"camera_{i}_raw": np.asarray(value).copy()
                                          for i, value in enumerate(rgb)})
                if decision_probabilities is not None:
                    self._trace_chunk["decision_probabilities"] = decision_probabilities
        action = np.asarray(self.queue.popleft(), dtype=np.float32)
        if self.trace_enabled:
            self.last_trace = {**self._trace_chunk, "request_state_raw": np.asarray(state).copy(),
                               "replanned": np.asarray(replanned), "chunk_index": np.asarray(self._chunk_index),
                               "action": action.copy()}
        self._chunk_index += 1
        return {"action": action}
