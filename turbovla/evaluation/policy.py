"""Model-to-controller adapter; the caller maps official observations explicitly.

No world-pose access, wheel kinematics, gripper binarization or extra base scaling.
The output stays in demonstration/controller-input order and units.
"""
from collections import deque

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

    def reset(self):
        self.queue.clear()

    @classmethod
    def from_checkpoint(cls, path, device="cuda", execute_steps=1, *, dinov3=None, bert=None):
        import torch
        from turbovla.models import build_turbovla

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["model_config"]
        if dinov3 is not None:
            config["vision"]["model_name_or_path"] = str(dinov3)
        if bert is not None:
            config["text"]["model_name_or_path"] = str(bert)
        if not str(device).startswith("cuda") or not torch.cuda.is_bf16_supported():
            config["vision"]["compute_precision"] = "fp32"
        model = build_turbovla(config)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device)
        return cls(model, Normalizer(**checkpoint["normalization"]), checkpoint["preprocessing"], execute_steps)

    def infer(self, rgb, state, instruction, *, depth=None, depth_encoding=None):
        import torch

        if (self.input_channels == 4) != (depth is not None):
            raise ValueError("Depth input must match the checkpoint RGB/RGB-D modality")
        if not self.queue:
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("A task instruction is required")
            images = preprocess_views(rgb, **self.preprocessing, depth=depth, depth_encoding=depth_encoding)
            device = next(self.model.parameters()).device
            with torch.inference_mode():
                predicted = self.model(
                    [instruction], torch.from_numpy(images[None]).to(device),
                    torch.from_numpy(self.normalizer.state(state)[None]).to(device),
                )[0].float().cpu().numpy()
            if predicted.shape != (self.model.chunk_size, 23):
                raise ValueError("Invalid predicted action chunk")
            actions = self.normalizer.inverse_action(predicted)
            self.queue.extend(actions[:self.execute_steps].copy())
        return {"action": np.asarray(self.queue.popleft(), dtype=np.float32)}
