"""CPU integration with tiny local HF backbones, never downloading model weights."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessorFast

from turbovla.evaluation.policy import BehaviorPolicy
from turbovla.data.preprocessing import CAMERAS, Normalizer, preprocess_views
from turbovla.models import TurboVLAConfig, build_turbovla
from turbovla.models.vision_encoder import DINOv3VisionEncoder
from turbovla.models.configuration import VisionEncoderConfig


class RGBDModelTests(unittest.TestCase):
    def test_frozen_rgb_depth_gradient_and_checkpoint_policy_reload(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vision_path, text_path = root / "vision", root / "bert"
            vision_path.mkdir()
            text_path.mkdir()
            vision = DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, image_size=16, patch_size=16,
                num_register_tokens=4))
            vision.save_pretrained(vision_path)
            DINOv3ViTImageProcessorFast().save_pretrained(vision_path)
            bert = BertModel(BertConfig(vocab_size=10, hidden_size=32, intermediate_size=64,
                                       num_hidden_layers=1, num_attention_heads=4))
            bert.save_pretrained(text_path)
            (text_path / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nturn\non\nradio\n.\n?\n")
            tokenizer = BertTokenizerFast(vocab_file=str(text_path / "vocab.txt"), do_lower_case=True)
            tokenizer.save_pretrained(text_path)
            config = TurboVLAConfig.from_mapping({
                "vision": {"model_name_or_path": str(vision_path), "image_size": 16, "num_views": 3,
                           "input_channels": 4, "frozen": True, "compute_precision": "fp32", "dropout": 0.0},
                "text": {"model_name_or_path": str(text_path), "attention_implementation": "eager"},
                "interaction": {"hidden_dim": 32, "nheads": 4, "num_layers": 1, "dim_feedforward": 64,
                                "enhancer_inner_dim": 64, "fusion_droppath": 0.0},
                "action": {"action_dim": 23, "state_dim": 61, "horizon": 2, "num_layers": 1,
                           "mlp_hidden_dim": 64, "state_hidden_dim": 32, "dropout": 0.0},
            })
            model = build_turbovla(config)
            patch = model.vision_encoder.rgbd_patch
            self.assertTrue(patch.depth_weight.requires_grad)
            self.assertFalse(patch.weight.requires_grad)
            trainable_backbone = [name for name, p in model.vision_encoder.backbone.named_parameters() if p.requires_grad]
            self.assertEqual(trainable_backbone, ["embeddings.patch_embeddings.depth_weight"])
            original_rgb = patch.weight[:, :3].detach().clone()
            rgb = [np.full((16, 16, 3), 128, dtype=np.uint8)] * 3
            depth = [np.full((16, 16), 2, dtype=np.float32)] * 3
            preprocessing = dict(size=16, mean=[0.5] * 3, std=[0.5] * 3)
            images = torch.from_numpy(preprocess_views(rgb, **preprocessing, depth=depth, depth_encoding="m")[None])
            model.eval()
            rgb_encoder = DINOv3VisionEncoder(VisionEncoderConfig(model_name_or_path=str(vision_path), image_size=16,
                num_views=3, frozen=True, compute_precision="fp32"))
            rgb_encoder.eval()
            with torch.no_grad():
                rgb_response = rgb_encoder(images[:, :, :3])
                rgbd_response = model.vision_encoder(images)
            torch.testing.assert_close(rgb_response, rgbd_response, atol=1e-6, rtol=1e-6)
            normalizer = Normalizer.fit(np.stack([np.zeros(61), np.ones(61)]),
                                        np.stack([np.zeros(23), np.ones(23)]))
            state = np.full(61, 0.5, dtype=np.float32)
            state_tensor = torch.from_numpy(normalizer.state(state)[None])
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            model.train()
            model.vision_encoder.backbone.eval()
            prediction = model(["turn on radio"], images, state_tensor)
            self.assertEqual(prediction.shape, (1, 2, 23))
            loss = (prediction - 0.5).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(patch.depth_weight.grad)
            self.assertGreater(patch.depth_weight.grad.abs().sum().item(), 0)
            optimizer.step()
            torch.testing.assert_close(patch.weight[:, :3], original_rgb, atol=0, rtol=0)
            self.assertGreater(patch.depth_weight.abs().sum().item(), 0)
            checkpoint = root / "rgbd.pt"
            torch.save({"model": model.state_dict(), "model_config": config.to_dict(),
                        "preprocessing": preprocessing, "normalization": normalizer.to_dict(),
                        "modality": "rgbd"}, checkpoint)
            policy = BehaviorPolicy.from_checkpoint(checkpoint, device="cpu", dinov3=vision_path, bert=text_path)
            self.assertEqual(policy.input_channels, 4)
            model.eval()
            with torch.inference_mode():
                expected = normalizer.inverse_action(model(["turn on radio"], images, state_tensor)[0].numpy())[0]
            action = policy.infer(rgb, state, "turn on radio", depth=depth, depth_encoding="m")["action"]
            np.testing.assert_allclose(action, expected, atol=1e-6)
            with self.assertRaises(ValueError):
                policy.infer(rgb, state, "turn on radio")
            policy.reset()
            self.assertEqual(len(policy.queue), 0)

            # Exercise the actual training CLI, its default fusion/action recipe,
            # loss masking, validation and checkpoint save using tiny backbones.
            paths = []
            rng = np.random.default_rng(12)
            for eid in (0, 1):
                path = root / f"episode_{eid}.npz"
                np.savez(path, rgb=np.full((2, 3, 16, 16, 3), 128, dtype=np.uint8),
                         depth=np.full((2, 3, 16, 16), 2, dtype=np.float32), depth_encoding="m", modality="rgbd",
                         states=rng.normal(size=(2, 61)).astype(np.float32),
                         actions=rng.normal(size=(2, 23)).astype(np.float32),
                         instruction="turn on radio", episode_id=eid, fps=30, cameras=np.asarray(CAMERAS))
                paths.append(path)
            output = root / "cli-smoke"
            command = [sys.executable, "-m", "turbovla.training.trainer", "--train", str(paths[0]),
                       "--validation", str(paths[1]), "--dinov3", str(vision_path), "--bert", str(text_path),
                       "--out", str(output), "--steps", "1", "--batch-size", "1", "--horizon", "2",
                       "--image-size", "16", "--device", "cpu", "--vision-precision", "fp32", "--rgbd"]
            environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1")
            completed = subprocess.run(command, cwd=Path(__file__).parents[1], env=environment,
                                       capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            saved = torch.load(output / "step_1.pt", map_location="cpu", weights_only=True)
            self.assertEqual(saved["modality"], "rgbd")
            self.assertEqual(saved["model_config"]["vision"]["input_channels"], 4)
            self.assertEqual(saved["step"], 1)
            learned_depth = saved["model"]["vision_encoder.backbone.embeddings.patch_embeddings.depth_weight"]
            self.assertGreater(learned_depth.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
