"""Task balancing statistics, split isolation and real multi-task training CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from turbovla.data.multitask import TaskBalancedSampler, reference_action_l1
from turbovla.data.preprocessing import CAMERAS, Normalizer
from turbovla.data.toggle_tasks import split_episodes, validate_task0_split


class MultitaskTests(unittest.TestCase):
    def test_balancing_ignores_task_frame_imbalance_and_samples_all_ranges(self):
        episodes = [{"task_id": 0, "states": range(1000)},
                    {"task_id": 66, "states": range(80000)},
                    {"task_id": 0, "states": range(1000)}]
        sampler = TaskBalancedSampler(episodes, seed=42)
        indices = np.asarray(list(sampler))
        task0 = (indices < 1000) | (indices >= 81000)
        self.assertEqual(len(indices), 82000)
        self.assertAlmostEqual(task0.mean(), .5, delta=.01)
        self.assertAlmostEqual((indices < 1000).mean(), .25, delta=.01)
        self.assertTrue(((indices >= 0) & (indices < 82000)).all())
        np.testing.assert_array_equal(indices, list(TaskBalancedSampler(episodes, seed=42)))
        self.assertFalse(np.array_equal(indices, list(sampler)))
        with self.assertRaises(ValueError):
            TaskBalancedSampler([{"states": range(2)}])

    def test_episode_split_is_fixed_disjoint_and_preserves_existing_task0(self):
        train, val = split_episodes(range(200))
        self.assertEqual((len(train), len(val)), (180, 20))
        self.assertFalse(set(train) & set(val))
        self.assertEqual((train, val), split_episodes(range(200)))
        old = {"train": [0, 2], "validation": [1]}
        self.assertEqual(validate_task0_split(old, [0, 1, 2]), ([0, 2], [1]))
        with self.assertRaises(ValueError):
            validate_task0_split({"train": [0, 1], "validation": [1]}, [0, 1, 2])

    def test_reference_error_matches_direct_affine_math_and_mask(self):
        current = Normalizer(np.zeros(61), np.ones(61), np.zeros(23), np.full(23, 10))
        reference = Normalizer(np.zeros(61), np.ones(61), np.zeros(23), np.full(23, 2))
        prediction = torch.zeros(1, 2, 23)  # raw prediction = 5, reference value = 4
        raw = torch.ones(1, 2, 23)  # reference target = 0
        mask = torch.tensor([[[1.], [0.]]])
        error = reference_action_l1(prediction, raw, mask, current, reference)
        torch.testing.assert_close(error, torch.full((1, 23), 4.))
        same = reference_action_l1(prediction, raw, mask, current, current)
        torch.testing.assert_close(same, torch.full((1, 23), .8))

    def test_real_two_task_cli_records_per_task_metrics_and_checkpoint(self):
        from transformers import (BertConfig, BertModel, BertTokenizerFast,
                                  DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessorFast)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vision, text = root / "vision", root / "text"
            vision.mkdir()
            text.mkdir()
            DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, image_size=16, patch_size=16,
                num_register_tokens=4)).save_pretrained(vision)
            DINOv3ViTImageProcessorFast().save_pretrained(vision)
            BertModel(BertConfig(vocab_size=10, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4)).save_pretrained(text)
            (text / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nturn\non\noff\nradio\nlight\n")
            BertTokenizerFast(vocab_file=str(text / "vocab.txt")).save_pretrained(text)
            rng = np.random.default_rng(42)
            paths = []
            for eid, task in enumerate((0, 0, 66, 66)):
                path = root / f"episode_{eid:06d}"
                path.mkdir()
                np.save(path / "rgb.npy", np.full((3, 3, 16, 16, 3), 128, dtype=np.uint8))
                np.save(path / "states.npy", rng.normal(size=(3, 61)).astype(np.float32))
                np.save(path / "actions.npy", rng.uniform(size=(3, 23)).astype(np.float32))
                (path / "metadata.json").write_text(json.dumps({"episode_id": eid, "task_id": task,
                    "instruction": "turn on radio" if task == 0 else "turn off light", "fps": 30,
                    "modality": "rgb", "cameras": list(CAMERAS), "source": {"revision": "test"}}))
                paths.append(str(path))
            manifest = root / "split.json"
            manifest.write_text(json.dumps({"revision": "test", "train_paths": [paths[0], paths[2]],
                "validation_paths": [paths[1], paths[3]], "tasks": {
                    "0": {"train": [0], "validation": [1]}, "66": {"train": [2], "validation": [3]}}}))
            reference = root / "reference.json"
            reference.write_text(json.dumps({"normalization": Normalizer(
                np.zeros(61), np.ones(61), np.zeros(23), np.ones(23)).to_dict()}))
            out = root / "run"
            command = [sys.executable, "-m", "turbovla.training.trainer", "--split-manifest", str(manifest),
                       "--task-balanced", "--reference-config", str(reference), "--dinov3", str(vision),
                       "--bert", str(text), "--out", str(out), "--steps", "2", "--batch-size", "2",
                       "--horizon", "2", "--image-size", "16", "--device", "cpu",
                       "--vision-precision", "fp32", "--wandb-mode", "disabled"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180,
                cwd=Path(__file__).parents[1],
                env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1"))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            lines = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            per_task = {line["task_id"]: line for line in lines if "task_id" in line}
            self.assertEqual(set(per_task), {0, 66})
            for line in per_task.values():
                self.assertEqual(len(line["reference_l1_per_action"]), 23)
                self.assertTrue(np.isfinite(line["validation_l1_per_action"]).all())
            saved = torch.load(out / "step_2.pt", map_location="cpu", weights_only=True)
            self.assertTrue(saved["training"]["task_balanced"])
            self.assertEqual(saved["step"], 2)
            self.assertIn("reference_normalization", saved)


if __name__ == "__main__":
    unittest.main()
