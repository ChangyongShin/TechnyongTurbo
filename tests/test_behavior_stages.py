import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from turbovla.data.stages import phase_labels, PhaseBalancedSampler
from turbovla.data.preprocessing import CAMERAS


class StageTests(unittest.TestCase):
    def test_motion_labels_priority_and_no_episode_gripper_carryover(self):
        a, s = np.zeros((4, 23)), np.zeros((4, 61))
        a[0, 0] = .2
        a[0, 2] = .1
        a[1, 2] = .1
        a[2, 7] = .1
        a[:, 14] = 1
        np.testing.assert_array_equal(phase_labels({"actions": a, "states": s}), [0, 1, 2, 3])

    def test_sampler_equalizes_tasks_and_available_phases(self):
        dataset = SimpleNamespace(episodes=[{"task_id": 0}, {"task_id": 66}],
                                  labels=[np.r_[np.zeros(1000, dtype=int), np.full(1000, 3)],
                                          np.r_[np.ones(10000, dtype=int), np.full(10000, 2)]])
        dataset_type = type("Dataset", (), {"__len__": lambda self: 22000})
        data = dataset_type()
        data.episodes, data.labels = dataset.episodes, dataset.labels
        indices = np.array(list(PhaseBalancedSampler(data, task_balanced=True)))
        self.assertAlmostEqual((indices < 2000).mean(), .5, delta=.02)
        for mask in (indices < 1000, (indices >= 1000) & (indices < 2000),
                     (indices >= 2000) & (indices < 12000), indices >= 12000):
            self.assertAlmostEqual(mask.mean(), .25, delta=.02)

    def test_real_staged_training_and_serving_checkpoint(self):
        from transformers import (BertConfig, BertModel, BertTokenizerFast,
                                  DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessorFast)
        from turbovla.evaluation.policy import BehaviorPolicy
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vision, text = root / "vision", root / "text"
            vision.mkdir(); text.mkdir()
            DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, image_size=16, patch_size=16,
                num_register_tokens=4)).save_pretrained(vision)
            DINOv3ViTImageProcessorFast().save_pretrained(vision)
            BertModel(BertConfig(vocab_size=10, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4)).save_pretrained(text)
            (text / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nturn\non\nradio\n.\n?\n")
            BertTokenizerFast(vocab_file=str(text / "vocab.txt")).save_pretrained(text)
            paths=[]
            for eid in (0, 1):
                path=root/f"episode_{eid}.npz"
                actions=np.zeros((4,23),np.float32)
                actions[0,0]=.2;actions[1,2]=.1;actions[2,7]=.1
                np.savez(path, rgb=np.full((4,3,16,16,3),128,np.uint8),
                    states=np.zeros((4,61),np.float32), actions=actions,
                    episode_id=eid, task_id=0, fps=30, instruction="turn on radio", cameras=np.asarray(CAMERAS))
                paths.append(path)
            out=root/"run"
            command=[sys.executable,"-m","turbovla.training.trainer","--train",str(paths[0]),
                "--validation",str(paths[1]),"--dinov3",str(vision),"--bert",str(text),"--out",str(out),
                "--steps","3","--batch-size","2","--horizon","2","--image-size","16","--device","cpu",
                "--vision-precision","fp32","--wandb-mode","disabled","--staged","--phase-balanced",
                "--group-balanced-loss","--decision-pretrain-steps","1","--temporal-weight","0.1"]
            result=subprocess.run(command,cwd=Path(__file__).parents[1],capture_output=True,text=True,timeout=180,
                env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",HF_HUB_OFFLINE="1"))
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            lines=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            self.assertTrue(any("decision_confusion_matrix" in line for line in lines))
            checkpoint=torch.load(out/"step_3.pt",map_location="cpu",weights_only=True)
            self.assertTrue(checkpoint["model_config"]["action"]["decision_conditioning"])
            self.assertEqual(checkpoint["train_phase_counts"],[1,1,1,1])
            policy=BehaviorPolicy.from_checkpoint(out/"step_3.pt",device="cpu")
            policy.trace_enabled=True
            output=policy.infer(np.full((3,16,16,3),128,np.uint8),np.zeros(61,np.float32),"turn on radio")
            self.assertTrue(np.isfinite(output["action"]).all())
            self.assertEqual(len(policy.last_trace["decision_probabilities"]),4)
            self.assertAlmostEqual(float(policy.last_trace["decision_probabilities"].sum()),1.,places=5)
            # A weight-only continuation starts its own optimizer step count and
            # records the exact source, rather than silently restarting randomly.
            warm_out = root / "warm"
            warm_command = command.copy()
            warm_command[warm_command.index("--out") + 1] = str(warm_out)
            warm_command[warm_command.index("--steps") + 1] = "1"
            warm_command[warm_command.index("--decision-pretrain-steps") + 1] = "0"
            warm_command += ["--warm-start", str(out / "step_3.pt")]
            result = subprocess.run(warm_command, cwd=Path(__file__).parents[1], capture_output=True,
                text=True, timeout=180, env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1"))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            warm = torch.load(warm_out / "step_1.pt", map_location="cpu", weights_only=True)
            self.assertEqual(warm["warm_start"]["source_step"], 3)
            self.assertFalse(warm["warm_start"]["optimizer_resumed"])
            self.assertEqual(warm["warm_start"]["initialized_keys"], [])


if __name__ == "__main__":
    unittest.main()
