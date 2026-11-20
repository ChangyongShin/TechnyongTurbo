"""Real tiny local-model smoke test: strict transfer, fresh optimizer and resume."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from turbovla.data.preprocessing import CAMERAS


class InterventionTrainingTests(unittest.TestCase):
    def test_finetune_and_resume_keep_contract_and_reset_only_requested_optimizer(self):
        from transformers import (BertConfig, BertModel, BertTokenizerFast,
                                  DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessorFast)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vision, text = root/'vision', root/'text'
            vision.mkdir()
            text.mkdir()
            DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, image_size=16, patch_size=16,
                num_register_tokens=4)).save_pretrained(vision)
            DINOv3ViTImageProcessorFast().save_pretrained(vision)
            BertModel(BertConfig(vocab_size=10, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4)).save_pretrained(text)
            (text/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nturn\non\nradio\n.\n?\n')
            BertTokenizerFast(vocab_file=str(text/'vocab.txt')).save_pretrained(text)
            paths = []
            for eid in (0, 1):
                path = root/f'episode_{eid:06d}'
                path.mkdir()
                actions = np.zeros((40, 23), np.float32)
                actions[20:30, 0] = .2
                states = np.zeros((40, 61), np.float32)
                states[21:31, 0] = .1
                np.save(path/'rgb.npy', np.full((40, 3, 16, 16, 3), 128, np.uint8))
                np.save(path/'states.npy', states)
                np.save(path/'actions.npy', actions)
                (path/'metadata.json').write_text(json.dumps(dict(episode_id=eid, task_id=0, fps=30,
                    cameras=list(CAMERAS), instruction='turning_on_radio', modality='rgb', source={'revision': 'local-test'})))
                paths.append(path)
            common = [sys.executable, '-m', 'turbovla.training.trainer', '--train', str(paths[0]),
                      '--validation', str(paths[1]), '--dinov3', str(vision), '--bert', str(text),
                      '--batch-size', '4', '--horizon', '2', '--image-size', '16', '--device', 'cpu',
                      '--vision-precision', 'fp32', '--wandb-mode', 'disabled', '--staged',
                      '--phase-balanced', '--group-balanced-loss', '--temporal-weight', '.1']
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1')

            def run(extra, succeeds=True):
                result = subprocess.run(common+extra, cwd=Path(__file__).parents[1], env=env,
                                        capture_output=True, text=True, timeout=180)
                if succeeds:
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result

            source_dir = root/'source'
            run(['--out', str(source_dir), '--steps', '2'])
            source = source_dir/'step_2.pt'
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            interventions = ['--transition-fraction', '.25', '--velocity-mask-probability', '.2',
                             '--visual-augmentation', 'mild']
            candidate_dir = root/'candidate'
            run(['--out', str(candidate_dir), '--steps', '1', '--lr', '3e-5', '--finetune', str(source)]+interventions)
            candidate_path = candidate_dir/'step_1.pt'
            candidate = torch.load(candidate_path, map_location='cpu', weights_only=True)
            original = torch.load(source, map_location='cpu', weights_only=True)
            self.assertEqual(candidate['step'], 1)
            self.assertEqual(candidate['finetune_source']['step'], 2)
            self.assertEqual(candidate['finetune_source']['sha256'], digest)
            self.assertEqual(candidate['normalization'], original['normalization'])
            self.assertEqual(candidate['train_episode_ids'], original['train_episode_ids'])
            self.assertEqual(candidate['model_config'], original['model_config'])
            self.assertEqual(candidate['training']['interventions']['velocity_mask_probability'], .2)
            self.assertTrue(all(int(v['step']) == 1 for v in candidate['optimizer']['state'].values()))
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            probe_path = root/'departure-report.json'
            probe = subprocess.run([sys.executable, '-m', 'turbovla.evaluation.departure_probe',
                '--checkpoint', str(candidate_path), '--cache', str(root), '--out', str(probe_path),
                '--device', 'cpu', '--max-events', '1'], cwd=Path(__file__).parents[1], env=env,
                capture_output=True, text=True, timeout=180)
            self.assertEqual(probe.returncode, 0, probe.stdout+probe.stderr)
            report = json.loads(probe_path.read_text())
            self.assertEqual(report['summary']['move']['events'], 1)
            self.assertIsNone(report['closed_loop_success'])
            self.assertEqual(report['checkpoint']['checkpoint_sha256'], hashlib.sha256(candidate_path.read_bytes()).hexdigest())
            resumed_dir = root/'resumed'
            run(['--out', str(resumed_dir), '--steps', '2', '--lr', '3e-5', '--resume', str(candidate_path)]+interventions)
            resumed = torch.load(resumed_dir/'step_2.pt', map_location='cpu', weights_only=True)
            self.assertEqual(resumed['finetune_source'], candidate['finetune_source'])
            self.assertTrue(all(int(v['step']) == 2 for v in resumed['optimizer']['state'].values()))
            changed = interventions.copy()
            changed[changed.index('--velocity-mask-probability')+1] = '.3'
            rejected = run(['--out', str(root/'rejected'), '--steps', '2', '--resume', str(candidate_path)]+changed, False)
            self.assertIn('Resume TRAIN interventions changed', rejected.stderr)


if __name__ == '__main__':
    unittest.main()
