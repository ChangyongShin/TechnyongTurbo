import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turbovla.data.startup import StartupSampler, StartupMetrics, startup_frames, startup_base_l1, startup_report
from turbovla.data.dataset import BehaviorDataset
from turbovla.data.preprocessing import CAMERAS, Normalizer


class Dataset:
    def __init__(self):
        self.labels = [np.r_[np.zeros(15, dtype=int), np.ones(15, dtype=int), np.full(9970, 3)],
                       np.full(10000, 2, dtype=int)]
        self.episodes = [{"episode_id": 0, "task_id": 0}, {"episode_id": 1, "task_id": 1}]
    def __len__(self):
        return sum(map(len, self.labels))


class StartupTests(unittest.TestCase):
    def test_dataset_frame_index_restarts_at_episode_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for eid in (10, 11):
                path = Path(directory) / f'{eid}.npz'
                np.savez(path, rgb=np.zeros((2, 3, 16, 16, 3), np.uint8),
                         states=np.zeros((2, 61)), actions=np.zeros((2, 23)),
                         episode_id=eid, task_id=0, fps=30, instruction='test', cameras=CAMERAS)
                paths.append(path)
            normalizer = Normalizer(np.zeros(61), np.ones(61), np.zeros(23), np.ones(23))
            data = BehaviorDataset(paths, normalizer, {'size': 16, 'mean': [0]*3, 'std': [1]*3}, horizon=2)
            self.assertEqual([(data[i]['episode_id'], data[i]['frame_index']) for i in range(4)],
                             [(10, 0), (10, 1), (11, 0), (11, 1)])
            np.testing.assert_array_equal(data[1]['mask'], [1, 0])

    def test_launcher_forwards_startup_options_without_running_training(self):
        path = Path(__file__).parents[1] / 'scripts/behavior_2026/train_staged.py'
        spec = importlib.util.spec_from_file_location('startup_launcher_test', path)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'baseline.json'
            config.write_text(json.dumps({'train_episode_ids': [0], 'validation_episode_ids': [1]}))
            for eid in (0, 1):
                cache = root / f'episode_{eid:06d}'
                cache.mkdir()
                (cache / 'metadata.json').write_text(json.dumps({'episode_id': eid, 'task_id': 0}))
            argv = ['train_staged.py', '--cache', str(root), '--baseline-config', str(config),
                    '--out', str(root/'out'), '--startup-fraction', '.25', '--startup-seconds', '10',
                    '--startup-base-weight', '.2']
            with patch('sys.argv', argv), patch.object(launcher.subprocess, 'run') as run, patch('builtins.print'):
                launcher.main()
            command = run.call_args.args[0]
            for key, value in [('--startup-fraction', '0.25'), ('--startup-seconds', '10.0'), ('--startup-base-weight', '0.2')]:
                self.assertEqual(command[command.index(key)+1], value)

    def test_startup_mix_preserves_background_and_nonmoving_episode(self):
        dataset = Dataset()
        draws = np.array(list(StartupSampler(dataset, range(len(dataset)), 1, .4, seed=1)))
        self.assertEqual(len(draws), len(dataset))
        self.assertTrue(((draws >= 0) & (draws < len(dataset))).all())
        early = (draws % 10000) < 30
        self.assertAlmostEqual(float(early.mean()), .4 + .6 * .003, delta=.02)
        self.assertAlmostEqual(float((draws[early] >= 10000).mean()), .5, delta=.03)
        # The manipulation-only episode is retained; no motion labels injected.
        self.assertTrue(np.all(dataset.labels[1][draws[early & (draws >= 10000)] - 10000] == 2))
        unchanged = list(StartupSampler(dataset, range(len(dataset)), 1, 0))
        self.assertEqual(unchanged, list(range(len(dataset))))

    def test_short_episodes_phase_balance_and_seed(self):
        dataset = Dataset()
        dataset.labels = [np.array([0] + [3] * 29), np.array([2])]
        a = StartupSampler(dataset, range(31), 10, .9, seed=7, task_balanced=True)
        b = StartupSampler(dataset, range(31), 10, .9, seed=7, task_balanced=True)
        self.assertEqual(list(a), list(b))
        self.assertEqual(list(a), list(b))
        report = startup_report(dataset, 10)
        self.assertEqual(report['episodes'][1]['frames'], 1)
        self.assertIsNone(report['episodes'][1]['first_motion_seconds'])
        self.assertEqual(report['episodes'][0]['first_motion_seconds'], 0)

    def test_startup_loss_targets_first_base_only_and_masks_tail(self):
        prediction = torch.ones(3, 2, 23, requires_grad=True)
        target = torch.zeros_like(prediction)
        target = target.detach()
        target[0, 0, :3] = .5
        mask = torch.tensor([[1., 1.], [1., 1.], [0., 0.]])
        value = startup_base_l1(prediction, target, mask, torch.tensor([0, 300, 2]), 300)
        self.assertAlmostEqual(float(value.detach()), .5)
        value.backward()
        self.assertTrue(torch.all(prediction.grad[0, 0, :3] > 0))
        self.assertEqual(torch.count_nonzero(prediction.grad).item(), 3)
        empty = startup_base_l1(prediction, target, mask, torch.tensor([300, 301, 302]), 300)
        self.assertEqual(float(empty.detach()), 0)

    def test_metrics_detects_false_motion_and_missed_early_motion(self):
        target = np.zeros((4, 23)); prediction = target.copy()
        target[0, 0] = .1; target[1, 2] = .2
        prediction[0, 0] = .1; prediction[2, 2] = .2
        metrics = StartupMetrics()
        self.assertEqual(metrics.result(), {"samples": 0})
        metrics.update(prediction[:2], target[:2]); metrics.update(prediction[2:], target[2:])
        result = metrics.result()
        self.assertEqual(result['motion_confusion'], [[1, 1], [1, 1]])
        self.assertEqual(result['motion_recall'], .5)
        self.assertEqual(result['false_motion_rate'], .5)
        self.assertAlmostEqual(result['base_raw_l1_per_axis'][2], .1)
        still = StartupMetrics(); still.update(np.zeros((1, 23)), np.zeros((1, 23)))
        self.assertIsNone(still.result()['motion_recall'])

    def test_invalid_options(self):
        for seconds in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): startup_frames(seconds)
        for fraction in (-.1, 1, float('nan')):
            with self.assertRaises(ValueError): StartupSampler(Dataset(), None, 1, fraction)
        self.assertEqual(startup_frames(.01), 1)


if __name__ == '__main__':
    unittest.main()
