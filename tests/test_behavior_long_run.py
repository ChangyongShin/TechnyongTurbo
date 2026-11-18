import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


path = Path(__file__).parents[1] / 'scripts/behavior_2026/train_task0_long.py'
spec = importlib.util.spec_from_file_location('task0_long_test', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def checkpoint():
    return {'step': 12048, 'optimizer': {}, 'seed': 42,
            'training': {'batch_size': 32, 'grad_accumulation': 1, 'lr': 1e-4, 'workers': 0},
            'preprocessing': {'size': 224},
            'model_config': {'action': {'action_dim': 23, 'state_dim': 61, 'horizon': 12},
                             'vision': {'num_views': 3, 'frozen': True, 'model_name_or_path': '/models/dino',
                                        'compute_precision': 'bf16_autocast'},
                             'text': {'model_name_or_path': '/models/bert'}},
            'train_episode_ids': list(range(180)), 'validation_episode_ids': list(range(180, 200))}


class LongRunTests(unittest.TestCase):
    def test_cache_identity_and_mappings_closed_on_success_or_failure(self):
        folder = SimpleNamespace(exists=lambda: True)
        for task_id in (0, 1):
            arrays = {key: SimpleNamespace(shape=(2, 3, 224, 224, 3), _mmap=SimpleNamespace(close=Mock()))
                      for key in ('rgb', 'states', 'actions')}
            episode = {**arrays, 'episode_id': 0, 'task_id': task_id, 'instruction': 'turning_on_radio',
                       'source': {'revision': 'abc'}, 'modality': 'rgb'}
            loader = lambda _: episode
            if task_id == 0:
                self.assertTrue(runner.check_cache(folder, 0, {'revision': 'abc'}, 'turning_on_radio', loader))
            else:
                with self.assertRaises(ValueError):
                    runner.check_cache(folder, 0, {'revision': 'abc'}, 'turning_on_radio', loader)
            for array in arrays.values(): array._mmap.close.assert_called_once()
        self.assertFalse(runner.check_cache(SimpleNamespace(exists=lambda: False), 0, {}, 'test', Mock()))

    def args(self):
        return SimpleNamespace(cache=Path('/cache'), checkpoint=Path('/original/step_12048.pt'),
                               out=Path('/new/checkpoints'), steps=100000, save_every=5000,
                               dinov3=None, bert=None, wandb_mode='online')

    def test_resume_preserves_training_settings_and_total_step(self):
        saved = checkpoint()
        settings = runner.baseline_settings(saved, 100000)
        self.assertEqual(settings['additional_steps'], 87952)
        self.assertEqual(settings['batch_size'], 32)
        command = runner.training_command(self.args(), saved, settings, '--augmentation {none,color}')
        for key, value in [('--resume', str(self.args().checkpoint)), ('--steps', '100000'),
                           ('--batch-size', '32'), ('--lr', '0.0001'), ('--augmentation', 'none'),
                           ('--grad-accumulation', '1'), ('--seed', '42')]:
            self.assertEqual(command[command.index(key)+1], value)
        self.assertNotIn('--staged', command)
        self.assertNotIn('--warm-start', command)
        self.assertNotIn('--initial-weights', command)
        train = command[command.index('--train')+1:command.index('--validation')]
        val = command[command.index('--validation')+1:command.index('--dinov3')]
        self.assertEqual(len(train), 180)
        self.assertEqual(len(val), 20)
        self.assertFalse(set(train) & set(val))

    def test_supported_on_preaugmentation_trainer(self):
        saved = checkpoint()
        command = runner.training_command(self.args(), saved, runner.baseline_settings(saved, 100000), '')
        self.assertNotIn('--augmentation', command)

    def test_rejects_staged_and_changed_objectives(self):
        saved = checkpoint()
        saved['model_config']['action']['decision_conditioning'] = True
        with self.assertRaisesRegex(ValueError, 'staged'):
            runner.baseline_settings(saved, 100000)
        for key in ('task_balanced', 'phase_balanced', 'group_balanced_loss', 'temporal_weight', 'decision_pretrain_steps'):
            saved = checkpoint()
            saved['training'][key] = 1
            with self.assertRaises(ValueError): runner.baseline_settings(saved, 100000)
        saved = checkpoint()
        saved['training']['augmentation'] = 'color'
        with self.assertRaisesRegex(ValueError, 'unaugmented'): runner.baseline_settings(saved, 100000)

    def test_rejects_missing_optimizer_or_overlapping_split_or_finished_target(self):
        saved = checkpoint()
        del saved['optimizer']
        with self.assertRaisesRegex(ValueError, 'Optimizer'): runner.baseline_settings(saved, 100000)
        saved = checkpoint()
        saved['validation_episode_ids'][0] = 0
        with self.assertRaisesRegex(ValueError, 'split'): runner.baseline_settings(saved, 100000)
        with self.assertRaisesRegex(ValueError, 'exceed'): runner.baseline_settings(checkpoint(), 12048)


if __name__ == '__main__':
    unittest.main()
