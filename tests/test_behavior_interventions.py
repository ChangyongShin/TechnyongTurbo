import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from turbovla.data.interventions import (departure_events, mask_base_velocity,
                                         mild_photometric, TransitionMixtureSampler)
from turbovla.data.preprocessing import CAMERAS, Normalizer
from turbovla.data.dataset import BehaviorDataset, training_normalizer
from turbovla.training.transfer import verify_finetune_contract


class InterventionTests(unittest.TestCase):
    def test_departures_include_ramps_and_yaw_but_not_jitter_or_initial_motion(self):
        actions = np.zeros((150, 23), np.float32)
        actions[:10, 0] = .2  # No preceding rest; not a departure label.
        actions[30:34, 0] = [.015, .025, .04, .06]
        actions[34:45, 0] = .2
        actions[70:80, 2] = -.1
        actions[110:115, 0] = .02  # Jitter never reaches .05.
        events = departure_events(actions)
        self.assertEqual([(e['frame'], e['kind']) for e in events], [(30, 'move'), (70, 'turn')])
        self.assertEqual((events[0]['start'], events[0]['stop']), (15, 46))
        self.assertEqual(departure_events(np.zeros((12, 23))), [])

    def test_velocity_mask_is_raw_physical_zero_and_preserves_other_state(self):
        state = np.arange(61, dtype=np.float32) + 1
        changed = mask_base_velocity(state, 1.)
        np.testing.assert_array_equal(changed[:3], 0)
        np.testing.assert_array_equal(changed[3:], state[3:])
        np.testing.assert_array_equal(state, np.arange(61) + 1)
        normalizer = Normalizer(np.ones(61), np.full(61, 2), np.zeros(23), np.ones(23))
        np.testing.assert_array_equal(normalizer.state(changed)[:3], -.5)
        np.testing.assert_array_equal(mask_base_velocity(state, 0.), state)

    def test_photo_preserves_view_order_geometry_dtype_and_input(self):
        rgb = np.zeros((3, 16, 16, 3), np.uint8)
        rgb[:, 4:8, 5:9] = [200, 50, 20]
        original = rgb.copy()
        photo = mild_photometric(rgb, np.random.default_rng(2))
        np.testing.assert_array_equal(rgb, original)
        self.assertEqual(photo.dtype, np.uint8)
        self.assertEqual(photo.shape, rgb.shape)
        np.testing.assert_array_equal(photo[0], photo[1])
        np.testing.assert_array_equal(photo[1], photo[2])
        self.assertGreater(float(photo[0, 5, 6].mean()), float(photo[0, 0, 0].mean()))

    def test_sampler_mixture_keeps_original_draws_and_respects_episode_bounds(self):
        class Dataset:
            def __len__(self): return 10000
        data = Dataset()
        data.episodes = []
        for size in (100, 9900):
            actions = np.zeros((size, 23), np.float32)
            actions[20:40, 0] = .2
            data.episodes.append({'actions': actions})
        unchanged = TransitionMixtureSampler(data, range(10000), 0.)
        self.assertEqual(list(unchanged), list(range(10000)))
        sampler = TransitionMixtureSampler(data, range(10000), 1., seed=7)
        draws = np.array(list(sampler))
        self.assertAlmostEqual(float((draws < 100).mean()), .5, delta=.025)
        self.assertTrue(np.isin(draws, np.r_[5:36, 105:136]).all())
        self.assertEqual(list(TransitionMixtureSampler(data, range(10000), 1., seed=7)), draws.tolist())
        mixed = TransitionMixtureSampler(data, range(10000), .25, seed=7)
        indices = np.array(list(mixed))
        replaced = (indices != np.arange(10000)).mean()
        self.assertAlmostEqual(float(replaced), .25, delta=.025)

    def test_no_departure_data_fails_instead_of_silently_sampling_something_else(self):
        class Dataset:
            episodes = [{'actions': np.zeros((60, 23))}]
            def __len__(self): return 60
        with self.assertRaisesRegex(ValueError, 'No departure'):
            TransitionMixtureSampler(Dataset(), None, .25)

    def test_dataset_train_only_corruption_does_not_change_labels_cache_or_normalizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'episode.npz'
            states = np.ones((40, 61), np.float32)
            states[:, 0] = np.linspace(0, .2, 40)
            actions = np.zeros((40, 23), np.float32)
            actions[20:, 0] = .2
            np.savez(path, rgb=np.full((40, 3, 16, 16, 3), 128, np.uint8), states=states,
                     actions=actions, episode_id=0, task_id=0, fps=30,
                     cameras=np.asarray(CAMERAS), instruction='turning_on_radio')
            norm = training_normalizer([path])
            prep = dict(size=16, mean=[.5]*3, std=[.5]*3)
            val = BehaviorDataset([path], norm, prep, horizon=3)
            train = BehaviorDataset([path], norm, prep, horizon=3,
                                    velocity_mask_probability=1., visual_augmentation='mild')
            target_before = norm.to_dict()
            clean, altered = val[21], train[21]
            np.testing.assert_array_equal(clean['actions'], altered['actions'])
            np.testing.assert_array_equal(clean['raw_actions'], altered['raw_actions'])
            np.testing.assert_array_equal(clean['mask'], altered['mask'])
            np.testing.assert_array_equal(clean['state'][3:], altered['state'][3:])
            np.testing.assert_array_equal(altered['state'][:3], norm.state(np.r_[np.zeros(3), states[21, 3:]])[:3])
            self.assertFalse(np.array_equal(clean['state'][:3], altered['state'][:3]))
            self.assertEqual(norm.to_dict(), target_before)
            np.testing.assert_array_equal(val.episodes[0]['states'], states)

    def test_strict_finetune_rejects_changed_units_split_order_and_model(self):
        from copy import deepcopy
        meta = dict(normalization={'range': [1, 2]}, preprocessing={'size': 224}, train_episode_ids=[0],
                    validation_episode_ids=[1], data_sources=[{'revision': 'fixed'}], action_order={'base': [0, 3]},
                    modality='rgb', phase_spec={'version': 1},
                    model_config={'vision': {'model_name_or_path': '/new/vision', 'frozen': True},
                                  'text': {'model_name_or_path': '/text'}, 'action': {'horizon': 12}})
        saved = deepcopy(meta)
        saved['model_config']['vision']['model_name_or_path'] = '/old/vision'
        verify_finetune_contract(saved, meta)
        for key in ('normalization', 'preprocessing', 'train_episode_ids', 'validation_episode_ids',
                    'data_sources', 'action_order', 'modality', 'phase_spec', 'model_config'):
            changed = deepcopy(saved)
            changed[key] = 'different'
            with self.assertRaises((ValueError, TypeError)):
                verify_finetune_contract(changed, meta)

    def test_pilot_arms_use_same_source_fresh_optimizer_budget_and_saved_objectives(self):
        root = Path(__file__).parents[1]
        spec = importlib.util.spec_from_file_location('pilot_test', root/'scripts/behavior_2026/train_task0_interventions.py')
        pilot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pilot)
        saved = dict(step=100000, optimizer={}, seed=42, train_episode_ids=list(range(180)),
                     validation_episode_ids=list(range(180, 200)), preprocessing={'size': 224},
                     model_config={'action': {'action_dim': 23, 'state_dim': 61, 'horizon': 12, 'decision_conditioning': True},
                                   'vision': {'num_views': 3, 'frozen': True, 'compute_precision': 'bf16_autocast',
                                              'model_name_or_path': '/vision'}, 'text': {'model_name_or_path': '/bert'}},
                     training={'batch_size': 64, 'lr': 1e-4, 'grad_accumulation': 1, 'phase_balanced': True,
                               'group_balanced_loss': True, 'decision_weight': .1, 'temporal_weight': .1,
                               'decision_pretrain_steps': 100})
        help_text = '--finetune --staged --phase-balanced --group-balanced-loss --decision-weight --temporal-weight --decision-pretrain-steps'
        for mode, (fraction, probability, photo) in pilot.MODES.items():
            args = SimpleNamespace(mode=mode, checkpoint=Path('/source/step_100000.pt'), out=Path('/new/checkpoints'),
                                   cache=Path('/cache'), steps=10000, lr=3e-5, save_every=1000,
                                   dinov3=None, bert=None, wandb_mode='disabled')
            command = pilot.pilot_command(args, saved, help_text)
            self.assertNotIn('--resume', command)
            for key, value in [('--finetune', str(args.checkpoint)), ('--steps', '10000'), ('--lr', '3e-05'),
                               ('--decision-pretrain-steps', '0'), ('--transition-fraction', str(fraction)),
                               ('--velocity-mask-probability', str(probability)), ('--visual-augmentation', photo)]:
                self.assertEqual(command[command.index(key)+1], value)

    def test_departure_probe_distinguishes_delay_wrong_direction_and_missing_start(self):
        from turbovla.evaluation.departure_probe import event_result
        expert = np.zeros((31, 23), np.float32)
        expert[2:, 0] = .2
        prediction = np.zeros_like(expert)
        missing = event_result(expert, prediction, 'move')
        self.assertTrue(missing['missed'])
        prediction[5:, 0] = .15
        delayed = event_result(expert, prediction, 'move')
        self.assertEqual(delayed['delay_frames'], 3)
        self.assertTrue(delayed['direction_agrees'])
        prediction[5:, 0] = -.15
        self.assertFalse(event_result(expert, prediction, 'move')['direction_agrees'])
        self.assertIsNone(event_result(expert, prediction, 'turn'))


if __name__ == '__main__':
    unittest.main()
