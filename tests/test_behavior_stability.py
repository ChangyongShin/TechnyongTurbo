from copy import deepcopy
import unittest

import numpy as np
import torch

from turbovla.evaluation.temporal import JointTemporalEnsemble
from turbovla.models.configuration import TurboVLAConfig
from turbovla.training.warm_start import load_policy_warm_start


class StabilityTests(unittest.TestCase):
    def test_ensemble_aligns_time_and_preserves_base_and_grippers(self):
        ensemble = JointTemporalEnsemble(3, window=3, decay=0)
        first = np.repeat(np.array([10, 20, 30], dtype=np.float32)[:, None], 23, axis=1)
        second = first + 100
        third = first + 200
        np.testing.assert_array_equal(ensemble.update(first), first[0])
        result = ensemble.update(second)
        self.assertEqual(result[7], (20 + 110) / 2)
        np.testing.assert_array_equal(result[[0, 1, 2, 14, 22]], second[0, [0, 1, 2, 14, 22]])
        self.assertEqual(ensemble.update(third)[7], (30 + 120 + 210) / 3)
        ensemble.reset()
        np.testing.assert_array_equal(ensemble.update(third), third[0])

    def test_ensemble_decay_and_invalid_inputs(self):
        ensemble = JointTemporalEnsemble(2, window=2, decay=1)
        ensemble.update(np.zeros((2, 23)))
        result = ensemble.update(np.ones((2, 23)))
        self.assertAlmostEqual(float(result[7]), 1 / (1 + np.exp(-1)), places=6)
        for window, decay in ((0, 1), (3, 1), (2, -1), (2, float('nan'))):
            with self.assertRaises(ValueError):
                JointTemporalEnsemble(2, window, decay)
        with self.assertRaises(ValueError):
            ensemble.update(np.full((2, 23), np.nan))

    def test_warm_start_checks_contract_before_transferring_all_core_weights(self):
        class Model(torch.nn.Module):
            def __init__(self, decision=False):
                super().__init__()
                self.core = torch.nn.Linear(2, 2)
                if decision:
                    self.decision_head = torch.nn.Linear(2, 4)
                    self.decision_embedding = torch.nn.Linear(4, 2, bias=False)
        source, target = Model(), Model(True)
        metadata = {key: [] for key in ('normalization', 'preprocessing', 'action_order',
                    'train_episode_ids', 'validation_episode_ids', 'data_sources')}
        metadata['model_config'] = TurboVLAConfig().to_dict()
        checkpoint = {**deepcopy(metadata), 'model': source.state_dict(), 'step': 12048}
        metadata['model_config']['action']['decision_conditioning'] = True
        result = load_policy_warm_start(target, checkpoint, metadata)
        torch.testing.assert_close(source.core.weight, target.core.weight)
        self.assertEqual(set(result['initialized_keys']),
                         {'decision_head.weight', 'decision_head.bias', 'decision_embedding.weight'})
        self.assertTrue(result['decision_conditioning_changed'])
        self.assertFalse(result['optimizer_resumed'])
        for key in ('normalization', 'action_order', 'validation_episode_ids', 'data_sources'):
            bad = deepcopy(checkpoint)
            bad[key] = ['changed']
            with self.assertRaisesRegex(ValueError, key):
                load_policy_warm_start(target, bad, metadata)
        bad = deepcopy(checkpoint)
        del bad['model']['core.weight']
        with self.assertRaisesRegex(ValueError, 'Incomplete policy transfer'):
            load_policy_warm_start(target, bad, metadata)


if __name__ == '__main__':
    unittest.main()
