import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from turbovla.data.preprocessing import (
    CAMERAS, Normalizer, action_chunk, depth_metres, preprocess_views,
)


class BehaviorContractTests(unittest.TestCase):
    def test_prepared_episode_to_training_batch(self):
        from turbovla.data.dataset import BehaviorDataset, training_normalizer

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.npz"
            actions = np.arange(69, dtype=np.float32).reshape(3, 23)
            np.savez(path, rgb=np.zeros((3, 3, 16, 16, 3), dtype=np.uint8),
                     states=np.ones((3, 61), dtype=np.float32), actions=actions,
                     instruction="move to the table", episode_id=0, fps=30,
                     cameras=np.asarray(CAMERAS))
            norm = training_normalizer([path])
            dataset = BehaviorDataset([path], norm, dict(size=16, mean=[0]*3, std=[1]*3), horizon=4)
            sample = dataset[2]
            self.assertEqual(sample["images"].shape, (3, 3, 16, 16))
            self.assertEqual(sample["state"].shape, (61,))
            self.assertEqual(sample["actions"].shape, (4, 23))
            np.testing.assert_array_equal(sample["mask"], [1, 0, 0, 0])
            np.testing.assert_allclose(norm.inverse_action(sample["actions"])[0], actions[-1])

    def test_registry_auto_discovery_module_has_no_invalid_import(self):
        path = Path(__file__).parents[1] / "experiments/behavior_2026/data_registry/data_config.py"
        namespace = {}
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        self.assertEqual(namespace["ROBOT_TYPE_CONFIG_MAP"], {})

    def test_rgba_alpha_is_not_depth(self):
        rgb = np.full((16, 16, 3), 128, dtype=np.uint8)
        rgba = np.concatenate([rgb, np.full((16, 16, 1), 255, dtype=np.uint8)], axis=-1)
        kwargs = dict(size=16, mean=[0.5]*3, std=[0.5]*3)
        np.testing.assert_array_equal(preprocess_views([rgb]*3, **kwargs), preprocess_views([rgba]*3, **kwargs))
        self.assertEqual(preprocess_views([rgba]*3, **kwargs).shape, (3, 3, 16, 16))

    def test_separate_depth_encoding(self):
        codes = np.array([[0, 4095]], dtype=np.float32)
        np.testing.assert_allclose(depth_metres(codes, "log12"), [[0.01, 10]], atol=1e-5)
        np.testing.assert_allclose(depth_metres([[5000]], "mm"), [[5]])
        np.testing.assert_allclose(depth_metres([[5]], "m"), [[5]])
        with self.assertRaises(ValueError):
            depth_metres([[5]], "auto")
        with self.assertRaises(ValueError):
            depth_metres([[4096]], "log12")

    def test_rgbd_shape_and_scale(self):
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        depth = np.full((16, 16), 5000)
        result = preprocess_views([rgb]*3, size=16, mean=[0]*3, std=[1]*3,
                                  depth=[depth]*3, depth_encoding="mm")
        self.assertEqual(result.shape, (3, 4, 16, 16))
        np.testing.assert_allclose(result[:, 3], 0.5)

    def test_invalid_alignment(self):
        with self.assertRaises(ValueError):
            preprocess_views([np.zeros((16, 16, 3), dtype=np.uint8)]*3,
                             size=16, mean=[0]*3, std=[1]*3,
                             depth=[np.ones((8, 8))]*3, depth_encoding="m")

    def test_whole_body_roundtrip_including_constants(self):
        rng = np.random.default_rng(1)
        actions = rng.normal(size=(50, 23)).astype(np.float32)
        actions[:, 7] = 0.25
        normalizer = Normalizer.fit(rng.normal(size=(50, 61)), actions)
        np.testing.assert_allclose(normalizer.inverse_action(normalizer.action(actions)), actions, atol=1e-6)
        self.assertTrue(np.all(normalizer.action(actions)[:, 7] == 0))

    def test_chunk_cannot_cross_episode(self):
        actions = np.arange(69, dtype=np.float32).reshape(3, 23)
        chunk, mask = action_chunk(actions, 2, 4)
        np.testing.assert_array_equal(mask, [1, 0, 0, 0])
        np.testing.assert_array_equal(chunk, np.repeat(actions[-1:], 4, axis=0))

    def test_invalid_state_is_rejected(self):
        norm = Normalizer.fit(np.ones((2, 61)), np.ones((2, 23)))
        for invalid in (np.zeros(32), np.full(61, np.nan)):
            with self.assertRaises(ValueError):
                norm.state(invalid)

    def test_rgb_defaults_use_behavior_interface(self):
        # Load the pure configuration module without importing Torch.
        path = Path(__file__).parents[1] / "turbovla/models/configuration.py"
        spec = importlib.util.spec_from_file_location("contract_config", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        config = module.TurboVLAConfig()
        self.assertEqual((config.action.action_dim, config.action.state_dim, config.vision.input_channels), (23, 61, 3))
        behavior = module.TurboVLAConfig.from_mapping({"vision": {"num_views": 3}, "action": {"action_dim": 23, "state_dim": 61}})
        self.assertEqual((behavior.action.action_dim, behavior.action.state_dim, behavior.vision.num_views), (23, 61, 3))


if __name__ == "__main__":
    unittest.main()
