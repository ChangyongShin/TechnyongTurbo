import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from turbovla.data.preprocessing import CAMERAS

spec = importlib.util.spec_from_file_location("full_prepare", Path(__file__).parents[1] / "scripts/behavior_2026/prepare_task0_full.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FullPrepareTests(unittest.TestCase):
    def test_requires_original_disjoint_full_split(self):
        config = {"train_episode_ids": list(range(180)), "validation_episode_ids": list(range(180,200))}
        self.assertEqual(module.baseline_split(config)[1], list(range(180,200)))
        config["validation_episode_ids"][0] = 0
        with self.assertRaises(ValueError):
            module.baseline_split(config)

    def test_reuse_only_complete_matching_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / "episode_000000"
            self.assertFalse(module.check_cache(p, 0, 1, "rev"))
            p.mkdir()
            with self.assertRaises(ValueError):
                module.check_cache(p, 0, 1, "rev")
            np.save(p / "rgb.npy", np.zeros((1,3,224,224,3),np.uint8))
            np.save(p / "states.npy", np.zeros((1,61),np.float32))
            np.save(p / "actions.npy", np.zeros((1,23),np.float32))
            (p / "metadata.json").write_text(json.dumps({"episode_id":0,"task_id":0,"fps":30,
                "modality":"rgb","cameras":list(CAMERAS),"instruction":"turn on radio","source":{"revision":"rev"}}))
            self.assertTrue(module.check_cache(p, 0, 1, "rev"))
            with self.assertRaises(ValueError):
                module.check_cache(p, 0, 1, "wrong")


if __name__ == "__main__":
    unittest.main()
