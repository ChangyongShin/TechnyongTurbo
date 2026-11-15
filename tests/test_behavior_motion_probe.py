import unittest
import numpy as np
from turbovla.evaluation.motion_probe import select_phases, probe


class ImagePolicy:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def infer(self, rgb, state, instruction):
        action = np.zeros(23, dtype=np.float32)
        action[0] = rgb[0, 0, 0, 0] / 255
        return {"action": action}


class MotionProbeTests(unittest.TestCase):
    def episode(self):
        rgb = np.zeros((8, 3, 4, 4, 3), dtype=np.uint8)
        for t in range(8):
            rgb[t] = t * 30
        actions = np.zeros((8, 23), dtype=np.float32)
        actions[:3, 0] = .2
        actions[2:4, 2] = .1
        actions[4:6, 7] = .05
        return {"episode_id": 0, "task_id": 0, "rgb": rgb,
                "actions": actions, "states": np.zeros((8, 61), dtype=np.float32),
                "instruction": "turn on radio", "modality": "rgb"}

    def test_phase_selection_includes_motion_and_preserves_overlap(self):
        selected, counts = select_phases(self.episode(), 2)
        self.assertEqual(counts, {"moving_xy": 3, "turning_yaw": 2,
                                  "arm_target_change": 2, "quiet_commands": 2})
        self.assertEqual(selected["moving_xy"], [0, 2])
        self.assertIn(2, selected["turning_yaw"])
        with self.assertRaises(ValueError):
            select_phases(self.episode(), 0)

    def test_fixed_state_image_conditions_metrics_and_reset(self):
        policy = ImagePolicy()
        result = probe(policy, [self.episode()], 2)
        self.assertEqual(policy.resets, len(result["records"]) * 3)
        r = next(r for r in result["records"] if r["frame"] == 2)
        self.assertEqual(r["expert_action"][0], float(np.float32(.2)))
        self.assertNotEqual(r["predictions"]["normal_rgb"], r["predictions"]["black_rgb"])
        self.assertEqual(result["phase_summary"]["moving_xy"]["samples"], 2)
        self.assertAlmostEqual(result["phase_summary"]["moving_xy"]["expert_xy_command_norm_mean"], .2)
        summary = result["phase_summary"]["moving_xy"]["raw_controller_l1"]["black_rgb"]
        self.assertEqual(summary["moving_frames_predicted_below_0_01_fraction"], 1.)

    def test_empty_motion_has_no_fabricated_statistics(self):
        ep = self.episode()
        ep["actions"][:] = 0
        result = probe(ImagePolicy(), [ep], 2)
        self.assertEqual(result["phase_summary"]["moving_xy"]["samples"], 0)
        self.assertNotIn("raw_controller_l1", result["phase_summary"]["moving_xy"])


if __name__ == "__main__":
    unittest.main()
