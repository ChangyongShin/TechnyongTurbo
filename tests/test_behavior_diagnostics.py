import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from turbovla.data.preprocessing import CAMERAS, Normalizer
from turbovla.evaluation.audit_behavior import audit, pose_hold_action
from turbovla.evaluation.diagnostics import TraceWriter, file_sha256, object_sha256
from turbovla.evaluation.observations import RGB_KEYS, STATE_KEY
from turbovla.evaluation.policy import BehaviorPolicy
from turbovla.evaluation.serve import PolicyServer


class TinyPolicyModel(torch.nn.Module):
    action_dim, state_dim, num_views, chunk_size = 23, 61, 3, 3

    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(vision=SimpleNamespace(input_channels=3))
        self.calls = 0

    def forward(self, instructions, images, state):
        self.calls += 1
        return torch.arange(3, device=images.device)[None, :, None].expand(1, 3, 23) * .2 + self.value


def make_policy(execute_steps=2):
    return BehaviorPolicy(TinyPolicyModel(), Normalizer(np.zeros(61), np.ones(61),
        np.zeros(23), np.ones(23)), {"size": 16, "mean": [.5]*3, "std": [.5]*3}, execute_steps)


class DiagnosticTests(unittest.TestCase):
    def test_queue_trace_reset_and_identity_do_not_change_actions(self):
        a, b = make_policy(), make_policy()
        b.trace_enabled = True
        rgb = np.full((3, 16, 16, 4), 128, dtype=np.uint8)
        state = np.arange(61, dtype=np.float32)
        for expected_index, replanned in ((0, True), (1, False), (0, True)):
            expected = a.infer(rgb, state, "turn on radio")["action"]
            actual = b.infer(rgb, state, "turn on radio")["action"]
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(int(b.last_trace["chunk_index"]), expected_index)
            self.assertEqual(bool(b.last_trace["replanned"]), replanned)
            np.testing.assert_array_equal(actual, b.last_trace["denormalized_chunk"][expected_index])
        b.reset()
        self.assertIsNone(b.last_trace)
        self.assertEqual(len(b.queue), 0)

    def test_trace_bound_and_wire_camera_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace"
            writer = TraceWriter(path, {"test": True}, limit=1)
            policy = make_policy(1)
            server = PolicyServer(policy, {0: "turn on radio"}, trace_writer=writer)
            obs = {key: np.full((16, 16, 4), i * 30, dtype=np.uint8)
                   for i, key in reversed(list(enumerate(RGB_KEYS)))}
            obs[STATE_KEY] = np.zeros(61, dtype=np.float32)
            obs["task_id"] = np.array([0])
            result = server.dispatch(obs)
            with np.load(path / "request_000000.npz", allow_pickle=False) as data:
                self.assertEqual(data["camera_keys"].tolist(), list(RGB_KEYS))
                np.testing.assert_array_equal(data["action"], result["action"])
                self.assertEqual(data["processed_images"].shape, (3, 3, 16, 16))
            server.dispatch(obs)
            self.assertEqual(len(list(path.glob("*.npz"))), 1)
            self.assertFalse(policy.trace_enabled)
            server.dispatch({"reset": True})
            server.dispatch(obs)
            with self.assertRaises(FileExistsError):
                TraceWriter(path, {}, limit=1)

    def test_audit_uses_same_preprocessing_and_preserves_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode_000000"
            path.mkdir()
            rng = np.random.default_rng(12)
            arrays = {"rgb": rng.integers(0, 255, (4, 3, 16, 16, 3), dtype=np.uint8),
                      "states": np.zeros((4, 61), dtype=np.float32),
                      "actions": np.full((4, 23), .5, dtype=np.float32)}
            for key, value in arrays.items():
                np.save(path / f"{key}.npy", value)
            (path / "metadata.json").write_text(json.dumps({"episode_id": 0, "task_id": 0,
                "instruction": "turn on radio", "fps": 30, "cameras": list(CAMERAS)}))
            digest = file_sha256(path / "actions.npy")
            result = audit(make_policy(1), [path], 2)
            self.assertEqual(result["sample_count"], 2)
            self.assertEqual(result["train_vs_server_processed_tensor_max_abs_diff"], 0)
            self.assertEqual(max(result["roundtrip_max_abs_per_axis"]), 0)
            self.assertEqual(result["raw_controller_unit_l1"]["model_first"]["per_axis"], [0.]*23)
            self.assertEqual(digest, file_sha256(path / "actions.npy"))

    def test_declared_pose_hold_mapping_and_hashes(self):
        state = np.arange(61, dtype=np.float32)
        result = pose_hold_action(state, np.full(23, .4))
        np.testing.assert_array_equal(result[:3], [0, 0, 0])
        np.testing.assert_array_equal(result[3:7], state[53:57])
        np.testing.assert_array_equal(result[7:14], state[3:10])
        np.testing.assert_array_equal(result[15:22], state[28:35])
        self.assertAlmostEqual(float(result[14]), .4)
        self.assertEqual(object_sha256({"a": 1, "b": 2}), object_sha256({"b": 2, "a": 1}))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "file"
            path.write_bytes(b"abc")
            self.assertEqual(file_sha256(path), hashlib.sha256(b"abc").hexdigest())


if __name__ == "__main__":
    unittest.main()
