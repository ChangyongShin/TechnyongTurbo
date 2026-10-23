import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np

from turbovla.data.dataset import BehaviorDataset, load_episode, training_normalizer
from turbovla.evaluation.observations import RGB_KEYS, STATE_KEY, map_observation
from turbovla.data.preprocessing import CAMERAS, Normalizer
from turbovla.data.tasks import load_tasks


class ServerContractTests(unittest.TestCase):
    def observation(self):
        obs = {key: np.full((16, 16, 4), i, dtype=np.uint8) for i, key in enumerate(RGB_KEYS)}
        obs.update({STATE_KEY: np.arange(61, dtype=np.float32), "task_id": np.array([7]),
                    "robot_r1::cam_rel_poses": "must not enter policy"})
        return obs

    def test_default_r1pro_mapping_preserves_camera_and_state_order(self):
        images, state, instruction = map_observation(self.observation(), {7: "turn on radio"})
        self.assertEqual([int(image[0, 0, 0]) for image in images], [0, 1, 2])
        np.testing.assert_array_equal(state, np.arange(61))
        self.assertEqual(instruction, "turn on radio")

    def test_unknown_task_or_wrong_state_fails(self):
        obs = self.observation()
        with self.assertRaises(ValueError):
            map_observation(obs, {0: "other task"})
        obs[STATE_KEY] = np.zeros(60)
        with self.assertRaises(ValueError):
            map_observation(obs, {7: "turn on radio"})

    def test_memory_mapped_cache_and_streaming_statistics(self):
        rng = np.random.default_rng(123)
        states, actions, paths = [], [], []
        with tempfile.TemporaryDirectory() as root:
            for eid, count in enumerate((3, 5)):
                directory = Path(root) / str(eid)
                directory.mkdir()
                s = rng.normal(size=(count, 61)).astype(np.float32)
                a = rng.normal(size=(count, 23)).astype(np.float32)
                a[:, 14] = 0.4
                states.append(s)
                actions.append(a)
                for key, value in (("states", s), ("actions", a),
                                   ("rgb", np.zeros((count, 3, 16, 16, 3), dtype=np.uint8))):
                    np.save(directory / f"{key}.npy", value)
                (directory / "metadata.json").write_text(json.dumps({"instruction": "test", "episode_id": eid,
                                                                      "fps": 30, "cameras": CAMERAS}))
                paths.append(directory)
            self.assertIsInstance(load_episode(paths[0])["rgb"], np.memmap)
            normalizer = training_normalizer(paths)
            expected = Normalizer.fit(np.concatenate(states), np.concatenate(actions))
            for name in expected.__dataclass_fields__:
                np.testing.assert_allclose(getattr(normalizer, name), getattr(expected, name), atol=1e-6)
            dataset = BehaviorDataset(paths, normalizer, {}, horizon=12)
            self.assertEqual(len(dataset), 8)
            np.testing.assert_array_equal(dataset.ends, [3, 8])
            with self.assertRaises(IndexError):
                dataset[8]
            # Windows requires closing mappings before TemporaryDirectory cleanup.
            for episode in dataset.episodes:
                for key in ("rgb", "states", "actions"):
                    episode[key]._mmap.close()
            del dataset

    def test_task_metadata_jsonl_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            meta = Path(root) / "meta"
            meta.mkdir()
            (meta / "tasks.jsonl").write_text('{"task_index": 7, "task": "turn on radio"}\n')
            self.assertEqual(load_tasks(root), {7: "turn on radio"})

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow required")
    def test_task_parquet_pandas_index(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as root:
            meta = Path(root) / "meta"
            meta.mkdir()
            pq.write_table(pa.table({"task_index": [7], "__index_level_0__": ["turn on radio"]}), meta / "tasks.parquet")
            self.assertEqual(load_tasks(root), {7: "turn on radio"})

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("pyarrow", "av", "PIL")),
                         "pyarrow/PyAV/Pillow required")
    def test_lerobot_video_offsets_prepare_to_mmap(self):
        import av
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            meta = root / "meta"
            (meta / "episodes/chunk-000").mkdir(parents=True)
            (root / "data/chunk-000").mkdir(parents=True)
            template = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            (meta / "info.json").write_text(json.dumps({"fps": 30, "codebase_version": "v3.0",
                "features": {"action": {"shape": [23]}, "observation.state": {"shape": [61]}},
                "video_path": template}))
            pq.write_table(pa.table({"task_index": [7], "task": ["turn on radio"]}), meta / "tasks.parquet")
            metadata, rows = [], []
            for eid in (0, 1):
                episode = {"episode_index": eid, "length": 3}
                for camera in CAMERAS:
                    prefix = f"videos/observation.rgb.{camera}"
                    episode.update({f"{prefix}/chunk_index": 0, f"{prefix}/file_index": 0,
                                    f"{prefix}/from_timestamp": eid * 3 / 30})
                metadata.append(episode)
                for frame in range(3):
                    rows.append({"index": eid * 3 + frame, "frame_index": frame, "episode_index": eid,
                                 "timestamp": frame / 30, "task_index": 7,
                                 "observation.state": [float(eid)] * 61, "action": [float(frame)] * 23})
            pq.write_table(pa.Table.from_pylist(metadata), meta / "episodes/chunk-000/file-000.parquet")
            pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")
            for camera in CAMERAS:
                video = root / template.format(video_key=f"observation.rgb.{camera}", chunk_index=0, file_index=0)
                video.parent.mkdir(parents=True)
                with av.open(str(video), mode="w") as container:
                    stream = container.add_stream("libx264", rate=30)
                    stream.width = stream.height = 16
                    stream.pix_fmt = "yuv420p"
                    for frame_id in range(6):
                        frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), frame_id * 20, dtype=np.uint8), format="rgb24")
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
            cache = Path(temporary) / "cache"
            subprocess.run([sys.executable, "-m", "turbovla.data.prepare", "--root", str(root),
                            "--episodes", "0", "1", "--out", str(cache), "--image-size", "16"],
                           check=True, cwd=Path(__file__).parents[1], capture_output=True, text=True)
            first = load_episode(cache / "episode_000000")
            second = load_episode(cache / "episode_000001")
            self.assertGreater(second["rgb"].mean(), first["rgb"].mean())
            np.testing.assert_array_equal(second["actions"][:, 0], [0, 1, 2])
            for episode in (first, second):
                for key in ("rgb", "states", "actions"):
                    episode[key]._mmap.close()

    @unittest.skipUnless(importlib.util.find_spec("msgpack"), "msgpack required")
    def test_wire_reset_has_no_reply_and_health_check(self):
        from turbovla.evaluation.serve import PolicyServer, pack, unpack

        class FakePolicy:
            resets = 0
            def reset(self):
                self.resets += 1
            def infer(self, rgb, state, instruction):
                return {"action": np.arange(23, dtype=np.float32)}

        policy = FakePolicy()
        server = PolicyServer(policy, {7: "turn on radio"})
        self.assertIsNone(server.dispatch(unpack(pack({"reset": True}))))
        self.assertEqual(policy.resets, 1)
        response = unpack(pack(server.dispatch(unpack(pack(self.observation())))))
        np.testing.assert_array_equal(response["action"], np.arange(23))

    @unittest.skipUnless(importlib.util.find_spec("msgpack"), "msgpack required")
    def test_websocket_roundtrip_against_official_client_wire_format(self):
        from urllib.request import urlopen
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve
        from turbovla.evaluation.serve import PolicyServer, health_check, pack, unpack

        class FakePolicy:
            def reset(self):
                pass
            def infer(self, rgb, state, instruction):
                return {"action": np.ones(23, dtype=np.float32)}

        async def scenario():
            policy_server = PolicyServer(FakePolicy(), {7: "turn on radio"})
            async with serve(policy_server.handler, "127.0.0.1", 0, process_request=health_check) as listener:
                port = listener.sockets[0].getsockname()[1]
                def health():
                    with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as response:
                        return response.status
                self.assertEqual(await asyncio.to_thread(health), 200)
                async with connect(f"ws://127.0.0.1:{port}") as client:
                    metadata = unpack(await client.recv())
                    self.assertEqual(metadata["action_dim"], 23)
                    await client.send(pack({"reset": True}))
                    await client.send(pack(self.observation()))
                    result = unpack(await asyncio.wait_for(client.recv(), timeout=5))
                    np.testing.assert_array_equal(result["action"], np.ones(23))
        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
