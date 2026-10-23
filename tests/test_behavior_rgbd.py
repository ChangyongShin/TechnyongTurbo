import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from turbovla.data.dataset import BehaviorDataset, load_episode, training_normalizer
from turbovla.data.depth import decode_depth, frame_depth_metres, validate_depth_feature
from turbovla.evaluation.observations import DEPTH_KEYS, RGB_KEYS, STATE_KEY, map_depth_observation
from turbovla.data.preprocessing import CAMERAS, depth_metres, preprocess_views
from turbovla.evaluation.serve import PolicyServer, pack, unpack


def depth_frame(codes):
    height, width = codes.shape
    frame = av.VideoFrame(width=width, height=height, format="gray12le")
    padded = np.zeros((height, frame.planes[0].line_size // 2), dtype="<u2")
    padded[:, :width] = codes
    frame.planes[0].update(padded.tobytes())
    return frame


def close_episode(episode):
    for value in episode.values():
        if isinstance(value, np.memmap):
            value._mmap.close()


class RGBDContractTests(unittest.TestCase):
    def test_native_plane_preserves_codes_and_row_padding(self):
        codes = np.arange(16 * 18, dtype=np.uint16).reshape(16, 18)
        codes[0, 0], codes[-1, -1] = 0, 4095
        frame = depth_frame(codes)
        self.assertGreater(frame.planes[0].line_size, 18 * 2)
        np.testing.assert_allclose(frame_depth_metres(frame), depth_metres(codes, "log12"))
        with self.assertRaises(ValueError):
            frame_depth_metres(av.VideoFrame(width=18, height=16, format="rgb24"))

    def test_depth_metadata_cannot_be_misread_as_mm_pixels(self):
        info = {"video.pix_fmt": "gray12le", "video.depth_min": 0.01, "video.depth_max": 10.0,
                "video.shift": 3.5, "video.use_log": True, "is_depth_map": True,
                "depth_unit": "mm", "video.output_unit": "mm"}
        feature = {"dtype": "video", "shape": [32, 32, 1], "info": info}
        validate_depth_feature(feature)
        info["video.use_log"] = False
        with self.assertRaises(ValueError):
            validate_depth_feature(feature)

    def test_live_depth_order_units_and_missing_inputs(self):
        rgb = [np.zeros((16, 16, 4), dtype=np.uint8)] * 3
        obs = {key: np.full((16, 16), i + 1, dtype=np.float32) for i, key in enumerate(DEPTH_KEYS)}
        obs[DEPTH_KEYS[0]][0, 0] = np.inf
        depths = map_depth_observation(obs, rgb)
        self.assertEqual(depths[0][0, 0], 10)
        self.assertEqual([depth[1, 1] for depth in depths], [1, 2, 3])
        obs[DEPTH_KEYS[0]][0, 0] = np.nan
        with self.assertRaises(ValueError):
            map_depth_observation(obs, rgb)
        del obs[DEPTH_KEYS[0]]
        with self.assertRaises(KeyError):
            map_depth_observation(obs, rgb)

    def test_rgbd_server_uses_separate_metric_depth(self):
        class FakePolicy:
            input_channels = 4
            def reset(self):
                pass
            def infer(self, rgb, state, instruction, *, depth, depth_encoding):
                self.depth, self.encoding = depth, depth_encoding
                return {"action": np.ones(23, dtype=np.float32)}
        policy = FakePolicy()
        server = PolicyServer(policy, {7: "turn on radio"})
        obs = {key: np.zeros((16, 16, 4), dtype=np.uint8) for key in RGB_KEYS}
        obs.update({key: np.full((16, 16), 2.0, dtype=np.float32) for key in DEPTH_KEYS})
        obs.update({STATE_KEY: np.zeros(61), "task_id": np.array([7])})
        result = server.dispatch(unpack(pack(obs)))
        self.assertEqual(result["action"].shape, (23,))
        self.assertEqual(policy.encoding, "m")
        np.testing.assert_array_equal(policy.depth, np.full((3, 16, 16), 2))

    def test_rgbd_video_to_cache_to_training_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            meta = root / "meta"
            (meta / "episodes/chunk-000").mkdir(parents=True)
            (root / "data/chunk-000").mkdir(parents=True)
            template = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            features = {"action": {"shape": [23]}, "observation.state": {"shape": [61]}}
            metadata, rows = [], []
            for eid in (0, 1):
                episode = {"episode_index": eid, "length": 2}
                for camera in CAMERAS:
                    for modality in ("rgb", "depth_linear"):
                        key = f"observation.{modality}.{camera}"
                        prefix = f"videos/{key}"
                        episode.update({f"{prefix}/chunk_index": 0, f"{prefix}/file_index": 0,
                                        f"{prefix}/from_timestamp": eid * 2 / 30})
                metadata.append(episode)
                for frame_id in range(2):
                    rows.append({"index": eid * 2 + frame_id, "frame_index": frame_id,
                                 "episode_index": eid, "timestamp": frame_id / 30, "task_index": 7,
                                 "observation.state": [float(eid)] * 61, "action": [float(frame_id)] * 23})
            for camera in CAMERAS:
                for modality in ("rgb", "depth_linear"):
                    key = f"observation.{modality}.{camera}"
                    features[key] = {"dtype": "video", "shape": [32, 32, 3 if modality == "rgb" else 1]}
                    if modality == "depth_linear":
                        features[key]["info"] = {"video.pix_fmt": "gray12le", "video.depth_min": 0.01,
                            "video.depth_max": 10.0, "video.shift": 3.5, "video.use_log": True,
                            "is_depth_map": True, "depth_unit": "mm", "video.output_unit": "mm"}
                    path = root / template.format(video_key=key, chunk_index=0, file_index=0)
                    path.parent.mkdir(parents=True)
                    with av.open(str(path), mode="w") as container:
                        stream = container.add_stream("libx264" if modality == "rgb" else "libx265", rate=30)
                        stream.width = stream.height = 32
                        stream.pix_fmt = "yuv420p" if modality == "rgb" else "gray12le"
                        if modality != "rgb":
                            stream.options = {"x265-params": "lossless=1:log-level=error:bframes=0:keyint=1"}
                        for index, code in enumerate((0, 1024, 2048, 4095)):
                            frame = (av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 20, dtype=np.uint8), format="rgb24")
                                     if modality == "rgb" else depth_frame(np.full((32, 32), code, dtype=np.uint16)))
                            for packet in stream.encode(frame):
                                container.mux(packet)
                        for packet in stream.encode():
                            container.mux(packet)
            (meta / "info.json").write_text(json.dumps({"fps": 30, "codebase_version": "v3.0",
                                                         "features": features, "video_path": template}))
            pq.write_table(pa.table({"task_index": [7], "task": ["turn on radio"]}), meta / "tasks.parquet")
            pq.write_table(pa.Table.from_pylist(metadata), meta / "episodes/chunk-000/file-000.parquet")
            pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")
            video = root / template.format(video_key=f"observation.depth_linear.{CAMERAS[0]}", chunk_index=0, file_index=0)
            decoded = decode_depth(video, [2 / 30, 3 / 30], 30)
            expected = depth_metres(np.array([[2048, 4095]], dtype=np.uint16), "log12")[0]
            np.testing.assert_allclose([frame[0, 0] for frame in decoded], expected, atol=1e-5)
            with self.assertRaises(ValueError):
                decode_depth(video, [4 / 30], 30)
            for output_format in ("mmap", "npz"):
                cache = Path(temporary) / output_format
                command = [sys.executable, "-m", "turbovla.data.prepare", "--root", str(root),
                           "--episodes", "0", "1", "--out", str(cache), "--image-size", "16", "--rgbd",
                           "--format", output_format]
                result = subprocess.run(command, cwd=Path(__file__).parents[1], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                paths = [cache / (f"episode_{eid:06d}" + (".npz" if output_format == "npz" else "")) for eid in (0, 1)]
                episode = load_episode(paths[1])
                self.assertEqual(episode["modality"], "rgbd")
                np.testing.assert_allclose(episode["depth"][:, 0, 0, 0], expected, atol=1e-5)
                close_episode(episode)
                normalizer = training_normalizer(paths[:1])
                preprocessing = dict(size=16, mean=[0.5] * 3, std=[0.5] * 3)
                dataset = BehaviorDataset(paths[1:], normalizer, preprocessing, horizon=12, modality="rgbd")
                batch = dataset[0]
                self.assertEqual(batch["images"].shape, (3, 4, 16, 16))
                np.testing.assert_allclose(batch["images"][:, 3, 0, 0], expected[0] / 10, atol=1e-6)
                live = preprocess_views(dataset.episodes[0]["rgb"][0], **preprocessing,
                                        depth=dataset.episodes[0]["depth"][0], depth_encoding="m")
                np.testing.assert_array_equal(batch["images"], live)
                for cached in dataset.episodes:
                    close_episode(cached)
                with self.assertRaises(ValueError):
                    BehaviorDataset(paths, normalizer, preprocessing, modality="rgb")


if __name__ == "__main__":
    unittest.main()
