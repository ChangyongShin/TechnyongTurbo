"""Decode released 12-bit depth planes without RGB conversion or unit guessing."""
import numpy as np

from turbovla.data.preprocessing import depth_metres


def validate_depth_feature(feature):
    info = feature.get("info", {})
    expected = {"video.pix_fmt": "gray12le", "video.depth_min": 0.01, "video.depth_max": 10.0,
                "video.shift": 3.5, "video.use_log": True, "is_depth_map": True}
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f"Unsupported released depth encoding: {key}={info.get(key)!r}")
    # output_unit describes the official loader's output, not the raw video plane.
    # This decoder dequantizes raw log12 codes directly to metres.
    if feature.get("dtype") != "video" or feature["shape"][-1] != 1:
        raise ValueError("Expected a single-channel depth video feature")


def frame_depth_metres(frame):
    if frame.format.name not in {"gray12le", "yuv420p12le", "yuv422p12le", "yuv444p12le"}:
        raise ValueError(f"Expected native 12-bit little-endian depth, got {frame.format.name}")
    # Extract the native Y/gray plane and honor FFmpeg row padding. Reformatting
    # to RGB/gray16 can rescale codes and destroy the released depth encoding.
    plane = frame.planes[0]
    codes = np.frombuffer(plane, dtype="<u2").reshape(frame.height, plane.line_size // 2)[:, :frame.width]
    return depth_metres(codes, "log12")


def decode_depth(path, timestamps, fps, size=None, destination=None, expected_shape=None):
    import av
    from PIL import Image

    timestamps = np.asarray(timestamps, dtype=np.float64)
    if not len(timestamps) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Expected finite, strictly increasing depth timestamps")
    images, index = [], 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(0, float(timestamps[0]) - 1) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            time = float(frame.pts * stream.time_base)
            if time < timestamps[index] - 0.51 / fps:
                continue
            if abs(time - timestamps[index]) > 0.51 / fps:
                raise ValueError(f"Missing depth frame in {path}: requested {timestamps[index]}, got {time}")
            image = frame_depth_metres(frame)
            if expected_shape is not None and image.shape != tuple(expected_shape):
                raise ValueError(f"Unexpected depth resolution: {image.shape} != {expected_shape}")
            if size is not None:
                image = np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.NEAREST))
            if destination is None:
                images.append(image)
            else:
                destination[index] = image
            index += 1
            if index == len(timestamps):
                break
    if index != len(timestamps):
        raise ValueError(f"Depth video ended before all timestamps were decoded: {path}")
    return images
