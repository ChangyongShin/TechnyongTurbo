"""Strict default R1Pro evaluation mapping, independent of OmniGibson imports."""
import numpy as np

from turbovla.data.preprocessing import finite_array, depth_metres

RGB_KEYS = (
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
)
STATE_KEY = "robot_r1::proprio"
DEPTH_KEYS = tuple(key.removesuffix("::rgb") + "::depth_linear" for key in RGB_KEYS)
ACTION_ORDER = {
    "base": [0, 3], "trunk": [3, 7], "left_arm": [7, 14],
    "left_gripper": [14, 15], "right_arm": [15, 22], "right_gripper": [22, 23],
}


def map_observation(observation, instructions, rgb_keys=RGB_KEYS, state_key=STATE_KEY):
    if len(rgb_keys) != 3:
        raise ValueError("Exactly three camera keys in head/left/right order are required")
    task_id = np.asarray(observation["task_id"])
    if task_id.size != 1 or task_id.dtype.kind not in "iu":
        raise ValueError("Expected one integer task_id")
    task_id = int(task_id.item())
    if task_id not in instructions:
        raise ValueError(f"No dataset instruction for evaluation task_id {task_id}")
    # Explicit whitelist: camera poses or other simulator fields never enter the model.
    rgb = [np.asarray(observation[key]) for key in rgb_keys]
    for image in rgb:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError("Expected HWC uint8 RGB/RGBA evaluation images")
    state = finite_array(observation[state_key], (61,), "R1Pro proprioception")
    return rgb, state, instructions[task_id]


def map_depth_observation(observation, rgb, depth_keys=DEPTH_KEYS):
    if len(depth_keys) != 3:
        raise ValueError("Exactly three depth keys in head/left/right order are required")
    depths = []
    for image, key in zip(rgb, depth_keys):
        # OmniGibson sensor output is metric distance_to_image_plane, not raw
        # log12 video codes or the official LeRobot loader's millimetre output.
        raw = np.asarray(observation[key], dtype=np.float32)
        # Isaac Sim uses +inf for pixels beyond a measured surface. Official
        # quantization saturates those at depth_max; match that same convention.
        depth = depth_metres(np.where(np.isposinf(raw), 10.0, raw), "m")
        if depth.shape != image.shape[:2]:
            raise ValueError(f"RGB/depth alignment mismatch for {key}")
        depths.append(depth)
    return depths
