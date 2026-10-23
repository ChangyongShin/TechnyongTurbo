"""BEHAVIOR uses its dedicated LeRobot-v3 loader, not RoboTwin modality aliases.

Do not register an incompatible StarVLA DataConfig. See ../data.py and train.py.
This module is intentionally safe for StarVLA registry auto-discovery.
"""

ROBOT_TYPE_CONFIG_MAP = {}
DATASET_NAMED_MIXTURES = {}
