# BEHAVIOR 2026 RGB baseline

For the A100 training / separate RTX evaluation setup, see
[A100_SETUP_KO.md](A100_SETUP_KO.md). This includes pinned setup and launch scripts,
revision-recorded selective downloads, memory-mapped episode caches and a
simulator-independent websocket policy server.

This adapter preserves all 23 demonstration actions (base, trunk, arms and
grippers) and all 61 proprioceptive fields. Camera order is head, left wrist,
right wrist. It does not reuse RoboTwin's 14-dimensional/binary-gripper recipe.
LIBERO and RoboTwin keep their original RGB defaults.

## Install and prepare a small subset

Install a compatible CUDA PyTorch build first, then `pip install -e '.[behavior]'`.
Place DINOv3 and BERT weights in local directories. This baseline initializes
the fusion/action modules from scratch unless `--initial-weights` is supplied.
That option transfers only matching core TurboVLA keys and shapes, reports
missing keys, and deliberately does not pretend that a LIBERO action head is
a pretrained BEHAVIOR action head. GroundingDINO-format key conversion is not
implemented by this entry point.

Download complete selected episodes from `behavior-1k/2026-challenge-demos`,
including `meta/info.json`, `meta/tasks.parquet` (legacy JSONL also supported), the episode metadata parquet,
data parquet shards and the three RGB video shards. A repository/submodule is
not the demonstration dataset. Record the Hugging Face revision used.

```bash
python -m turbovla.data.prepare \
  --root /data/2026-challenge-demos --episodes 0 1 2 \
  --out /data/behavior-small --image-size 224 --format npz
```

The preparer reads LeRobot v3 episode offsets and verifies complete 30 Hz
episodes. It preserves all action columns without unit conversion or frame
dropping. No simulator is required. IDs above are examples: select episodes
of the intended task before preparation. Prepared NPZ episodes and this
in-memory loader are for a small-subset smoke test, NOT the full 20k-episode
training workload. A streaming/sharded cache is required for full-scale runs.
The default `--format mmap` streams decoded RGB into disk-backed NPY arrays;
the trainer reads these without loading all RGB into RAM and computes normalization
one episode at a time. Pass episode directories instead of NPZ paths for this format.
Full-dataset throughput still requires benchmarking. The raw LeRobot adapter requires validation against an actual downloaded
episode; the development environment did not contain the demo/video files.

## Train and validate

```bash
python -m turbovla.training.trainer \
  --train /data/behavior-small/episode_000000.npz /data/behavior-small/episode_000001.npz \
  --validation /data/behavior-small/episode_000002.npz \
  --dinov3 /models/dinov3-vitb --bert /models/bert-base-uncased \
  --out outputs/behavior-smoke --steps 1 --batch-size 1
```

Use a new output directory for each run. After the single-step check, train a
small overfit experiment and then a held-out episode experiment. Statistics
are fitted ONLY to the listed training episodes; duplicate/overlapping episode
IDs are rejected. Chunk padding is excluded from loss. RGB DINOv3 and BERT
are frozen for the initial baseline; fusion and action modules train. Validation
reports normalized L1 for each action column so base failures are not hidden
by arm dimensions. Saved checkpoints include model configuration, preprocessing,
normalization and train/validation IDs. GPU memory/throughput are not benchmarked.

## Policy integration

`BehaviorPolicy.from_checkpoint(path, device='cuda', execute_steps=1)` loads
the same preprocessing and normalization used in training. Call
`infer(rgb_views, state61, task_instruction)` to receive `{'action': array23}`.
The caller must map official flattened observation keys to the camera order
above and call `reset()` on EVERY episode reset (also when switching tasks).
The policy reuses a predicted chunk for at most `execute_steps` observations.
Start at one; increase only after closed-loop validation.

The action is in ORIGINAL DEMONSTRATION ORDER and controller-input units. It
does not multiply by wheel radius or convert all components to joint angles.
Use the matching official R1Pro controller configuration; custom controllers
require an explicit, tested remapping. In particular, avoid double scaling the
local-frame base velocity commands. Do not use an all-zero 23-vector as an
emergency hold for absolute-position-controlled arms/trunk.

`python -m turbovla.evaluation.serve --help` provides a simulator-independent
websocket server using the official R1Pro observation keys, msgpack array format,
health check and reset-without-ACK protocol. Camera poses and other extra fields
are excluded. Checkpoint loading accepts local DINOv3/BERT path overrides.
The server observation mapping, episode-reset handshake and controller order must
still be verified in the actual simulator before reporting challenge results.
No privileged scene/object/world-pose input is required by this adapter.

## RGB-D training and serving

`preprocess_views` supports separate aligned depth with an explicit `log12`,
`mm` or `m` encoding. It never treats RGBA alpha as depth. RGB keeps the DINOv3
mean/std; depth is converted once to metres, clipped to [0.01,10], nearest-resized
and divided by 10. NaN/negative values fail; live sensor +inf is explicitly
saturated at 10 m to match the official demonstration quantizer.
The model has opt-in `vision.input_channels=4`; the added patch weights start
at zero so initial RGB responses are preserved. The frozen-backbone recipe
trains only the new depth kernel inside the backbone; pretrained RGB weights
remain frozen. Standard HF DINOv3 Conv2d,
wrapped projection and timm projection paths are supported.

Pass `--rgbd` to `download`, `preflight`, `prepare` and `train`, or set `RGBD=1`
for `scripts/behavior_2026/train_a100.sh`. Use a separate RGB-D cache/output
directory. The depth decoder reads native gray12/Y 12-bit planes including row
padding, verifies the released logarithmic metadata and writes aligned float32
metres. It does not run depth through an RGB decoder and does not interpret
the official loader's `video.output_unit=mm` as the raw video's encoding.

The server detects the checkpoint's channel count and requires three aligned
`::depth_linear` observations for RGB-D. Live OmniGibson depth is in metres.
Use `RGBDFullResWrapper` for evaluation. RGB-D model inputs are `[B,3,4,H,W]`;
state/action contracts remain 61/23. RGB and RGB-D caches cannot be mixed.

CPU tests exercise actual synthetic lossless 12-bit HEVC decoding, RGB-D cache
preparation, identical train/inference preprocessing, depth gradients with RGB
weights frozen and checkpoint policy reload. The training CLI also runs one
synthetic step with tiny local HF backbones. Actual released demo episodes,
A100 BF16 execution and closed-loop OmniGibson evaluation remain unverified.

## Checks

```bash
python -m unittest discover -s tests -p 'test_behavior*.py' -v
```

These CPU tests cover alpha handling, depth units, shape/alignment failures,
all-23-dimension normalization round trips, constant action dimensions, chunk
masks and configuration isolation. They do not replace GPU forward/backward,
official dataset/video decoding tests or closed-loop task evaluation.
