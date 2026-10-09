# Third-party notices

FastWAM's own code is released under the MIT License ([LICENSE](./LICENSE)). The files below
were copied or adapted from other open-source projects and remain under their original
licenses. Full license texts are in [LICENSES/](./LICENSES/).

## Code in this repository

| Project | License | Copyright | Files here |
|---|---|---|---|
| [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | Apache-2.0 ([text](./LICENSES/Apache-2.0.txt)) | DiffSynth-Studio authors (ModelScope) | `src/fastwam/models/wan22/wan_video_vae.py`, `wan_video_text_encoder.py` (adapted), `wan_video_dit.py` (adapted and extended); the model loading and download helpers in `src/fastwam/models/wan22/helpers/` follow DiffSynth's `ModelConfig` and hash-based loading |
| [LeRobot](https://github.com/huggingface/lerobot) | Apache-2.0 ([text](./LICENSES/Apache-2.0.txt)) | Copyright 2024 The HuggingFace Inc. team | `src/fastwam/datasets/lerobot/constants.py`, `src/fastwam/datasets/lerobot/lerobot/` (`lerobot_dataset.py`, `datasets/compute_stats.py`, `datasets/utils.py`, `datasets/video_utils.py`) |
| [openpi](https://github.com/Physical-Intelligence/openpi) | Apache-2.0 ([text](./LICENSES/Apache-2.0.txt)) | Physical Intelligence | `experiments/teleavatar_v2_deploy/server/openpi_msgpack_numpy.py` — verbatim copy of `packages/openpi-client/src/openpi_client/msgpack_numpy.py` @ 231a1cf |
| [msgpack-numpy](https://github.com/lebedov/msgpack-numpy) | BSD-3-Clause ([text](./LICENSES/msgpack-numpy.txt)) | Copyright (c) 2013-2022, Lev E. Givon | `experiments/teleavatar_v2_deploy/server/openpi_msgpack_numpy.py` — openpi adapted its encoding from this project |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | BSD-3-Clause ([text](./LICENSES/pytorch3d.txt)) | Copyright (c) Meta Platforms, Inc. and affiliates | `src/fastwam/datasets/lerobot/utils/rotation.py` (adapted from PyTorch3D); `src/fastwam/datasets/lerobot/transforms/rotation.py` carries Meta's BSD-style header |
| NVIDIA (source project not named in the file) | Apache-2.0 ([text](./LICENSES/Apache-2.0.txt)) | Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES | `src/fastwam/utils/misc.py` |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) | MIT ([text](./LICENSES/robosuite.txt)) | Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab | `quat2axisangle` in `experiments/libero/libero_utils.py` |
| [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | MIT ([text](./LICENSES/RoboTwin.txt)) | Copyright (c) 2025 Tianxing Chen | the RoboTwin evaluation integration in `experiments/robotwin/` (adapted) |

## Model weights (not included)

The base weights are downloaded at first use (see the README's
[Base weights](./README.md#base-weights)) and are governed by their own licenses:

| Weights | Source | License |
|---|---|---|
| Wan2.2-TI2V-5B DiT, VAE and umT5 text encoder | `Wan-AI/Wan2.2-TI2V-5B` | Apache-2.0 |
| umT5 tokenizer | `Wan-AI/Wan2.1-T2V-1.3B` | Apache-2.0 |
| Converted VAE / T5 safetensors (default download) | `DiffSynth-Studio/Wan-Series-Converted-Safetensors` (ModelScope) | Apache-2.0 |

Python packages installed with pip or conda are not redistributed here and carry their own
licenses.
