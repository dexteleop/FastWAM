# FastWAM — LeRobot Training & Teleavatar 2.0 Deployment

A fork of [FastWAM](https://github.com/zhou-yh19/FastWAM) focused on one path end to end:
**train a world-action model on LeRobot-format robot data, then serve it on a Teleavatar 2.0
(TA2) dual-arm robot over ROS 2.**

Nothing here is specific to one task. You supply a LeRobot dataset; the configs, scripts and
deploy chain are parameterised by task name throughout.

> Looking for the original paper code (LIBERO / RoboTwin benchmarks)? Its configs still ship in
> `configs/{data,task}/`, but the benchmark documentation lives upstream at
> [zhou-yh19/FastWAM](https://github.com/zhou-yh19/FastWAM). This document covers the
> LeRobot + TA2 path instead.

| | |
|---|---|
| **Input** | LeRobot 2.x dataset (`meta/` + `data/` + `videos/`), 3 cameras, 72-d action/state |
| **Model** | Wan2.2-TI2V-5B video expert + 1.0B action expert (MoT), flow matching |
| **Output** | 16-d action chunks at 20 Hz → ROS 2 joint/gripper commands at 200 Hz |
| **Hardware** | Training: 8×A100/A800 80 GB, DeepSpeed ZeRO-1/2. Serving: 1 GPU ≥24 GB |

This README is the happy path: one command per step, clone → data → train → deploy, with the
caveats you will hit on a first run. The *why*, the variants, the failure modes and the full
variable tables are in [docs/GUIDE.md](./docs/GUIDE.md).

---

## Contents

- [How the pieces fit](#how-the-pieces-fit)
- [Requirements](#requirements)
- [Install](#install)
  - [Training / serving host](#training--serving-host)
  - [Base weights](#base-weights)
  - [Robot host](#robot-host)
- [Data](#data)
  - [Step 1. Transcode the video](#step-1-transcode-the-video)
  - [Step 2. Write two configs](#step-2-write-two-configs)
  - [Step 3. Precompute text embeddings](#step-3-precompute-text-embeddings)
- [Train](#train)
  - [Step 4. Check that the data loads](#step-4-check-that-the-data-loads)
  - [Step 5. Launch training](#step-5-launch-training)
  - [Resume and LR anneal](#resume-and-lr-anneal)
- [Deploy on TA2](#deploy-on-ta2)
  - [Step 6. Move the checkpoint to the deploy host](#step-6-move-the-checkpoint-to-the-deploy-host)
  - [Step 7. Start the server (GPU host)](#step-7-start-the-server-gpu-host)
  - [Step 8. Start the client (robot host)](#step-8-start-the-client-robot-host)
  - [Optional: a task library for switching instructions](#optional-a-task-library-for-switching-instructions)
- [Analysis tools](#analysis-tools)
- [Data conventions](#data-conventions)
- [Further reading](#further-reading)
- [Citation](#citation)
- [License](#license)

---

## How the pieces fit

```
LeRobot dataset ──transcode──> *_lowres ──transcode──> *_mono
  meta/tasks.jsonl                                       │
        │                                                │
        └──precompute_text_embeds──> text_embeds_cache/   │
                                            │            │
                              configs/data/<task>.yaml ───┤
                              configs/task/<task>.yaml    │
                                            │            │
                                    launch_train.sh ──────┘
                                            │
                              runs/<task>/<RUN_ID>/
                                ├── checkpoints/weights/step_*.pt
                                ├── dataset_stats.json      ← must travel with the weights
                                └── config.yaml
                                            │
                        ┌───────────────────┴───────────────────┐
                        │                                       │
            start_local_serve_ws.sh                    make_task_map.py
            (GPU host, WebSocket)                      (taskmap.json)
                        │
            run_task_ws.sh  (robot host, ROS 2 + RTP video)
```

Two hosts are normal: a GPU box runs inference, the robot-side machine runs ROS 2 and
receives camera frames over RTP. They talk over WebSocket + msgpack.

---

## Requirements

| Role | Needs |
|---|---|
| Training host | 8×A100/A800 80 GB, DeepSpeed ZeRO-1/2. Python 3.10, `torch==2.7.1+cu128`, `nvcc` (DeepSpeed checks the CUDA toolkit on import), `ffmpeg`/`ffprobe` with libx264, `tmux`. Disk: base weights ≈32 GB, each ZeRO state snapshot ~80 GB, each weights file ~12 GB |
| Serving (GPU) host | 1 GPU ≥24 GB; each server holds ~14 GB of GPU memory, so one server per GPU. Same conda env as training, but only the VAE (~1.4 GB) from the base weights |
| Robot host | ROS 2 Humble on the **system `/usr/bin/python3`** (not conda), GStreamer with the H.265 decoder plugin (`gst-inspect-1.0 nvh265dec`), RTP video arriving on port 8890 |
| Host RAM | The ActionDiT preprocess and the text-embedding precompute both fit in 15 GB, so a workstation can run them (see [Base weights](#base-weights)) |

The GPU host and the robot host may be the same machine (the client's `SERVER_HOST` defaults
to `127.0.0.1`); the two interpreters may not.

---

## Install

### Training / serving host

Use [Miniforge](https://github.com/conda-forge/miniforge) for conda. It ships with
conda-forge as its only channel, so Anaconda's Terms of Service, which cover the
default Anaconda channels, never come into play. If you have no conda yet:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh        # installs to ~/miniforge3 by default
```

Then, in a new shell:

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install -e ".[analysis]"   # optional: the rosbag / plotting tools under Analysis tools
```

Already on Miniconda or Anaconda? Recent versions refuse `conda create` on the default
Anaconda channels until their Terms of Service are accepted (`CondaToSNonInteractiveError`).
Create the env from conda-forge instead of accepting them:
`conda create -n fastwam -c conda-forge --override-channels python=3.10 -y`. If PyPI is slow
from your network, a PyPI mirror (`pip install -i <mirror> ...`) works for everything except
the torch line, which needs the PyTorch index.

System tools: `ffmpeg`/`ffprobe` with libx264 (transcoding) and `tmux` (`launch_train.sh`).
Training also needs `nvcc`: DeepSpeed checks the CUDA toolkit version on import and
`accelerate launch` fails with `MissingCUDAException: CUDA_HOME does not exist` without it.
If the host has no CUDA toolkit, `conda install -y -c conda-forge --override-channels cuda-nvcc=12.8` into the env is
enough; `conda activate` puts it on `PATH`.

### Base weights

Training needs the Wan2.2 base weights under `checkpoints/` (≈32 GB); a deploy-only machine
needs just the VAE (see [Step 6](#step-6-move-the-checkpoint-to-the-deploy-host)).
Missing files are downloaded automatically from ModelScope on first use: the preprocess step
below pulls the DiT and VAE, the text-embedding precompute pulls T5 and the tokenizer. Budget
up to an hour for the first ≈32 GB (single streams ran at 4–25 MB/s in our runs). Partial
files sit in `._____temp/` until complete; that directory and `.msc`/`.mv` are ModelScope
bookkeeping. Once the files are present nothing touches the network.
`DIFFSYNTH_SKIP_DOWNLOAD=true` forbids downloading; a missing file then surfaces later as
`Cannot detect model type ... File: []`. Downloading from Hugging Face instead needs an env
var and a model-config change — see
[Base weights and download sources](./docs/GUIDE.md#base-weights-and-download-sources).

If you already have the files, place them like this — the directory names are load-bearing:

```
checkpoints/
├── Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors   # 19 GB
├── Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/                     # tokenizer files
└── DiffSynth-Studio/Wan-Series-Converted-Safetensors/
    ├── Wan2.2_VAE.safetensors                                  # 1.4 GB
    └── models_t5_umt5-xxl-enc-bf16.safetensors                 # 11 GB, precompute only
```

Then pre-generate the ActionDiT backbone once (it is interpolated from the Wan2.2 DiT, not
downloaded; a deploy-only machine can skip it):

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

The Wan component loader resolves all model paths relative to `DIFFSYNTH_MODEL_BASE_PATH`
(default `./checkpoints/`, i.e. relative to the working directory). Set it whenever the
weights live elsewhere; note that `launch_train.sh` starts training in tmux, which inherits
the tmux server's environment rather than your shell's if a server is already running.

This step and the text-embedding precompute below both fit in 15 GB of host RAM (measured
under a cgroup cap on a 32 GB / RTX 4090 box), so a workstation can run them.

### Robot host

The robot-side client has a different, much lighter dependency set and **must** run on the
system interpreter, because ROS 2 Humble's C extensions are built for it. Install its
dependencies into that interpreter's user site:

```bash
/usr/bin/python3 -m pip install --user -r experiments/teleavatar_v2_deploy/client/requirements.txt
```

Pre-flight check: `bash experiments/teleavatar_v2_deploy/check_deployment.sh`. Why conda
cannot be used, and what the check does and does not prove:
[Two machines, two interpreters](./docs/GUIDE.md#two-machines-two-interpreters).

---

## Data

A LeRobot 2.x directory tree:

```
data/<your_dataset>/
├── meta/
│   ├── info.json          # fps, total_episodes, total_frames, feature shapes
│   ├── tasks.jsonl        # {"task_index": 0, "task": "<natural-language instruction>"}
│   ├── episodes.jsonl
│   └── episodes_stats.jsonl
├── data/chunk-000/episode_NNNNNN.parquet     # action + observation.state
└── videos/chunk-000/<camera>/episode_NNNNNN.mp4
```

Three cameras are expected by the TA2 configs: `head_camera`, `left_color`, `right_color`.
Each records a side-by-side stereo pair; only the **left eye** is used for training. The
72-d action/state layout is described under [Data conventions](#data-conventions).

### Step 1. Transcode the video

Source video is typically 3840×1920 HEVC. Decoding one 33-frame window costs ~3.2 CPU-seconds,
which starves the GPU (~94 % of wall time spent waiting on data). Two stages fix that:

```bash
# stage 1: original SBS -> downscaled SBS
DATASETS="<your_dataset>" bash scripts/transcode.sh

# stage 2: SBS -> left eye only (what training consumes)
DATASETS="<your_dataset>" MODE=mono bash scripts/transcode.sh

# required: rewrite meta/info.json, or LeRobot still reports the source resolution
python scripts/patch_transcoded_meta.py --mode lowres data/<your_dataset>_lowres
python scripts/patch_transcoded_meta.py --mode mono   data/<your_dataset>_mono
```

`DATASETS` takes dataset **base names** (no `_lowres` / `_mono` suffix), relative to
`SRC_ROOT` (default `./data`). Omit it and the script discovers every dataset with the right
source suffix. Output keeps the LeRobot layout: `meta/` and `data/` are symlinked back to the
source, only `videos/` is rewritten. Re-running skips finished files. Keep `data/*_lowres`
after producing `_mono` — the mono directory links back into it. Resolutions per stage and
the symlink details: [Onboarding a new dataset](./docs/GUIDE.md#onboarding-a-new-dataset).

### Step 2. Write two configs

```bash
cp configs/data/ta2_mono_template.yaml configs/data/<your_task>.yaml
cp configs/task/ta2_mono_template.yaml configs/task/<your_task>.yaml
```

In the **data** config set `dataset_dirs` (one or more `*_mono` dirs) and
`text_embedding_cache_dir` (convention: `./data/text_embeds_cache/<your_task>`; the commands
below assume it). In the **task** config point `override /data:` at your data config name;
wandb is off by default, enable it there if you want it. Both templates document every field
inline.

Camera resolutions in `shape_meta.images` must match the transcoded video exactly — a
mismatch produces silently wrong results, not an error.

Two things the templates already get right, so do not edit them away: the first line of the
task config must stay `# @package _global_` (`launch_train.sh` checks it; see
[The config chain](./docs/GUIDE.md#the-config-chain)), and its `defaults` pick the
lifting-column mode `/lift: "off"` — turn it on only if the dataset moves the column (see
[The lifting column](./docs/GUIDE.md#the-lifting-column-liftoffon)).

### Step 3. Precompute text embeddings

The umT5-XXL text encoder is ~11 GB. Training and serving both read cached embeddings instead
of loading it:

```bash
python scripts/precompute_text_embeds.py task=<your_task>
```

This reads instructions straight from each dataset's `meta/tasks.jsonl` and writes
`<sha256 of the formatted prompt>.t5_len128.wan22ti2v5b.pt` into `text_embedding_cache_dir`.
Because the filename is a hash of the exact prompt text, **any** wording difference — including
whitespace — produces a file the trainer will not find.

---

## Train

### Step 4. Check that the data loads

Before committing GPUs, check that the data loads (under a minute, ~2 GB RSS; it needs the
text embeddings from step 3; the normalization stats it computes go to a temporary directory):

```bash
python - <<'EOF'
import os, tempfile
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
register_default_resolvers()
misc.register_work_dir(tempfile.mkdtemp(prefix="fastwam_datacheck_"))
with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base="1.3"):
    cfg = compose(config_name="train", overrides=["task=<your_task>"])
ds = instantiate(cfg.data.train)
print(len(ds), {k: tuple(v.shape) for k, v in ds[0].items() if hasattr(v, "shape")})
EOF
```

Expect `video (3, 9, 352, 256)`, `action (32, 16)`, `proprio (32, 14)` for the TA2 mono
template with `lift=off`.

### Step 5. Launch training

```bash
bash scripts/launch_train.sh <your_task>
```

That is the whole command. It starts a tmux session with two windows: `train`, and `prune` —
the latter is **not optional**, since each ZeRO state snapshot is ~80 GB and will fill the disk
without it. Everything is controlled by environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `ZERO` | `1` | DeepSpeed stage, 1 or 2 |
| `NPROC` | `8` | GPUs per node |
| `KEEP` | `2` | ZeRO state snapshots to retain (with `1`, the previous one is deleted while the next is still being written) |
| `SESSION` | `fastwam_<task>` | tmux session name |
| `CONDA_SH` | the active conda's (`$CONDA_EXE`), else the first of `~/miniforge3`, `~/miniconda3`, `~/anaconda3` | `<conda base>/etc/profile.d/conda.sh`; the launcher stops before starting tmux if it is not found |
| `CONDA_ENV` | `fastwam` | conda env to activate (name or path) |
| `DRY_RUN` | — | print the command, start nothing |

```bash
# ZeRO-2 on 4 GPUs
ZERO=2 NPROC=4 bash scripts/launch_train.sh <your_task>
```

Outputs land in `runs/<your_task>/<RUN_ID>/`: `checkpoints/weights/step_*.pt`,
`dataset_stats.json` and the resolved `config.yaml`. Checkpoint directories are zero-padded:
`state/step_000600`, `weights/step_000600.pt`. What a healthy training log looks like, and
what to do when it is not: [Monitoring](./docs/GUIDE.md#monitoring).

### Resume and LR anneal

Any extra argument is passed through to hydra, so resume and LR-anneal need no new config file:

```bash
# resume full state (weights + Adam momentum + LR schedule + dataloader cursor + RNG);
# repeat the original run's overrides, and see below for additional_steps
bash scripts/launch_train.sh <your_task> \
  resume=./runs/<your_task>/<RUN_ID>/checkpoints/state/step_<N> \
  additional_steps=<M>

# LR anneal: rebuild the schedule over the remaining steps
bash scripts/launch_train.sh <your_task> \
  resume=./runs/<your_task>/<RUN_ID>/checkpoints/state/step_<N> \
  resume_reinit_lr=true additional_steps=<M> \
  resume_warmup_frac=0.0 learning_rate=<lr at the resume point>
```

Three rules, each of which fails quietly when broken:

- **A resume is the original command plus `resume=`.** Pass the same overrides as the run
  you resume (`max_steps=`, `batch_size=`, ...). If the warmup (5% of `max_steps`) rounds to
  zero in one run but not the other, the scheduler is rebuilt as the wrong type: `KeyError:
  '_schedulers'` in one direction, a silently different LR curve in the other.
- **`additional_steps` extends the step budget, not the LR curve.** `max_steps` becomes
  `N + M`, but the stored cosine keeps its original length; past the original end it climbs
  back up. To pick up after a crash, leave it out (the unchanged config gives back the
  original `max_steps`) or set it to exactly the remaining steps. To train longer, use the
  LR-anneal form. A resume with nothing left to do warns `training will exit immediately`,
  but still writes a full state (~80 GB) and weights file into the new run directory.
- **`batch_size` must match the run you resume.** The sampler stores `batch_in_epoch`
  (a batch count, not a sample count), so changing `batch_size` silently resumes from the
  wrong position in the epoch. No error, no crash.

Full semantics — the resume/anneal/fine-tune table, how `max_steps` is derived, the LR
formula for an anneal, and the `trainer_state.json` arithmetic for changing batch size
mid-run: [Resume, fine-tune and LR anneal](./docs/GUIDE.md#resume-fine-tune-and-lr-anneal).

---

## Deploy on TA2

Read [Part 2 of the guide](./docs/GUIDE.md#part-2--deployment) before touching a real robot —
in particular [Safety](./docs/GUIDE.md#safety). The short version follows; the server runs on
the GPU host, the client on the robot host.

### Step 6. Move the checkpoint to the deploy host

A checkpoint alone is not servable. These must travel together, from the **same run**:

| File | Why |
|---|---|
| `checkpoints/weights/step_*.pt` | the weights (~12 GB) |
| `dataset_stats.json` | action/state normalization — mixing runs silently drifts the action scale |
| `config.yaml` | the run's resolved config (provenance only, optional) |
| `text_embeds_cache/` entries | the conditioning, byte-identical to training — the directory is the data config's `text_embedding_cache_dir`, which need not equal the task name (the shipped example uses `mkfloor2to8_20hz` for task `mkfloor2to8_20hz_mono`) |
| `Wan2.2_VAE.safetensors` | ~1.4 GB, **not** inside the `.pt` (only `mot` + `proprio_encoder` are saved) |

The DiT shards (19 GB), ActionDiT payload (2 GB) and T5 encoder (11 GB) do **not** need to be
copied — serving sets `skip_dit_load_from_pretrain=True` and reads cached text embeddings.

Copy them with whatever moves files between your machines, keeping the repo-relative layout
so that `start_local_serve_ws.sh` finds the newest run and highest step on its own. For
example, from the repo root on the training server:

```bash
rsync -aR runs/<task>/<RUN_ID>/dataset_stats.json runs/<task>/<RUN_ID>/config.yaml \
  runs/<task>/<RUN_ID>/checkpoints/weights/step_<N>.pt \
  data/text_embeds_cache/<cache dir from the data config>/ \
  checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors \
  <deploy-host>:/path/to/FastWAM/
```

`config.yaml` is provenance only; serving rebuilds the model from `configs/task/<task>.yaml`,
which must match the one used for training. The full list with sizes, and the directory
layout the server expects:
[What a deploy machine actually needs](./docs/GUIDE.md#what-a-deploy-machine-actually-needs).

### Step 7. Start the server (GPU host)

```bash
TASK=<your_task> bash start_local_serve_ws.sh 10     # 10 = denoising steps
```

`RUN` and `STEP` default to the newest run and the highest step number; override either to pin
a specific checkpoint. Wait for `WebSocket server listening on ws://0.0.0.0:8000` before
starting the client. Health check: `curl http://<host>:8000/healthz`. The script runs
`$PYTHON_BIN` (default: `python` on `PATH`, so activate the env first); every variable it
reads (`RUN`, `STEP`, `CHECKPOINT`, `DATASET_STATS`, `TASK_MAP`, `PROMPT_TASK`, `TEXT_EMBED`,
`CONFIG_OVERRIDES`, `HOST`/`PORT`, ...) is tabulated under
[Startup order](./docs/GUIDE.md#startup-order). It already warms CUDA Graphs for 8, 10 and
12 steps.

`TASK` must name the config the checkpoint was **trained** with — hydra recomposes the data
and model dimensions from it, so a mismatch fails at load time (by design; it used to fail
much later and less clearly).

A deploy machine usually has the text embeds but not the datasets, so the startup
instruction cannot be read from `meta/tasks.jsonl`. In that case pass the embed directly —
`taskmap.json` maps each short name to its `<sha256>`, so use it to pick the file. A
checkpoint whose lift mode differs from the task default takes `CONFIG_OVERRIDES`:

```bash
TASK=<your_task> TEXT_EMBED=data/text_embeds_cache/<task>/<sha256>.t5_len128.wan22ti2v5b.pt \
  bash start_local_serve_ws.sh 10
TASK=<your_task> CONFIG_OVERRIDES="lift=legacy" TEXT_EMBED=... \
  bash start_local_serve_ws.sh 10      # pre-split 17/15 run
```

### Step 8. Start the client (robot host)

With ROS 2 up and the arms zeroed (the policy assumes a sane starting pose):

```bash
cd experiments/teleavatar_v2_deploy/client
./run_task_ws.sh <taskmap-key> --dry-run      # infer only, nothing moves
./run_task_ws.sh <taskmap-key>                # live
./run_task_ws.sh                              # omit the key: server's startup instruction
```

**Always do a `--dry-run` pass first.** The script also records a rosbag of camera frames,
policy action chunks and robot state for the whole session, which is what the analysis tools
below consume.

`SERVER_HOST` / `SERVER_PORT` (default `127.0.0.1` / `8000`) point it at the GPU host; the
other variables are tabulated under [Startup order](./docs/GUIDE.md#startup-order). If
something does not come up, the symptom table in
[Troubleshooting](./docs/GUIDE.md#troubleshooting) covers the common cases.

### Optional: a task library for switching instructions

To select among several instructions by short name at runtime, generate a task map on the
GPU host before starting the server:

```bash
python experiments/teleavatar_v2_deploy/server/make_task_map.py \
  --dataset-dir data/<your_dataset>_mono \
  --cache-dir data/text_embeds_cache/<your_task> \
  --out taskmap.json
```

Keys are derived from the dataset directory name (`_lerobot_20fps_mono` etc. stripped); a
dataset recording several instructions gets one key per `task_index`. Add or override entries
with `--instruction 'key=full instruction text'`. `taskmap.json` is a generated,
per-deployment artifact and is gitignored.

---

## Analysis tools

All under `experiments/teleavatar_v2_deploy/server/`; run any of them with `--help`. The ones
that run the model need `--task` plus a matching `--checkpoint` / `--dataset-stats` pair from
the same run; the rest only read a bag or a stream.

| Script | What it answers | Runs the model |
|---|---|---|
| `analyze_deploy_bag.py` | predicted vs commanded vs measured joint traces from a rosbag | no |
| `joint_error_report.py` | per-joint tracking error tables | no |
| `rtp_stream_export.py` | dump the RTP video stream to files | no |
| `bag_visualize_and_predict.py` | re-run the policy over a recorded bag, render a comparison video | yes |
| `offline_infer_from_export.py` | replay exported frames without a robot | yes |
| `viz_trainset_pred_compare.py` | sanity check: prediction vs ground truth on training samples | yes |

Commands and flags: [Recording and analysing rosbags](./docs/GUIDE.md#recording-and-analysing-rosbags).
Inference latency per task and step count: `bash bench_latency_ws.sh` — see
[Latency](./docs/GUIDE.md#latency).

---

## Data conventions

**Action / state layout.** Datasets store 72-d vectors;
`TeleavatarSelectTransform` slices them to the openpi convention:

```
raw 72-d:  positions[0:16]  = [L_arm(7), L_grip_pos, R_arm(7), R_grip_pos]
           velocities[16:32]
           efforts[32:48]   = [L_arm(7), L_grip_effort, R_arm(7), R_grip_effort]
           + optional EE / chassis / ...

model:     state  14-d = [L_arm(7), R_arm(7)]
           action 16-d = [L_arm(7), L_grip, R_arm(7), R_grip]
           action 17-d = the above + lifting-column command (lift=on)
```

The lifting column is a hydra config group, `lift=off|on` (`configs/lift/`), chosen in the
task config's defaults. `lift=on` adds the lift command to the action but never feeds the
measured velocity back as state, because no lift state topic exists at deploy time. See
[docs/GUIDE.md](./docs/GUIDE.md#the-lifting-column-liftoffon).

Gripper channels are **force**-controlled: the dataset stores effort in N·m, converted to a
platform trigger in [0, 1] before normalization and converted back on the robot side.

**Paths are repo-relative.** Scripts derive their own root; no absolute path is baked in, and
the layout under the repo root is fixed. To serve files kept elsewhere, point
`CHECKPOINT`, `DATASET_STATS`, `TEXT_EMBED` and `DIFFSYNTH_MODEL_BASE_PATH` at them (see
[Startup order](./docs/GUIDE.md#startup-order)).

**Hydra composition** (`configs/train.yaml` → `data`/`model` config → task config → command
line) and why `# @package _global_` matters: [The config chain](./docs/GUIDE.md#the-config-chain).

---

## Further reading

| Document | Contents |
|---|---|
| [docs/GUIDE.md](./docs/GUIDE.md) | base-weight download sources, config chain in depth, launcher internals, lifting column, resume/anneal semantics, batch-size arithmetic, checkpoint transfer, two-host deployment, startup order and variables, safety, rosbag analysis, latency, troubleshooting |
| [upstream repo](https://github.com/zhou-yh19/FastWAM) | original FastWAM: LIBERO / RoboTwin training and evaluation |

---

## Citation

This fork adds LeRobot ingestion and TA2 deployment; the model and method are from the FastWAM
paper. If you use this work, please cite:

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026},
  url={https://arxiv.org/abs/2603.16666}
}
```

The RoboTwin evaluation code is adapted from the
[RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin).

---

## License

The code is released under the [MIT License](./LICENSE). Some files were copied or adapted
from other open-source projects — DiffSynth-Studio, LeRobot, openpi, msgpack-numpy,
PyTorch3D, robosuite, RoboTwin and one NVIDIA file — and stay under their original licenses
(Apache-2.0, BSD-3-Clause or MIT). [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) lists
each one with its files and copyright; the full license texts are in [LICENSES/](./LICENSES/).

The base weights are not part of this repository. They are downloaded under their own
licenses: Wan2.2-TI2V-5B, the Wan2.1 umT5 tokenizer and DiffSynth-Studio's converted
VAE/T5 files are all Apache-2.0.
