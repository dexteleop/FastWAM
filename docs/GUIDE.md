# FastWAM TA2 — Training & Deployment Guide

Reference for the details [README.md](../README.md) only summarises. Nothing here is tied to a
particular dataset or task.

- [Part 1 — Training](#part-1--training)
  - [The config chain](#the-config-chain)
  - [Launching](#launching)
  - [Onboarding a new dataset](#onboarding-a-new-dataset)
  - [The lifting column](#the-lifting-column-liftoffon)
  - [Resume, fine-tune and LR anneal](#resume-fine-tune-and-lr-anneal)
  - [Checkpoint management and transfer](#checkpoint-management-and-transfer)
  - [Monitoring](#monitoring)
- [Part 2 — Deployment](#part-2--deployment)
  - [Two machines, two interpreters](#two-machines-two-interpreters)
  - [What a deploy machine actually needs](#what-a-deploy-machine-actually-needs)
  - [Startup order](#startup-order)
  - [Safety](#safety)
  - [Recording and analysing rosbags](#recording-and-analysing-rosbags)
  - [Latency](#latency)
- [Troubleshooting](#troubleshooting)

---

# Part 1 — Training

## The config chain

Hydra always enters through `configs/train.yaml`. That file leaves `data`, `model` and `task`
all `null`, so **`task=` is not optional** — without it the composed config has no `data` or
`model` key at all, only placeholder defaults like `batch_size: 2`.

```
scripts/train.py
  @hydra.main(config_path="../configs", config_name="train")   ← always train.yaml
        │
        │  command line passes task=<your_task>
        ▼
configs/task/<your_task>.yaml
  # @package _global_          ← hydra directive, not a comment (see below)
  defaults:
    - /lift: "off"                         ── action/state widths (see lifting column)
    - override /data: <your_data_config>   ─┐
    - override /model: fastwam             ─┼─ the task config picks these
    - _self_                                │  (_self_ last = this file wins)
        │                                   │
        ▼                                   ▼
configs/data/<your_data_config>.yaml   configs/model/fastwam.yaml
  dataset_dirs / shape_meta / processor    architecture, pretrained paths
```

Override precedence, low to high: `train.yaml` → `data`/`model` config → task config →
command line.

**`# @package _global_` must be the first line of every task config.** Without it every key
nests under `task.` and composition fails with `Could not override 'data@task.data'`.
`launch_train.sh` pre-checks this, because it is the most common mistake and the error
message does not point at the cause.

### What `train.yaml` provides

Defaults every task config inherits unless it overrides them. All of these are read by code:

| Field | Default | Consumer |
|---|---|---|
| `batch_size` / `num_workers` | 2 / 4 | placeholders — real runs must override |
| `mixed_precision` | bf16 | `runtime.py` |
| `max_grad_norm` | 1.0 | `trainer.py` |
| `dist_timeout_sec` | 600 | `trainer.py` — how long a NCCL collective may stall before the watchdog aborts. Shorter than DeepSpeed's 1800 s default so a rank divergence surfaces in minutes |
| `eval_seed` / `eval_num_loss_*` | 12345 / 5 / 2 | pins eval samples and noise so consecutive evals are comparable |
| `resume*` | null / false | see [Resume](#resume-fine-tune-and-lr-anneal) |
| `wandb.enabled` | false | task configs must opt in |
| `output_dir` | `./runs/train/<timestamp>` | always overwritten by the launcher |

It also sets `hydra.job.chdir: false`. That matters: hydra 1.2+ otherwise changes the working
directory, which breaks every `./data/...` relative path in the configs.

Change `train.yaml` only for things that should affect **all** tasks. Task-specific values
belong in the task config.

## Launching

```bash
bash scripts/launch_train.sh <your_task>
```

Starts a tmux session with two windows: `train` and `prune`. The second is **not optional** —
each ZeRO state snapshot is ~80 GB and will fill the disk without it.

| Variable | Default | Meaning |
|---|---|---|
| `ZERO` | 1 | DeepSpeed stage (1 or 2) |
| `NPROC` | 8 | GPUs per node |
| `KEEP` | 2 | state snapshots to retain |
| `SESSION` | `fastwam_<task>` | tmux session name |
| `CONDA_SH` / `CONDA_ENV` | `~/miniconda3/...` / `fastwam` | conda activation |
| `DRY_RUN` | — | print the command, start nothing |

Beyond the launcher, it pre-checks that the task config exists and starts with
`# @package _global_`, validates `lift=`, verifies wandb credentials when wandb is enabled,
refuses to start a duplicate session, and
pins `RUN_ID` so the pruner watches the right directory.

The stage comes from `ZERO`, **not** from the config file name:

```bash
bash scripts/launch_train.sh <your_task>            # ZeRO-1
ZERO=2 bash scripts/launch_train.sh <your_task>     # ZeRO-2
ZERO=2 NPROC=4 bash scripts/launch_train.sh <your_task>
```

| ZERO | launcher | accelerate config | DeepSpeed JSON |
|---|---|---|---|
| 1 | `scripts/train_zero1.sh` | `accelerate_zero1_ds.yaml` | `ds_zero1_config.json` |
| 2 | `scripts/train_zero2.sh` | `accelerate_zero2_ds.yaml` | `ds_zero2_config.json` |

Resuming may switch stage — DeepSpeed rebuilds the shard layout on load. `batch_size` may
**not** change; see below.

Without the wrapper:

```bash
bash scripts/train_zero1.sh 8 task=<your_task>
#                          ^  ^
#                    GPUs per node   configs/task/<name>.yaml
```

Use `DRY_RUN=1` to inspect the expanded command before committing 8 GPUs.

## Onboarding a new dataset

### 1. Transcode

Source video is typically 3840×1920 HEVC. Decoding one 33-frame window costs ~3.2 CPU-seconds
and starves the GPU (~94 % of wall time waiting on data). The source aspect ratio already
matches the target tiles, so rescaling is equivalent to the letterbox the processor would
apply anyway.

```bash
DATASETS="<your_dataset>" bash scripts/transcode.sh              # original SBS -> _lowres
DATASETS="<your_dataset>" MODE=mono bash scripts/transcode.sh    # _lowres -> _mono

python scripts/patch_transcoded_meta.py --mode lowres data/<your_dataset>_lowres
python scripts/patch_transcoded_meta.py --mode mono   data/<your_dataset>_mono
```

| Stage | head_camera | left/right_color |
|---|---|---|
| `MODE=lowres` | 3840×1920 → 512×256 | 2560×800 → 256×80 |
| `MODE=mono` (left eye crop) | 512×256 → 256×256 | 256×80 → 128×80 |

`DATASETS` takes base names without the `_lowres` / `_mono` suffix; omit it and the script
discovers datasets carrying the right source suffix. Output keeps the LeRobot layout —
`meta/` and `data/` are symlinked back to the source, only `videos/` is rewritten — and
re-running skips completed files.

**The `patch_transcoded_meta.py` step is mandatory.** Without it `meta/info.json` still
advertises the source resolution and LeRobot reads the wrong geometry.

> Do not delete `data/*_lowres` after producing `_mono`: the mono directories symlink their
> `meta/` and `data/` (the parquet files) back into the lowres copy. Only its `videos/` is
> redundant.

### 2. Two config files

```bash
cp configs/data/ta2_mono_template.yaml configs/data/<your_task>.yaml
cp configs/task/ta2_mono_template.yaml configs/task/<your_task>.yaml
```

In the **data** config: `dataset_dirs` (one or more `*_mono` dirs) and
`text_embedding_cache_dir`. Camera `raw_shape` / `shape` under `shape_meta.images` must match
the transcoded video exactly — a mismatch silently produces wrong results rather than an
error. Leave `pretrained_norm_stats: null` for a fresh run.

In the **task** config: point `override /data:` at your data config name, set `wandb`, adjust
`batch_size` / `num_epochs` / `learning_rate`.

Both templates document every field inline.

### 3. Precompute text embeddings

```bash
python scripts/precompute_text_embeds.py task=<your_task>
```

Reads instructions from each dataset's `meta/tasks.jsonl`, encodes them, and writes
`<sha256 of formatted prompt>.t5_len128.wan22ti2v5b.pt` into `text_embedding_cache_dir`. This
is what lets training and serving skip the ~11 GB umT5-XXL encoder entirely.

Because the filename is a hash of the exact prompt string, **any** wording change — including
whitespace — produces a file the trainer will not find. If you regenerate a dataset and the
instruction text shifts even slightly, re-run this step.

Host RAM: the encoder is built on the meta device and adopts the loaded bf16 weights, so a
process needs roughly the 11 GB of weights and no more (it ran in a 15 GB cgroup cap). Under
`torchrun` every rank loads its own copy at once, so multiply by `--nproc_per_node`.

## The lifting column (`lift=off|on`)

Channel 71 of the raw 72-d vector is the lifting column (`kinco_velocity` in `info.json`):
the **command** in `action`, the **measured velocity** in `observation.state`. The hydra
config group `configs/lift/{off,on,legacy}.yaml` controls it. Each file defines `shape_meta`,
`*_output_dim` and the transform flags in one place, so data configs never repeat them.

| | `lift=off` | `lift=on` |
|---|---|---|
| state | 14 = `[L_arm(7), R_arm(7)]` | 14 = `[L_arm(7), R_arm(7)]` |
| action | 16 = `[L_arm(7), L_grip, R_arm(7), R_grip]` | 17 = the above + `lift_vel_cmd` |

Set the default in the task config (hydra wants non-override entries before `override`
ones) and switch on the command line without touching any yaml:

```yaml
defaults:
  - /lift: "on"          # or "off"
  - override /data: <your-data-config>
  - override /model: fastwam
  - _self_
```

```bash
bash scripts/launch_train.sh <your_task>             # the task config's default
bash scripts/launch_train.sh <your_task> lift=off    # 16-d instead
```

**`lift=on` puts the lift command in the action, never the measured velocity in the
state.** The deploy interface (dev docs §4.8) has no lift state topic, so on the robot that
channel can only be filled with 0. In the training data it tracks the command with a lag-0
correlation of 0.9978, so a model given it copies it instead of looking at the images. At
deploy time the constant 0 normalizes to −0.0604, next to the −0.0584 of a stationary column
in training: the policy reads "not moving" every frame and never moves the column. That is
exactly what the first lift-enabled deployment did. Keeping lift on the output side only
removes both the shortcut and the train/deploy skew.

Runs from before 2026-09-12 predate the split and
are 17/15: the state also carries the measured lift velocity. Load them with **`lift=legacy`**
(`--config-override lift=legacy` on the server and offline tools). Otherwise the
checkpoint-dimension check fails before the model is built. Do not train new runs with it: that
state channel is the shortcut described above. `TeleavatarSelectTransform(include_lift=True)`
is the deprecated spelling of the same 17/15 transform and warns once. New code uses the
independent `lift_in_action` / `lift_in_state` flags, or the config group.

- **Enable it only if the dataset moves the column.** It is constant 0 on every MK_tower
  floor, and turning it on there only feeds the model a constant column. Check with:
  ```bash
  python -c "
  import glob, numpy as np, pandas as pd
  fs = sorted(glob.glob('data/<your_dataset>/data/chunk-000/*.parquet'))[:40]
  a = np.concatenate([np.stack(pd.read_parquet(f, columns=['action'])['action'].values) for f in fs])[:, 71]
  print('absmax', abs(a).max(), 'nonzero_frac', (abs(a) > 1e-6).mean())"
  ```
- **A 16-d checkpoint cannot be resumed as 17-d.** ActionDiT's `action_encoder` / `head`
  are `Linear(action_dim, ...)`, so accelerate refuses the shape change. Starting from the
  ActionDiT pretrained backbone is fine: that payload excludes both layers
  (`ACTION_BACKBONE_SKIP_PREFIXES`), and they initialise randomly.
- **Recompute stats.** Set `pretrained_norm_stats: null`: 16/14-d stats do not fit a 17-d
  model, and the normalizer rejects the width mismatch at load time.
- **No scaling.** The channel stays in dataset units (motor-side target speed, rad/s,
  saturating at ±150); the min/max normalizer maps it to [−1, 1].

## Resume, fine-tune and LR anneal

Training and resuming share one command; only config fields differ.

| Goal | `resume` | `resume_reinit_lr` | `additional_steps` |
|---|---|---|---|
| Crash recovery (continue the LR curve) | directory `state/step_<N>` | `false` | required |
| LR anneal to finish | directory `state/step_<N>` | `true` | required (raises if absent) |
| Fine-tune on new data | file `weights/step_<N>.pt` | ignored | ignored |

A **directory** path restores everything: weights, Adam momentum, LR schedule, dataloader
cursor and RNG state. A **file** path loads weights only — optimizer, step counter and data
progress all reset. The trainer warns and ignores the other two fields in that case.

`resume_reinit_lr: false` keeps the scheduler stored in the state and carries on decaying;
`true` discards it and builds a fresh curve over `(learning_rate, max_steps - global_step)`.

```bash
bash scripts/launch_train.sh <your_task> \
  resume=./runs/<your_task>/<RUN_ID>/checkpoints/state/step_<N> \
  additional_steps=<M>
```

### Two fields that fail quietly

**`additional_steps` is required, or training exits immediately.** The trainer computes
`max_steps = global_step + additional_steps`. Leave it null and, if the `num_epochs`-derived
`max_steps` is already ≤ the current step, you get a **warning** and a clean exit — not an
error:

```
Resumed at global_step=<N> with max_steps=<M>; training will exit immediately.
Set additional_steps=N or a larger max_steps to continue.
```

So the value is "how many more steps", not an absolute target.

**Changing `batch_size` silently replays data.** The sampler computes:

```python
sample_offset = resume_batch_offset * batch_size * num_processes
indices = indices[sample_offset:]
```

`trainer_state.json` stores `batch_in_epoch` — a **batch** count, not a sample count. Change
`batch_size` and the reconstructed position moves with it. Nothing errors; training simply
continues from the wrong place in the epoch.

To change it anyway (e.g. halving `batch_size` and doubling `gradient_accumulation_steps` to
save memory at the same global batch), either accept the offset, or rescale the stored count:

```bash
S=runs/<task>/<RUN_ID>/checkpoints/state/step_<N>/trainer_state.json
cp "$S" "$S.bak"
python -c "
import json,sys; p=sys.argv[1]; d=json.load(open(p))
d['batch_in_epoch'] = d['batch_in_epoch'] * <old_batch_size> // <new_batch_size>
json.dump(d, open(p,'w'), indent=2); print(d)
" "$S"
```

`gradient_accumulation_steps` does not enter that formula (`batch_in_epoch` counts
micro-batches), so changing it alone is safe.

### Choosing `learning_rate` for an anneal

Use **the LR in effect at the resume point**, not the original peak. Passing the peak reheats
already-converged weights — a mistake that has been made here before.

For a cosine schedule with warmup:

```
warmup  = int(total_steps * warmup_frac)
T_max   = total_steps - warmup
eta_min = peak_lr * 0.01
p  = (resume_step - warmup) / T_max
lr = eta_min + (peak_lr - eta_min) * (1 + cos(pi * p)) / 2
```

Set `resume_warmup_frac: 0.0` for a final anneal (monotone decay). Use ~`0.05` instead when
lowering the peak LR mid-training to escape a loss plateau, so the new curve joins smoothly.

### Recovering a lost config

Every run directory holds hydra's fully resolved snapshot:

```bash
grep -E "^(resume|resume_reinit_lr|additional_steps|resume_warmup_frac|learning_rate|batch_size|gradient_accumulation_steps):" \
  runs/<task>/<RUN_ID>/config.yaml
```

## Checkpoint management and transfer

### Pruning ZeRO state

Each `state/step_<N>/` is ~80 GB. `launch_train.sh` starts `scripts/prune_states.py --watch`
alongside training, keeping the newest `KEEP` (default 2). Run it manually if you started
training without the wrapper.

Weights (`weights/step_<N>.pt`, ~12 GB) are never pruned — only optimizer state is.

### A checkpoint is not self-contained

`save_checkpoint` stores only two things: `mot` and `proprio_encoder`. **The VAE is not in the
`.pt` file** and loads separately.

| File | Size | Needed | Why |
|---|---|---|---|
| `runs/<task>/<RUN_ID>/checkpoints/weights/step_<N>.pt` | 12 G | yes | mot + proprio_encoder |
| `runs/<task>/<RUN_ID>/dataset_stats.json` | ~100 K | yes | action/state denormalization |
| `configs/{train,task,data,model}` | small | yes | serving recomposes via hydra from `task=` |
| `data/text_embeds_cache/<task>/` | ~1 M | yes | precomputed T5 context |
| `checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors` | 1.4 G | **yes** | absent from the `.pt` |
| `taskmap.json` | <1 K | optional | only for selecting instructions by name |
| `runs/<task>/<RUN_ID>/config.yaml` | small | optional | provenance; serving does not read it |
| `checkpoints/Wan-AI/Wan2.2-TI2V-5B/*.safetensors` | 19 G | no | `skip_dit_load_from_pretrain=True` |
| `checkpoints/ActionDiT_*.pt` | 2 G | no | same |
| T5 encoder safetensors | 11 G | no | only with `--load-text-encoder` |

About 13.4 GiB, not 33 — skipping those 32 GB is the point of the design.

**`dataset_stats.json` must come from the same run as the weights.** Two runs with matching
dimensions but different statistics load without complaint and produce subtly wrong actions.
Vector widths are validated; the values are not.

### Transfer

Any file-transfer tool works; keep the repo-relative paths (`runs/<task>/<RUN_ID>/...`,
`data/text_embeds_cache/<task>/`) so the deploy scripts find everything without extra
variables. See the README's [Moving checkpoints between machines](../README.md#moving-checkpoints-between-machines)
for an `rsync -R` example.

Datasets travel the same way. Copy each `*_lowres` directory together with its `*_mono`
one: the mono copy symlinks `meta/` and `data/` back into it.

## Monitoring

A healthy training line:

```
speed=0.51 step/s, 65.67 samples/s
data_wait=0.0s/0.0s(mean/max)    # near zero = compute-bound
compute=1.9s  stall=1%
```

| Symptom | What to do |
|---|---|
| `data_wait` clearly non-zero | confirm video was transcoded to `_mono`; raise `num_workers` to 12–16; check data-disk contention |
| `stall` > 5 % | NCCL communication is slow — check the network or switch ZeRO stage |
| NCCL hang | `train_zero*.sh` sets `TORCH_NCCL_TRACE_BUFFER_SIZE` / `DUMP_ON_TIMEOUT` / `DESYNC_DEBUG`, so a timeout dumps per-rank stacks; `dist_timeout_sec: 600` surfaces it in 10 minutes rather than 30 |
| missing text embedding | the precompute step was skipped, or the instruction text changed |
| disk full | `prune_states.py` is not running |
| unreadable hydra error | re-run with `HYDRA_FULL_ERROR=1` |

---

# Part 2 — Deployment

## Two machines, two interpreters

| Role | Runs | Interpreter |
|---|---|---|
| GPU host | `serve_policy_ws.py` (WebSocket + msgpack, port 8000) | conda `fastwam` env |
| Robot host | `run_client_ws.py`, ROS 2 topics, RTP video decode | **system `/usr/bin/python3`** |

The split is not a preference. ROS 2 Humble's C extensions are built for the system
interpreter (3.10); conda's python cannot import `rclpy` or `gi`. `rclpy` comes from
`setup.bash`'s `PYTHONPATH`, not from dist-packages, so sourcing it is required.

Robot-host dependencies (`openpi_client`, `websockets`, `msgpack`) go in that interpreter's
user site — see `experiments/teleavatar_v2_deploy/client/requirements.txt`. GStreamer needs
the H.265 decoder plugin (`gst-inspect-1.0 nvh265dec`).

Pre-flight check: `bash experiments/teleavatar_v2_deploy/check_deployment.sh`.

## What a deploy machine actually needs

`serve_policy.py` resolves everything relative to `PROJECT_ROOT`, and the intermediate
directory names are load-bearing:

```
runs/<task>/<RUN_ID>/checkpoints/weights/step_<N>.pt
runs/<task>/<RUN_ID>/dataset_stats.json
configs/task/<task>.yaml         # serving recomposes via hydra; the run's config.yaml is not read
configs/data/<data>.yaml
configs/model/fastwam.yaml
data/text_embeds_cache/<task>/
taskmap.json                     # optional
checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
```

`DIFFSYNTH_MODEL_BASE_PATH` points at `checkpoints/`; the loader appends the
`DiffSynth-Studio/...` path itself, so none of those directory names may be renamed.

## Startup order

**1. GPU host — start the server**

```bash
TASK=<your_task> bash start_local_serve_ws.sh 10      # 10 = denoising steps
```

`RUN` and `STEP` default to the newest run and highest step; override to pin a specific
checkpoint, or pass `CHECKPOINT` / `DATASET_STATS` directly. Wait for
`WebSocket server listening on ws://0.0.0.0:8000`.

`--task` must name the config the checkpoint was **trained** with: hydra recomposes the data
and model dimensions from it, so a mismatch fails at load.

**2. Verify**

```bash
curl http://<gpu-host>:8000/healthz      # expect OK
```

**3. Optional — build a task library**

```bash
python experiments/teleavatar_v2_deploy/server/make_task_map.py \
  --dataset-dir data/<your_dataset>_mono \
  --cache-dir data/text_embeds_cache/<your_task> \
  --out taskmap.json
```

Keys derive from the dataset directory name; a dataset with several instructions gets one key
per `task_index`. Override with `--instruction 'key=text'`. This file is generated per
deployment and gitignored.

**4. Robot host — ROS 2 up**, then zero the arms before handing control to the policy.

**5. Robot host — start the client**

```bash
cd experiments/teleavatar_v2_deploy/client
./run_task_ws.sh <taskmap-key> --dry-run     # infer only, nothing moves
./run_task_ws.sh <taskmap-key>               # live
./run_task_ws.sh                             # server's startup instruction
```

Useful variables: `SERVER_HOST` / `SERVER_PORT`, `BAG_ROOT`, `ROS_DOMAIN_ID`, `ROS_SETUP`,
`PYTHON_BIN`.

## Safety

- **Always `--dry-run` first.** It runs the full inference path and publishes nothing.
- Keep the e-stop within reach; keep the workspace clear of people.
- Zero the arms before every session — the policy assumes a sane starting pose.
- Check the startup log's canvas size. If it does not match the geometry the checkpoint was
  trained on, **stop**: the task config is pointing at the wrong data config.
- One server per GPU. `PORT` is configurable, but each server holds ~14 GB of GPU memory, so
  a second one does not fit on a 24 GB card.
- The client infers synchronously in the control loop: it blocks for one inference every
  `--open-loop-horizon` steps. Budget accordingly at 20 Hz.
- **Lifting column (17-d policies).** The client takes the column over automatically when
  the server reports a lift channel, and republishes `/api/servo/cmd` at 20 Hz on its own
  keep-alive timer. §4.6 requires ≥ 10 Hz, disables the servo after 3 s of silence, and does
  not use the FSM heartbeat. On exit it sends `[0.0, 0.0]` before stopping the arms. On
  first power-up, run with `--lift-dry-run`, which logs the command without moving the
  column, then lower `lift.max_normalized_velocity` in `arm_config.yml` to confirm the
  direction. `--lift-mode off` never drives the column, even with a 17-d checkpoint.
- **The lift sign is inverted.** §4.6 defines motor target speed = −normalized_velocity ×
  150 rad/s, while the dataset records the motor side, so the client publishes
  `normalized_velocity = −action[16] / 150` (`lift.cmd_scale` / `lift.cmd_sign`). A positive
  dataset value means the column goes **down**. Both the §4.6 formula and the recorded video
  confirm this: in one lift recording the dataset has `action[71] > +100` at t = 3.75–13.05 s and
  the column lowers. Getting the sign wrong drives the column the wrong way.

## Recording and analysing rosbags

`run_task_ws.sh` records for the whole session automatically — camera frames, policy action
chunks, joint/gripper state, and the commands actually published. `record_fastwam_bag.sh`
does the same standalone.

The `/fastwam/policy/*` topics are what make predicted-vs-commanded-vs-measured analysis
possible; without them a bag only supports commanded-vs-measured.

```bash
python experiments/teleavatar_v2_deploy/server/analyze_deploy_bag.py <bag dir>
python experiments/teleavatar_v2_deploy/server/joint_error_report.py <bag dir>

# re-run the policy over a recorded bag (needs a checkpoint)
python experiments/teleavatar_v2_deploy/server/bag_visualize_and_predict.py \
  --task <your_task> \
  --checkpoint runs/<task>/<RUN_ID>/checkpoints/weights/step_<N>.pt \
  --dataset-stats runs/<task>/<RUN_ID>/dataset_stats.json \
  <bag dir>
```

`offline_infer_from_export.py` replays exported frames without a robot;
`viz_trainset_pred_compare.py` checks predictions against ground truth on training samples.

## Latency

```bash
bash bench_latency_ws.sh                      # single point, server's default instruction
bash bench_latency_ws.sh <taskmap-key> 12     # specific task and step count
bash bench_latency_ws.sh --sweep              # every served task x 8/10/12 steps
```

The sweep reads `available_tasks` from the server's connection metadata, so it needs no task
list of its own. Denoising steps are per-request and need no restart, but start the server
with `--warmup-steps 8 10 12` before comparing values so every CUDA Graph is warm.

`bench_infer_ws.py` drives the same `openpi_client` stack the real client uses, with synthetic
frames and no ROS 2 — nothing moves.

---

# Troubleshooting

| Symptom | Check | Cause / fix |
|---|---|---|
| Canvas size in the startup log is unexpected | first server log line | task config points at the wrong data config — stop before moving anything |
| `Cannot reach policy server` | `curl http://<ip>:8000/healthz` | server down, firewall, or `--host` left at `127.0.0.1` |
| `ModuleNotFoundError: websockets` | which interpreter | client must run on system `python3`, not conda |
| `ROS2 interface not initialized` | `ros2 topic list` | ROS bridge not running, or `ROS_DOMAIN_ID` mismatch |
| `Timeout waiting for sensor data` | `ros2 topic hz /left_arm/joint_states`, `gst-inspect-1.0 nvh265dec` | RTP not arriving on 8890, or GStreamer plugin missing |
| Action dimension mismatch at load | — | checkpoint, `dataset_stats.json` and task config must all come from the same run |
| Server cannot find the VAE | `ls checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/` | the VAE is not inside the `.pt`; copy it separately |
| Task name rejected at startup | server log | the key is not in `taskmap.json` — regenerate it with `make_task_map.py` |
| Training exits right after resuming | trainer warning | `additional_steps` not set |
| Resumed run seems to repeat data | `trainer_state.json` | `batch_size` differs from the original run |
| Text embedding not found | `ls data/text_embeds_cache/<task>/` | precompute step skipped, or instruction text changed |
