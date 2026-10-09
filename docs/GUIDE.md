# FastWAM TA2 — Training & Deployment Guide

Reference for the details [README.md](../README.md) only summarises: the README is the happy
path, one command per step; this file is where each command's variants, semantics and failure
modes are spelled out once. Nothing here is tied to a particular dataset or task.

- [Part 1 — Training](#part-1--training)
  - [Base weights and download sources](#base-weights-and-download-sources)
  - [The config chain](#the-config-chain)
  - [Launching](#launching)
  - [Onboarding a new dataset](#onboarding-a-new-dataset)
  - [The lifting column](#the-lifting-column-liftoffon)
  - [Resume, fine-tune and LR anneal](#resume-fine-tune-and-lr-anneal)
  - [Pruning ZeRO state](#pruning-zero-state)
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

## Base weights and download sources

The README's [Base weights](../README.md#base-weights) section gives the directory tree,
the ModelScope auto-download and the ActionDiT preprocess command. This section covers the
alternative download route.

Hugging Face: `DIFFSYNTH_DOWNLOAD_SOURCE=huggingface` on its own fails at the first file with
`RepositoryNotFoundError: 401`, because the converted VAE/T5 come from
`DiffSynth-Studio/Wan-Series-Converted-Safetensors`, which exists only on ModelScope. Also
set `redirect_common_files: false` in the model config (`configs/model/fastwam.yaml`) to load
the original `Wan2.2_VAE.pth` (fp32; identical to the converted file after the bf16 cast) and
`models_t5_umt5-xxl-enc-bf16.pth` from `Wan-AI/Wan2.2-TI2V-5B` instead; both then live
under `checkpoints/Wan-AI/Wan2.2-TI2V-5B/`, so copy that VAE path when
[moving checkpoints](../README.md#step-6-move-the-checkpoint-to-the-deploy-host).

Either way, `DIFFSYNTH_SKIP_DOWNLOAD=true` forbids downloading, and a missing file then
surfaces later as `Cannot detect model type ... File: []`.

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

### How `max_steps` is derived

`max_steps` is derived when left `null`:
`steps/epoch = ceil(ceil(len(dataset) / (batch_size × n_gpu)) / grad_accum)`,
and `len(dataset)` is the **frame count**. Sliding windows overlap heavily, so "one epoch"
is not "every independent sample once".

## Launching

```bash
bash scripts/launch_train.sh <your_task>
```

Starts a tmux session with two windows: `train` and `prune`. The second is **not optional** —
each ZeRO state snapshot is ~80 GB and will fill the disk without it. The environment
variables it reads (`ZERO`, `NPROC`, `KEEP`, `SESSION`, `CONDA_SH`, `CONDA_ENV`, `DRY_RUN`)
are tabulated in the README under [Step 5](../README.md#step-5-launch-training).

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
> `data/` (the parquet files) back into the lowres copy, which in turn links to the source.
> `meta/` starts as a symlink too; `patch_transcoded_meta.py` replaces it with a real copy
> before writing, so the source `info.json` is never modified. Only lowres `videos/` is
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
state.** The deploy interface
([TeleAvatar 2.0 developer docs](https://www.dexteleop.com/docs/teleavatar-2), §4.8 state
feedback) has no lift state topic, so on the robot that
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

Training and resuming share one command; only config fields differ. Repeat the original
run's overrides (`max_steps=`, `batch_size=`, ...) and add the fields below. Step directories
are zero-padded (`state/step_000600`).

| Goal | `resume` | `resume_reinit_lr` | `additional_steps` |
|---|---|---|---|
| Crash recovery (continue the LR curve) | directory `state/step_<N>` | `false` | omit, or exactly the remaining steps |
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

### Fields that fail quietly

**`additional_steps` moves the end of training, not the end of the LR curve.** The trainer
computes `max_steps = global_step + additional_steps` ("how many more steps", not an absolute
target). Without it, `max_steps` is recomputed from the config exactly as in the original run
(`trainer_state.json` does not store it), so an unchanged command resumes to the original end.

With `resume_reinit_lr: false` the scheduler comes back from the state with its original
length. Steps past the original `max_steps` therefore follow the cosine back **up**: on a
6-step test run the LR went 1e-6 → 7.6e-6 → 2.6e-5 for steps 7 and 8. To train longer, rebuild
the curve with `resume_reinit_lr=true` (see the anneal section below).

If there is nothing left to do (`max_steps` ≤ the current step) you get a **warning**, not an
error, and the run still saves a full state (~80 GB) and weights file into its new run
directory before exiting:

```
Resumed at global_step=<N> with max_steps=<M>; training will exit immediately.
Set additional_steps=N or a larger max_steps to continue.
```

**The scheduler type comes from the current command, the scheduler state from the
checkpoint.** `_build_scheduler` uses `SequentialLR` (warmup + cosine) when
`int(max_steps * 0.05) > 0` and a bare `CosineAnnealingLR` otherwise. If the resumed command
lands on the other side of that line from the original (in practice: short runs, or a
`max_steps=` override that was not repeated), the types differ. Original without warmup,
resume with: `KeyError: '_schedulers'`. The other way round loads without complaint and
the bare cosine runs with the new command's `T_max`.

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

## Pruning ZeRO state

Each `state/step_<N>/` is ~80 GB. `launch_train.sh` starts `scripts/prune_states.py --watch`
alongside training, keeping the newest `KEEP` (default 2). Run it manually if you started
training without the wrapper. The newest state counts toward `KEEP` while it is still being
written, so with `KEEP=1` the last complete state is gone before the next one is finished.

Weights (`weights/step_<N>.pt`, ~12 GB) are never pruned — only optimizer state is. Which
files must leave the training host together, and how to copy them, is a deployment question:
[What a deploy machine actually needs](#what-a-deploy-machine-actually-needs).

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

Robot-host dependencies (`openpi_client`, `websockets`, `msgpack`, `typing_extensions`) go in
that interpreter's user site:
`/usr/bin/python3 -m pip install --user -r experiments/teleavatar_v2_deploy/client/requirements.txt`
(`openpi-client` is installed from the openpi GitHub repo; it imports `typing_extensions`
without declaring it). GStreamer needs
the H.265 decoder plugin (`gst-inspect-1.0 nvh265dec`).

Pre-flight check: `bash experiments/teleavatar_v2_deploy/check_deployment.sh`. It verifies
the deploy files are present and imports what `run_client_ws.py` imports
(`openpi_client.websocket_client_policy`, not just `openpi_client`). It does not exercise the
link to the server;
`/usr/bin/python3 experiments/teleavatar_v2_deploy/test_websocket_connection.py --host <gpu-host>`
does.

## What a deploy machine actually needs

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

About 13.4 GB, not 33 — skipping those 32 GB is the point of the design.

**`dataset_stats.json` must come from the same run as the weights.** Two runs with matching
dimensions but different statistics load without complaint and produce subtly wrong actions.
Vector widths are validated; the values are not.

The server (`serve_policy_ws.py`, which loads the policy from `serve_policy.py`) resolves
everything relative to the repo root it is installed in (derived from its own file location,
not from the `PROJECT_ROOT` variable), and the intermediate directory names are load-bearing:

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

Any file-transfer tool works; keep the repo-relative paths (`runs/<task>/<RUN_ID>/...`,
`data/text_embeds_cache/<task>/`) so the deploy scripts find everything without extra
variables. See the README's [Step 6](../README.md#step-6-move-the-checkpoint-to-the-deploy-host)
for an `rsync -R` example.

Datasets travel the same way. Copy each `*_lowres` directory together with its `*_mono`
one: the mono copy symlinks `data/` back into it (`meta/` is a real copy once
`patch_transcoded_meta.py` has run).

## Startup order

The commands are in the README ([Step 7](../README.md#step-7-start-the-server-gpu-host) and
[Step 8](../README.md#step-8-start-the-client-robot-host)); this is the order they must run
in, and what each wrapper reads from the environment.

**1. GPU host — start the server.** `TASK=<your_task> bash start_local_serve_ws.sh 10`
(`10` = denoising steps). `TASK` must name the config the checkpoint was **trained** with:
hydra recomposes the data and model dimensions from it, so a mismatch fails at load. The
model loads first (~2 min); wait for `WebSocket server listening on ws://0.0.0.0:8000`.

| Variable | Default | Meaning |
|---|---|---|
| `TASK` | required | config name under `configs/task/` (no `.yaml`); must be the one the checkpoint was trained with |
| `RUN` | newest under `runs/$TASK` | run directory |
| `STEP` | highest step in that run | filename under `weights/` |
| `CHECKPOINT` | — | explicit weights path (overrides `RUN`/`STEP`) |
| `DATASET_STATS` | `$RUN/dataset_stats.json` | explicit stats path |
| `TASK_MAP` | `taskmap.json` | task library; skipped if absent (the server still runs, you just cannot switch instructions by name) |
| `PROMPT_TASK` | resolved from the dataset meta | startup instruction |
| `TEXT_EMBED` | — | precomputed T5 context `.pt` for the startup instruction; skips the lookup. Use it on a deploy machine that has the embeds but not the datasets |
| `CONFIG_OVERRIDES` | — | space-separated hydra overrides, each passed as `--config-override`, e.g. `"lift=legacy"` for a pre-split 17/15 checkpoint |
| `PROJECT_ROOT` | the script's directory | repo the wrapper `cd`s into; only needed when calling it from outside the repo. It does not relocate configs or caches — the server derives those from its own location |
| `PYTHON_BIN` | `python` | interpreter (activate the env first) |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | listen address |
| `CUDA_VISIBLE_DEVICES` | `0` | which GPU |
| `DIFFSYNTH_MODEL_BASE_PATH` | `$PROJECT_ROOT/checkpoints` | where the VAE is found |

The wrapper passes `--warmup-steps 8 10 12` itself, so CUDA Graphs for those step counts are
warm before the first request.

**2. Verify.**

```bash
curl http://<gpu-host>:8000/healthz      # expect OK
```

**3. Optional — build a task library** with `make_task_map.py` (command in the README under
[Optional: a task library](../README.md#optional-a-task-library-for-switching-instructions)).
Keys derive from the dataset directory name; a dataset with several instructions gets one key
per `task_index`. Override with `--instruction 'key=text'`. This file is generated per
deployment and gitignored.

**4. Robot host — ROS 2 up**, then zero the arms before handing control to the policy.

**5. Robot host — start the client.** `./run_task_ws.sh <taskmap-key> --dry-run` first, then
without `--dry-run`; omit the key to use the server's startup instruction. The script sources
ROS 2, starts a rosbag recording, and runs `run_client_ws.py` on the system interpreter.

| Variable | Default | Meaning |
|---|---|---|
| `SERVER_HOST` / `SERVER_PORT` | `127.0.0.1` / `8000` | the GPU host |
| `BAG_ROOT` | `$HOME/fastwam_bags` | where rosbag2 writes |
| `ROS_DOMAIN_ID` | `19` | must match the robot's bridge |
| `ROS_SETUP` | `/opt/ros/humble/setup.bash` | sourced before the client starts |
| `PYTHON_BIN` | `/usr/bin/python3` | the client interpreter — see [Two machines, two interpreters](#two-machines-two-interpreters) |
| `CLIENT_DIR` | the script's directory | directory holding `run_client_ws.py` |

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
  keep-alive timer. The developer docs (§4.6, lift servo control) recommend ≥ 10 Hz while
  moving; after more than 3 s without a command the servo decelerates and then disables
  itself, and it does not use the FSM heartbeat. On exit the client sends `[0.0, 0.0]` (stop
  and disable) before stopping the arms. On
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

All tools live under `experiments/teleavatar_v2_deploy/server/`; run any of them with
`--help` (the README's [Analysis tools](../README.md#analysis-tools) table says what each one
answers). The bag tools take `--bag <bag dir>` and write next to the bag unless given `--out`.
The ones that run the model need `--task` plus a matching `--checkpoint` / `--dataset-stats`
pair from the same run, and on a machine without the datasets also `--text-embed` (as for the
server; otherwise the instruction is read from `meta/tasks.jsonl`); the rest only read a bag
or a stream.

```bash
python experiments/teleavatar_v2_deploy/server/analyze_deploy_bag.py --bag <bag dir> --out <out dir>
python experiments/teleavatar_v2_deploy/server/joint_error_report.py --bag <bag dir> --out <out dir>

# re-run the policy over a recorded bag (needs a checkpoint)
python experiments/teleavatar_v2_deploy/server/bag_visualize_and_predict.py \
  --task <your_task> \
  --checkpoint runs/<task>/<RUN_ID>/checkpoints/weights/step_<N>.pt \
  --dataset-stats runs/<task>/<RUN_ID>/dataset_stats.json \
  --text-embed data/text_embeds_cache/<cache dir>/<sha256>.t5_len128.wan22ti2v5b.pt \
  --bag <bag dir> --out <out dir>
```

`offline_infer_from_export.py --export` takes the `export/` directory that
`bag_visualize_and_predict.py` writes under its `--out` (or `client/export_bag_for_offline.py`):

```bash
# replay the export/ directory written by the command above
python experiments/teleavatar_v2_deploy/server/offline_infer_from_export.py \
  --export <out dir>/export --out <out dir 2> --task <your_task> \
  --checkpoint ... --dataset-stats ... --text-embed ... --start 1000 --max-samples 2
```

`offline_infer_from_export.py` replays exported frames without a robot;
`viz_trainset_pred_compare.py` checks predictions against ground truth on training samples.

## Latency

```bash
bash bench_latency_ws.sh                      # single point, server's default instruction
bash bench_latency_ws.sh <taskmap-key> 12     # specific task and step count
bash bench_latency_ws.sh --sweep              # every served task x 8/10/12 steps
```

It connects to `HOST`/`PORT` (default `127.0.0.1`/`8000`) with `PYTHON_BIN` (default `python`
on `PATH`). The sweep reads `available_tasks` from the server's connection metadata, so it
needs no task list of its own. Denoising steps are per-request — no restart needed;
`start_local_serve_ws.sh` already warms the CUDA Graphs for 8, 10 and 12 steps, so every
value in the sweep is compared warm (pass `--warmup-steps` yourself only when calling
`serve_policy_ws.py` directly).

`bench_infer_ws.py` uses the same msgpack wire format as the real client (raw `websockets`
plus `openpi_client.msgpack_numpy`, falling back to a vendored copy), with synthetic frames and
no ROS 2 — nothing moves. It does not import `openpi_client.websocket_client_policy`, so a
passing bench does not prove the client stack is installed; `test_websocket_connection.py`
does.

---

# Troubleshooting

| Symptom | Check | Cause / fix |
|---|---|---|
| Canvas size in the startup log is unexpected | `TA2 mosaic geometry from config: canvas=...` in the server log | task config points at the wrong data config — stop before moving anything |
| `Cannot reach policy server` | the URL in the error; `curl http://<ip>:8000/healthz` | server down or still loading (~2 min); firewall; the client's `SERVER_HOST` left at its default `127.0.0.1` while the server runs on another machine; or the server started with `HOST=127.0.0.1` |
| `ModuleNotFoundError: websockets` | which interpreter | client must run on system `python3`, not conda |
| `ROS2 interface not initialized` | `ros2 topic list` | ROS bridge not running, or `ROS_DOMAIN_ID` mismatch |
| `Timeout waiting for initial sensors` / `Failed to receive initial sensor data` | `ros2 topic hz /left_arm/joint_states`, `gst-inspect-1.0 nvh265dec` | RTP not arriving on 8890, or GStreamer plugin missing |
| Action dimension mismatch at load | — | checkpoint, `dataset_stats.json` and task config must all come from the same run |
| Server cannot find the VAE | `ls checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/` | the VAE is not inside the `.pt`; copy it separately |
| `Unknown task=...` on the first inference | server log / client error | the key is not in `taskmap.json` — regenerate it with `make_task_map.py` |
| `MissingCUDAException: CUDA_HOME does not exist` at `accelerate launch` | `which nvcc` | no CUDA toolkit on the host — `conda install -y -c nvidia cuda-nvcc=12.8` into the env |
| `RepositoryNotFoundError: 401` while downloading weights | `DIFFSYNTH_DOWNLOAD_SOURCE` | Hugging Face source needs `redirect_common_files: false` — see [Base weights and download sources](#base-weights-and-download-sources) |
| `Cannot detect model type ... File: []` | `ls checkpoints/` against the README's tree | a base-weight file is missing — e.g. `DIFFSYNTH_SKIP_DOWNLOAD=true` forbade fetching it |
| Training exits right after resuming | trainer warning | `max_steps` ≤ the resumed step: add `additional_steps` (with `resume_reinit_lr=true` to go past the original end) |
| `KeyError: '_schedulers'` when resuming | traceback in `accelerator.load_state` | the resumed command's step count differs from the original run's; repeat its overrides |
| LR rises after resuming | `lr=` in the train log | resumed past the original `max_steps` with `resume_reinit_lr=false` |
| Resumed run seems to repeat data | `trainer_state.json` | `batch_size` differs from the original run |
| Text embedding not found | `ls data/text_embeds_cache/<task>/` | precompute step skipped, or instruction text changed |
