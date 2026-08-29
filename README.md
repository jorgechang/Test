# ANYmal v13.45 — simple per-run folders and three checkpoints

This bundle keeps the v13.43 task and model conditions unchanged while simplifying experiment storage.

It retains:

- the exact v13.20 dense camera-pose reward;
- the visible green-arrow task;
- balanced, reset-order-independent task randomization;
- the 11-action arcs-plus-strafes action set;
- 128×128 RGB, 90° horizontal field of view, and `train_ratio=256`;
- an RGB-only baseline and an optional RGB + 33-D proprioception condition.

The run-management change is intentionally small: every launch gets its own folder, only code-relevant parameters are saved, and only three checkpoint files are retained.

## Installation

Extract the folder anywhere inside your IsaacLab checkout:

```bash
cd ~/Documents/Serre/VPT/IsaacLab-beta
unzip r2dreamer_isaaclab_anymal_v13_45_simple_runs_three_checkpoints_proprio_complete.zip
cd r2dreamer_isaaclab_anymal_v13_45_simple_runs_three_checkpoints_proprio_complete
```

Run the offline checks:

```bash
python verify_bundle.py
```

Then run the Isaac Sim proprioception smoke test:

```bash
NUM_ENVS=1 STEPS=2000 ./run_anymal_proprio_smoke_test.sh
```

## Separate folder for every run

The launchers create a unique timestamped directory:

```text
r2dreamer/logdir/isaaclab_anymal_v13_45/
├── rgb_only/
│   ├── latest_run -> runs/<most-recent-run>
│   └── runs/
│       └── 20260829T040000Z_headless_rgb_seed0/
└── rgb_proprio/
    ├── latest_run -> runs/<most-recent-run>
    └── runs/
        └── 20260829T041000Z_headless_proprio_seed0/
```

Two launches never append to the same folder. Even simultaneous launches with the same timestamp and tag receive distinct suffixes.

Use a descriptive tag:

```bash
SEED=0 \
TASK_SEED=37 \
RUN_TAG=rgb_seed0_task37 \
NUM_ENVS=8 \
STEPS=8000000 \
./run_anymal_headless.sh
```

The matched proprioception run is:

```bash
SEED=0 \
TASK_SEED=37 \
RUN_TAG=proprio_seed0_task37 \
NUM_ENVS=8 \
STEPS=8000000 \
./run_anymal_proprio_headless.sh
```

List available runs:

```bash
./run_list_runs.sh
```

## Contents of one run

```text
run/
├── config/
│   ├── parameters.yaml
│   ├── parameters.json
│   ├── resolved_config.yaml
│   └── hydra_overrides.txt
├── plots/
├── gifs/
├── evaluations/
├── metrics.jsonl
├── train.log
├── run_status.json
├── checkpoint_summary.json
├── latest.pt
├── best_success.pt
└── best_reward.pt
```

Only parameters used by the launch and training code are recorded:

- `parameters.yaml` and `parameters.json`: launcher variables, run mode, and the exact command;
- `resolved_config.yaml`: the final resolved Hydra configuration seen by r2dreamer;
- `hydra_overrides.txt`: the exact Hydra overrides.

The JSON copy is retained only so the evaluation launcher can reliably recover architecture-sensitive settings from a run folder. No package list, machine report, Git snapshot, or source-tree copy is created.

## The three checkpoints

### `latest.pt`

The newest checkpoint. It is refreshed every:

```text
CHECKPOINT_EVERY=10000
```

and once at normal training completion.

### `best_success.pt`

The checkpoint with the highest rolling success rate over the most recent completed episodes at a checkpoint save. Defaults:

```text
BEST_WINDOW_EPISODES=100
BEST_MIN_EPISODES=100
```

Selection key:

```text
1. highest success rate
2. highest mean episode return as an exact-tie breaker
```

A successful episode is one in which the camera reaches the precise `0.12 m / 5°` tolerance for the required hold duration.

### `best_reward.pt`

The checkpoint with the highest rolling mean episode return over the same window.

Selection key:

```text
1. highest mean episode return
2. highest success rate as an exact-tie breaker
```

This checkpoint is useful because the dense task reward may improve before the precise success rate does. Keeping both files lets you compare “best actual success” against “best reward optimization” without retaining many checkpoint files.

The best criteria are reconsidered whenever `latest.pt` is saved. Therefore, decreasing `CHECKPOINT_EVERY` gives finer checkpoint selection at the cost of more frequent disk writes.

For a smoke test that ends before `BEST_MIN_EPISODES`, both best files are copied from the final latest checkpoint and marked as fallbacks in `checkpoint_summary.json`.

## Evaluation

Evaluation defaults to `best_success.pt`:

```bash
RUN_DIR=/absolute/path/to/run \
TASK_SEED=9001 \
EPISODES=100 \
./run_anymal_eval.sh
```

Select another retained checkpoint with:

```bash
CHECKPOINT_KIND=best_success ./run_anymal_eval.sh
CHECKPOINT_KIND=best_reward  ./run_anymal_eval.sh
CHECKPOINT_KIND=latest       ./run_anymal_eval.sh
```

For a proprioception run, either use the generic evaluator with `RUN_DIR` or the dedicated wrapper:

```bash
RUN_DIR=/absolute/path/to/rgb_proprio/run \
CHECKPOINT_KIND=best_success \
TASK_SEED=9001 \
./run_anymal_proprio_eval.sh
```

The evaluator reloads image size, horizon, RGB/proprioception routing, decoder settings, model seed, balanced-randomization setting, and arrow anchor from `config/parameters.json`. An explicitly supplied environment variable overrides the saved value. `TASK_SEED` is deliberately not inherited, so validation or test tasks can use a held-out seed.

Evaluation outputs are placed inside the selected run's `evaluations/` directory unless `OUTPUT_DIR` is supplied.

## RGB-only and proprioception conditions

The original launchers remain RGB-only:

```bash
./run_anymal_headless.sh
./run_anymal_gui.sh
```

The proprioception launchers add a 33-D robot-only vector:

```text
body-frame linear velocity       3
body-frame angular velocity      3
projected gravity                3
joint position relative to default 12
joint velocity                  12
                               ----
total                            33
```

```bash
./run_anymal_proprio_headless.sh
./run_anymal_proprio_gui.sh
```

The proprioception vector contains no global XY, global yaw, target pose, distance, bearing, or `goal_vec`.

The optional 4-D goal-vector decoder remains disabled by default. To use it as decoder-only supervision:

```bash
GOALVEC_DECODER_ENABLED=1 \
GOALVEC_LOSS_SCALE=100 \
./run_anymal_proprio_headless.sh
```

It never enters the encoder or policy.

## Randomness controls

```bash
# Reproduce model initialization and task sequence
SEED=0 TASK_SEED=0 ./run_anymal_headless.sh

# Same model initialization, different task sequence
SEED=0 TASK_SEED=101 ./run_anymal_headless.sh

# Generate a new task seed and record it in the run parameters
TASK_SEED=auto ./run_anymal_headless.sh

# Disable balanced strata while retaining independent deterministic streams
BALANCED_RANDOMIZATION=0 TASK_SEED=0 ./run_anymal_headless.sh
```

Task samples are keyed by task seed, environment ID, episode index, and component. Asynchronous reset order therefore does not alter another vector environment's future task sequence.

## Plots

Live plots are enabled by default and saved in each run's `plots/` directory. Disable them with:

```bash
AUTO_PLOTS=0 ./run_anymal_headless.sh
```

Regenerate plots for an existing run:

```bash
RUN_DIR=/absolute/path/to/run ./run_anymal_plots.sh
```

The plot summary also reports the step, rolling success rate, and rolling return associated with the three retained checkpoints.
