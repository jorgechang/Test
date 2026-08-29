#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_isaaclab_root() {
  local current="$BUNDLE_DIR"
  while [[ "$current" != "/" ]]; do
    if [[ -f "$current/isaaclab.sh" ]]; then printf '%s\n' "$current"; return 0; fi
    current="$(dirname "$current")"
  done
  echo "[v13.45][ERROR] Extract this folder anywhere inside IsaacLab-beta." >&2
  return 1
}

_abs_path() {
  python - "$1" <<'PY'
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}

ISAACLAB_ROOT="$(find_isaaclab_root)"
R2DREAMER_REPO="$ISAACLAB_ROOT/r2dreamer"
ISAACLAB_SH="$ISAACLAB_ROOT/isaaclab.sh"
LOG_ROOT="${LOG_ROOT:-$R2DREAMER_REPO/logdir/isaaclab_anymal_v13_45}"

# Evaluation can recover architecture/task settings from the selected run.
# Explicit environment variables always win. TASK_SEED is never inherited,
# allowing evaluation to use a separate held-out task seed.
_preload_saved_run_hyperparameters() {
  [[ "${R2DREAMER_LOAD_RUN_CONFIG:-0}" == "1" ]] || return 0
  local candidate="" requested="" variant="rgb_only" config=""
  requested="${RUN_DIR:-${LOGDIR:-}}"
  if [[ -n "$requested" ]]; then
    candidate="$(_abs_path "$requested")"
  elif [[ -n "${CHECKPOINT:-}" ]]; then
    candidate="$(python - "$CHECKPOINT" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]).expanduser().resolve()
print(p.parent.parent if p.parent.name == "checkpoints" else p.parent)
PY
)"
  else
    [[ "${PROPRIOCEPTION_ENABLED:-0}" == "1" ]] && variant="rgb_proprio"
    if [[ -d "$LOG_ROOT/$variant/latest_run" || -L "$LOG_ROOT/$variant/latest_run" ]]; then
      candidate="$(python - "$LOG_ROOT/$variant/latest_run" <<'PY'
import os, sys
print(os.path.realpath(sys.argv[1]))
PY
)"
    fi
  fi
  [[ -n "$candidate" ]] || return 0
  config="$candidate/config/parameters.json"
  [[ -f "$config" ]] || return 0

  while IFS=$'\t' read -r key value; do
    [[ -n "$key" ]] || continue
    if [[ ! -v "$key" ]]; then
      printf -v "$key" '%s' "$value"
      export "$key"
    fi
  done < <(python - "$config" <<'PY'
import json, sys
from pathlib import Path
saved = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
hyper = saved.get("hyperparameters", {})
for key in (
    "MODEL_HORIZON", "IMAGE_SIZE", "SEED", "BALANCED_RANDOMIZATION",
    "PROPRIOCEPTION_ENABLED", "PROPRIO_LOSS_SCALE",
    "GOALVEC_DECODER_ENABLED", "GOALVEC_LOSS_SCALE", "ARROW_TARGET_ANCHOR",
):
    value = hyper.get(key)
    if value is not None and str(value) != "":
        print(f"{key}\t{str(value).replace(chr(9), ' ').replace(chr(10), ' ')}")
PY
)
  export R2DREAMER_LOADED_RUN_CONFIG="$candidate"
  echo "[V13.45] Loaded evaluation parameters from $config"
}

_preload_saved_run_hyperparameters

MODEL_HORIZON="${MODEL_HORIZON:-333}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
SEED="${SEED:-0}"
TASK_SEED="${TASK_SEED:-$SEED}"
BALANCED_RANDOMIZATION="${BALANCED_RANDOMIZATION:-1}"
PROPRIOCEPTION_ENABLED="${PROPRIOCEPTION_ENABLED:-0}"
PROPRIO_LOSS_SCALE="${PROPRIO_LOSS_SCALE:-1.0}"
GOALVEC_LOSS_SCALE="${GOALVEC_LOSS_SCALE:-100}"
GOALVEC_DECODER_ENABLED="${GOALVEC_DECODER_ENABLED:-0}"
ARROW_TARGET_ANCHOR="${ARROW_TARGET_ANCHOR:-base}"
ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}"

# Only three checkpoints are retained: latest, best success, and best reward.
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
BEST_WINDOW_EPISODES="${BEST_WINDOW_EPISODES:-100}"
BEST_MIN_EPISODES="${BEST_MIN_EPISODES:-100}"

if [[ "$TASK_SEED" == "auto" ]]; then
  TASK_SEED="$(python - <<'PY'
import secrets
print(secrets.randbelow(2**31 - 1))
PY
)"
  echo "[V13.45] Generated TASK_SEED=$TASK_SEED"
fi

_is_nonnegative_number() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; }
_is_positive_integer() { [[ "$1" =~ ^[0-9]+$ ]] && (( 10#$1 > 0 )); }
_is_nonnegative_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }
_is_flag() { [[ "$1" == "0" || "$1" == "1" ]]; }

if ! _is_positive_integer "$MODEL_HORIZON" || (( MODEL_HORIZON <= 1 )); then echo "MODEL_HORIZON must be an integer > 1" >&2; exit 2; fi
if ! _is_positive_integer "$IMAGE_SIZE" || (( IMAGE_SIZE < 32 )) || (( IMAGE_SIZE % 16 != 0 )); then echo "IMAGE_SIZE must be >=32 and divisible by 16" >&2; exit 2; fi
if ! [[ "$SEED" =~ ^-?[0-9]+$ ]]; then echo "SEED must be an integer" >&2; exit 2; fi
if ! [[ "$TASK_SEED" =~ ^-?[0-9]+$ ]]; then echo "TASK_SEED must be an integer or auto" >&2; exit 2; fi
if ! _is_nonnegative_number "$PROPRIO_LOSS_SCALE"; then echo "PROPRIO_LOSS_SCALE must be non-negative" >&2; exit 2; fi
if ! _is_nonnegative_number "$GOALVEC_LOSS_SCALE"; then echo "GOALVEC_LOSS_SCALE must be non-negative" >&2; exit 2; fi
for flag_name in BALANCED_RANDOMIZATION PROPRIOCEPTION_ENABLED GOALVEC_DECODER_ENABLED ALLOW_EXISTING_RUN; do
  flag_value="${!flag_name}"
  if ! _is_flag "$flag_value"; then echo "$flag_name must be 0 or 1" >&2; exit 2; fi
done
if [[ "$ARROW_TARGET_ANCHOR" != "tip" && "$ARROW_TARGET_ANCHOR" != "base" ]]; then echo "ARROW_TARGET_ANCHOR must be tip or base" >&2; exit 2; fi
for integer_name in CHECKPOINT_EVERY BEST_WINDOW_EPISODES BEST_MIN_EPISODES; do
  integer_value="${!integer_name}"
  if ! _is_nonnegative_integer "$integer_value"; then echo "$integer_name must be a non-negative integer" >&2; exit 2; fi
done
if (( BEST_WINDOW_EPISODES <= 0 || BEST_MIN_EPISODES <= 0 )); then echo "BEST_WINDOW_EPISODES and BEST_MIN_EPISODES must be > 0" >&2; exit 2; fi
if (( BEST_MIN_EPISODES > BEST_WINDOW_EPISODES )); then echo "BEST_MIN_EPISODES cannot exceed BEST_WINDOW_EPISODES" >&2; exit 2; fi

if [[ ! -f "$R2DREAMER_REPO/train.py" ]]; then
  echo "[v13.45][ERROR] Expected NM512/r2dreamer at: $R2DREAMER_REPO" >&2
  exit 1
fi
if [[ "${GOALVEC_SKIP_INSTALL:-0}" != "1" ]]; then python "$BUNDLE_DIR/install.py" --repo "$R2DREAMER_REPO"; fi

arrow_overrides=(env.show_goal_arrow=true env.arrow_target_anchor="$ARROW_TARGET_ANCHOR")
if [[ "$BALANCED_RANDOMIZATION" == "1" ]]; then BALANCED_RANDOMIZATION_HYDRA=true; else BALANCED_RANDOMIZATION_HYDRA=false; fi
randomization_overrides=(seed="$SEED" env.task_seed="$TASK_SEED" env.balanced_randomization="$BALANCED_RANDOMIZATION_HYDRA")
checkpoint_overrides=(
  ++trainer.checkpoint_every="$CHECKPOINT_EVERY"
  ++trainer.best_checkpoint_window="$BEST_WINDOW_EPISODES"
  ++trainer.best_checkpoint_min_episodes="$BEST_MIN_EPISODES"
)

LIVE_PLOT_PID=""
start_live_plotter() {
  local logdir="$1"
  [[ "${AUTO_PLOTS:-1}" == "1" ]] || return 0
  mkdir -p "$logdir/plots"
  python "$BUNDLE_DIR/plot_training.py" --logdir "$logdir" --window "${PLOT_WINDOW:-50}" \
    --watch-episodes --every-episodes "${PLOT_EVERY_EPISODES:-1}" \
    --poll-interval "${PLOT_POLL_INTERVAL:-2.0}" \
    >> "$logdir/plots/live_plotter.log" 2>&1 &
  LIVE_PLOT_PID=$!
}
stop_live_plotter() {
  if [[ -n "${LIVE_PLOT_PID:-}" ]]; then
    kill "$LIVE_PLOT_PID" 2>/dev/null || true
    wait "$LIVE_PLOT_PID" 2>/dev/null || true
    LIVE_PLOT_PID=""
  fi
}

encoder_overrides=()
if [[ "$PROPRIOCEPTION_ENABLED" == "1" ]]; then
  encoder_overrides+=(env.proprioception_enabled=true env.encoder.mlp_keys=proprio)
  ENCODER_DESC="RGB+33D proprio"
else
  ENCODER_DESC="RGB only"
fi

decoder_overrides=()
if [[ "$PROPRIOCEPTION_ENABLED" == "1" && "$GOALVEC_DECODER_ENABLED" == "1" ]]; then
  decoder_overrides+=('env.decoder.mlp_keys=proprio|goal_vec' ++model.loss_scales.proprio="$PROPRIO_LOSS_SCALE" ++model.loss_scales.goal_vec="$GOALVEC_LOSS_SCALE")
  DECODER_DESC="image+proprio+goal_vec"
elif [[ "$PROPRIOCEPTION_ENABLED" == "1" ]]; then
  decoder_overrides+=(env.decoder.mlp_keys=proprio ++model.loss_scales.proprio="$PROPRIO_LOSS_SCALE")
  DECODER_DESC="image+proprio"
elif [[ "$GOALVEC_DECODER_ENABLED" == "1" ]]; then
  decoder_overrides+=(env.decoder.mlp_keys=goal_vec ++model.loss_scales.goal_vec="$GOALVEC_LOSS_SCALE")
  DECODER_DESC="image+goal_vec"
else
  DECODER_DESC="image only"
fi

_sanitize_run_component() { printf '%s' "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//'; }

prepare_training_run() {
  local variant="$1" mode="$2" raw_tag="${RUN_TAG:-seed${SEED}_task${TASK_SEED}}"
  local run_tag stamp base candidate suffix variant_root runs_root requested
  run_tag="$(_sanitize_run_component "$raw_tag")"; [[ -n "$run_tag" ]] || run_tag="run"
  variant_root="$LOG_ROOT/$variant"; runs_root="$variant_root/runs"; requested="${LOGDIR:-${RUN_DIR:-}}"
  mkdir -p "$runs_root"
  if [[ -n "$requested" ]]; then
    candidate="$(_abs_path "$requested")"
    if [[ -d "$candidate" && -n "$(find "$candidate" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" && "$ALLOW_EXISTING_RUN" != "1" ]]; then
      echo "[V13.45][ERROR] Refusing to append to non-empty run directory: $candidate" >&2
      exit 2
    fi
    mkdir -p "$candidate"
  else
    stamp="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
    base="$(_sanitize_run_component "${RUN_ID:-${stamp}_${mode}_${run_tag}}")"
    candidate="$runs_root/$base"; suffix=2
    while ! mkdir "$candidate" 2>/dev/null; do
      candidate="$runs_root/${base}_$suffix"; suffix=$((suffix + 1))
    done
  fi
  mkdir -p "$candidate/config" "$candidate/plots" "$candidate/gifs" "$candidate/evaluations"
  LOGDIR="$candidate"; RUN_TAG="$run_tag"; RUN_ID="$(basename "$candidate")"; RUN_MODE="$mode"; VARIANT_ROOT="$variant_root"
  export LOGDIR RUN_DIR="$LOGDIR" RUN_TAG RUN_ID RUN_MODE VARIANT_ROOT LOG_ROOT R2DREAMER_RUN_DIR="$LOGDIR"
}

initialize_training_run() {
  local variant="$1" mode="$2" launcher="$3"; shift 3
  python "$BUNDLE_DIR/run_manager.py" init \
    --logdir "$LOGDIR" --mode "$mode" --variant "$variant" --run-id "$RUN_ID" --run-tag "$RUN_TAG" \
    --launcher "$launcher" --latest-link "$VARIANT_ROOT/latest_run" \
    --param "SEED=$SEED" --param "TASK_SEED=$TASK_SEED" \
    --param "BALANCED_RANDOMIZATION=$BALANCED_RANDOMIZATION" \
    --param "PROPRIOCEPTION_ENABLED=$PROPRIOCEPTION_ENABLED" \
    --param "PROPRIO_LOSS_SCALE=$PROPRIO_LOSS_SCALE" \
    --param "GOALVEC_DECODER_ENABLED=$GOALVEC_DECODER_ENABLED" \
    --param "GOALVEC_LOSS_SCALE=$GOALVEC_LOSS_SCALE" \
    --param "ARROW_TARGET_ANCHOR=$ARROW_TARGET_ANCHOR" \
    --param "MODEL_HORIZON=$MODEL_HORIZON" --param "IMAGE_SIZE=$IMAGE_SIZE" \
    --param "NUM_ENVS=${NUM_ENVS:-}" --param "BATCH_SIZE=${BATCH_SIZE:-}" \
    --param "BATCH_LENGTH=${BATCH_LENGTH:-}" --param "STEPS=${STEPS:-}" \
    --param "CHECKPOINT_EVERY=$CHECKPOINT_EVERY" \
    --param "BEST_WINDOW_EPISODES=$BEST_WINDOW_EPISODES" \
    --param "BEST_MIN_EPISODES=$BEST_MIN_EPISODES" \
    --param "AUTO_PLOTS=${AUTO_PLOTS:-1}" --param "PLOT_WINDOW=${PLOT_WINDOW:-50}" \
    --param "PLOT_EVERY_EPISODES=${PLOT_EVERY_EPISODES:-1}" \
    --param "PLOT_POLL_INTERVAL=${PLOT_POLL_INTERVAL:-2.0}" \
    --param "SAVE_GIFS=${SAVE_GIFS:-}" --param "GIF_FPS=${GIF_FPS:-8}" \
    --param "GIF_SCALE=${GIF_SCALE:-2}" --param "GIF_MAX_BATCH=${GIF_MAX_BATCH:-1}" \
    --param "GIF_NAMES=${GIF_NAMES:-}" -- "$@"
}

execute_training_run() {
  local status
  start_live_plotter "$LOGDIR"; trap 'stop_live_plotter' EXIT
  set +e
  (cd "$ISAACLAB_ROOT" && "$@") 2>&1 | tee -a "$LOGDIR/train.log"
  status=${PIPESTATUS[0]}
  set -e
  stop_live_plotter
  if [[ -f "$LOGDIR/metrics.jsonl" ]]; then
    python "$BUNDLE_DIR/plot_training.py" --logdir "$LOGDIR" --window "${PLOT_WINDOW:-50}" >> "$LOGDIR/plots/final_plot.log" 2>&1 || true
  fi
  python "$BUNDLE_DIR/run_manager.py" finish --logdir "$LOGDIR" --exit-code "$status" || true
  trap - EXIT
  return "$status"
}

latest_run_dir() {
  local variant="$1" link="$LOG_ROOT/$variant/latest_run"
  [[ -d "$link" || -L "$link" ]] || return 1
  python - "$link" <<'PY'
import os, sys
print(os.path.realpath(sys.argv[1]))
PY
}

resolve_training_run_dir() {
  local variant="$1" requested="${RUN_DIR:-${LOGDIR:-}}" candidate
  if [[ -n "$requested" ]]; then candidate="$(_abs_path "$requested")"; else candidate="$(latest_run_dir "$variant" 2>/dev/null || true)"; fi
  if [[ -z "$candidate" || ! -d "$candidate" ]]; then echo "[V13.45][ERROR] No run directory found. Set RUN_DIR=/path/to/run." >&2; return 1; fi
  printf '%s\n' "$candidate"
}

resolve_checkpoint_path() {
  local variant="$1" run_dir kind candidate
  if [[ -n "${CHECKPOINT:-}" ]]; then
    candidate="$(_abs_path "$CHECKPOINT")"; [[ -f "$candidate" ]] || { echo "[V13.45][ERROR] Checkpoint not found: $candidate" >&2; return 1; }
    printf '%s\n' "$candidate"; return 0
  fi
  run_dir="$(resolve_training_run_dir "$variant")"; kind="${CHECKPOINT_KIND:-best_success}"
  case "$kind" in
    best|best_success) candidate="$run_dir/best_success.pt" ;;
    best_reward|best_return|reward) candidate="$run_dir/best_reward.pt" ;;
    latest) candidate="$run_dir/latest.pt" ;;
    *) echo "[V13.45][ERROR] CHECKPOINT_KIND must be best_success, best_reward, or latest" >&2; return 2 ;;
  esac
  if [[ ! -f "$candidate" && "$kind" != "latest" ]]; then
    echo "[V13.45][WARNING] $(basename "$candidate") is unavailable; using latest.pt" >&2
    candidate="$run_dir/latest.pt"
  fi
  [[ -f "$candidate" ]] || { echo "[V13.45][ERROR] No usable checkpoint in $run_dir" >&2; return 1; }
  printf '%s\n' "$candidate"
}

run_dir_from_checkpoint() {
  python - "$1" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]).expanduser().resolve()
print(p.parent.parent if p.parent.name == "checkpoints" else p.parent)
PY
}

default_eval_output_dir() {
  local checkpoint="$1" label="${2:-}" run_dir stamp checkpoint_name base candidate suffix
  run_dir="$(run_dir_from_checkpoint "$checkpoint")"; stamp="$(date -u +%Y%m%dT%H%M%SZ)"; checkpoint_name="$(basename "${checkpoint%.pt}")"
  if [[ -n "$label" ]]; then label="$(_sanitize_run_component "$label")"; [[ -n "$label" ]] && checkpoint_name="${checkpoint_name}_${label}"; fi
  mkdir -p "$run_dir/evaluations"; base="$run_dir/evaluations/${checkpoint_name}_task${TASK_SEED}_${stamp}"; candidate="$base"; suffix=2
  while ! mkdir "$candidate" 2>/dev/null; do candidate="${base}_$suffix"; suffix=$((suffix + 1)); done
  printf '%s\n' "$candidate"
}
