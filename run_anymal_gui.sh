#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
NUM_ENVS="${NUM_ENVS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
STEPS="${STEPS:-1000000}"
if [[ "$PROPRIOCEPTION_ENABLED" == "1" ]]; then VARIANT="rgb_proprio"; else VARIANT="rgb_only"; fi
prepare_training_run "$VARIANT" gui
unset R2DREAMER_INIT_CHECKPOINT || true
export R2DREAMER_SAVE_GIFS="${SAVE_GIFS:-1}" R2DREAMER_GIF_FPS="${GIF_FPS:-8}" R2DREAMER_GIF_SCALE="${GIF_SCALE:-2}" R2DREAMER_GIF_MAX_BATCH="${GIF_MAX_BATCH:-1}" R2DREAMER_GIF_NAMES="${GIF_NAMES:-train_video,open_loop}"
cmd=(
  "$ISAACLAB_SH" -p "$BUNDLE_DIR/r2dreamer_train.py" --num_envs "$NUM_ENVS" --visualizer kit --enable_cameras
  env=isaaclab_anymal_room "env.size=[$IMAGE_SIZE,$IMAGE_SIZE]" "env.steps=$STEPS"
  "${randomization_overrides[@]}" "${arrow_overrides[@]}" "${encoder_overrides[@]}" env.show_target_debug=false
  model=size12M model.rep_loss=dreamer "model.horizon=$MODEL_HORIZON" "${decoder_overrides[@]}"
  "${checkpoint_overrides[@]}"
  "batch_size=$BATCH_SIZE" "batch_length=$BATCH_LENGTH" buffer.storage_device=cpu buffer.max_size=100000
  trainer.video_pred_log=true "logdir=$LOGDIR" "$@"
)
initialize_training_run "$VARIANT" gui "run_anymal_gui.sh" "${cmd[@]}"
echo "[V13.45] run=$RUN_ID | exact v13.20 reward | balanced randomness=$BALANCED_RANDOMIZATION | task_seed=$TASK_SEED | encoder=$ENCODER_DESC | decoder=$DECODER_DESC | envs=$NUM_ENVS | steps=$STEPS"
echo "[V13.45] logdir=$LOGDIR"
echo "[V13.45] checkpoints: latest every ${CHECKPOINT_EVERY} steps | best_success + best_reward over ${BEST_WINDOW_EPISODES} episodes (minimum=${BEST_MIN_EPISODES})"
execute_training_run "${cmd[@]}"
