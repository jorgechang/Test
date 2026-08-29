#!/usr/bin/env bash
set -euo pipefail
export R2DREAMER_LOAD_RUN_CONFIG=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
NUM_ENVS="${NUM_ENVS:-1}"
EPISODES="${EPISODES:-100}"
VIDEO_STEPS="${VIDEO_STEPS:-64}"
if [[ "$PROPRIOCEPTION_ENABLED" == "1" ]]; then VARIANT="rgb_proprio"; else VARIANT="rgb_only"; fi
CHECKPOINT_PATH="$(resolve_checkpoint_path "$VARIANT")"
if [[ -z "${OUTPUT_DIR:-}" ]]; then OUTPUT_DIR="$(default_eval_output_dir "$CHECKPOINT_PATH" gui)"; fi
export R2DREAMER_SAVE_GIFS="${SAVE_GIFS:-1}" R2DREAMER_GIF_FPS="${GIF_FPS:-8}" R2DREAMER_GIF_SCALE="${GIF_SCALE:-2}" R2DREAMER_GIF_MAX_BATCH=1 R2DREAMER_GIF_NAMES="eval_video,eval_open_loop"
extra=(--output_dir "$OUTPUT_DIR")
[[ "${SAVE_GIFS:-1}" == "0" ]] && extra+=(--no_video)
echo "[V13.45 EVAL] checkpoint=$CHECKPOINT_PATH"
echo "[V13.45 EVAL] output=$OUTPUT_DIR"
cd "$ISAACLAB_ROOT"
exec "$ISAACLAB_SH" -p "$BUNDLE_DIR/r2dreamer_eval.py" --checkpoint "$CHECKPOINT_PATH" --episodes "$EPISODES" --num_envs "$NUM_ENVS" --video_steps "$VIDEO_STEPS" --visualizer kit --enable_cameras "${extra[@]}" \
  env=isaaclab_anymal_room env.size="[$IMAGE_SIZE,$IMAGE_SIZE]" "${randomization_overrides[@]}" "${arrow_overrides[@]}" "${encoder_overrides[@]}" env.show_target_debug=false \
  model=size12M model.rep_loss=dreamer model.horizon="$MODEL_HORIZON" "${decoder_overrides[@]}" "$@"
