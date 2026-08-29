#!/usr/bin/env bash
set -euo pipefail
export GOALVEC_SKIP_INSTALL="${GOALVEC_SKIP_INSTALL:-1}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if [[ "$PROPRIOCEPTION_ENABLED" == "1" ]]; then VARIANT="rgb_proprio"; else VARIANT="rgb_only"; fi
LOGDIR="$(resolve_training_run_dir "$VARIANT")"
exec python "$BUNDLE_DIR/plot_training.py" --logdir "$LOGDIR" --window "${PLOT_WINDOW:-50}" --watch-episodes --every-episodes "${PLOT_EVERY_EPISODES:-1}" --poll-interval "${PLOT_POLL_INTERVAL:-2.0}"
