#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NUM_ENVS="${NUM_ENVS:-1}"
export STEPS="${STEPS:-2000}"
export AUTO_PLOTS="${AUTO_PLOTS:-0}"
export SAVE_GIFS="${SAVE_GIFS:-0}"
exec "$DIR/run_anymal_proprio_gui.sh" env.time_limit=100 model.compile=false "$@"
