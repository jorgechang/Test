#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_ENVS="${NUM_ENVS:-1}" STEPS="${STEPS:-2000}" AUTO_PLOTS=0 SAVE_GIFS=0 \
  "$DIR/run_anymal_gui.sh" env.time_limit=100 model.compile=false "$@"
