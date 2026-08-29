#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROPRIOCEPTION_ENABLED=1
exec "$DIR/run_anymal_eval_gui.sh" "$@"
