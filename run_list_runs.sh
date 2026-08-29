#!/usr/bin/env bash
set -euo pipefail
export GOALVEC_SKIP_INSTALL="${GOALVEC_SKIP_INSTALL:-1}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
exec python "$BUNDLE_DIR/list_runs.py" --root "$LOG_ROOT"
