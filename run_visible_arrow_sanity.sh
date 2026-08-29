#!/usr/bin/env bash
set -euo pipefail
python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/visible_arrow_sanity.py"
