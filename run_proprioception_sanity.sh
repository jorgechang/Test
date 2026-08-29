#!/usr/bin/env bash
set -euo pipefail
python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/proprioception_sanity.py"
