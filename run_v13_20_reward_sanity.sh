#!/usr/bin/env bash
set -euo pipefail
python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/v13_20_reward_sanity.py"
