#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP="r2dreamer_isaaclab_anymal_v13_45_simple_runs_three_checkpoints_proprio_complete.zip"
OUT="$ROOT/$ZIP"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

cat "$ROOT"/bundle_parts/part_*.b64 > "$TMP"
base64 -d "$TMP" > "$OUT"

EXPECTED="553db463a3be0149b90a89c319df22b31adcedefd870607c1464e34ed4ae0aff"
ACTUAL="$(sha256sum "$OUT" | awk '{print $1}')"

if [[ "$ACTUAL" != "$EXPECTED" ]]; then
    echo "ERROR: checksum mismatch"
    echo "expected: $EXPECTED"
    echo "actual:   $ACTUAL"
    rm -f "$OUT"
    exit 1
fi

unzip -t "$OUT" >/dev/null

echo "Created and verified: $OUT"
echo "$ACTUAL  $ZIP"
