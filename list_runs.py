#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="List v13.45 ANYmal run folders.")
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    rows = []
    if root.is_dir():
        for status_path in root.rglob("run_status.json"):
            run = status_path.parent
            status = load(status_path)
            params = load(run / "config" / "parameters.json")
            checkpoints = load(run / "checkpoint_summary.json")
            latest = checkpoints.get("latest") or {}
            best_success = checkpoints.get("best_success") or {}
            best_reward = checkpoints.get("best_reward") or {}
            success_selection = best_success.get("selection") or {}
            reward_selection = best_reward.get("selection") or {}
            rows.append(
                {
                    "started": status.get("started_at_utc", params.get("started_at_utc", "?")),
                    "state": status.get("state", "?"),
                    "variant": params.get("variant", status.get("variant", "?")),
                    "mode": params.get("mode", status.get("mode", "?")),
                    "run": run.name,
                    "latest_step": latest.get("step", "-"),
                    "success_step": best_success.get("step", "-"),
                    "success": success_selection.get("success_rate"),
                    "reward_step": best_reward.get("step", "-"),
                    "reward": reward_selection.get("mean_return"),
                    "path": str(run),
                }
            )
    rows.sort(key=lambda row: str(row["started"]), reverse=True)
    if not rows:
        print(f"No runs found under {root}")
        return 0

    print(
        f"{'STARTED (UTC)':20}  {'STATE':21}  {'VARIANT':12}  {'MODE':8}  "
        f"{'LATEST':>9}  {'SUCCESS':>9}  {'RATE':>8}  {'REWARD':>9}  {'RETURN':>10}  RUN"
    )
    for row in rows:
        success = "-" if row["success"] is None else f"{100.0 * float(row['success']):6.2f}%"
        reward = "-" if row["reward"] is None else f"{float(row['reward']):10.3f}"
        print(
            f"{str(row['started']):20.20}  {str(row['state']):21.21}  "
            f"{str(row['variant']):12.12}  {str(row['mode']):8.8}  "
            f"{str(row['latest_step']):>9.9}  {str(row['success_step']):>9.9}  "
            f"{success:>8}  {str(row['reward_step']):>9.9}  {reward:>10}  {row['run']}"
        )
        print(f"  {row['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
