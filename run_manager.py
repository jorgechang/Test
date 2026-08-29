#!/usr/bin/env python3
"""Create and finalize lightweight, isolated experiment run directories."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def absolute(path: Path) -> Path:
    # Do not resolve latest_run itself: an existing symlink must be replaced.
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def write_simple_yaml(path: Path, mapping: dict[str, Any]) -> None:
    lines: list[str] = []
    for key, value in mapping.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for subkey, subvalue in value.items():
                lines.append(f"  {subkey}: {yaml_scalar(subvalue)}")
        elif isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {yaml_scalar(item)}" for item in value)
        else:
            lines.append(f"{key}: {yaml_scalar(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_params(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE parameter, got: {item}")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def update_latest_link(link_arg: Path, logdir: Path) -> None:
    link = absolute(link_arg)
    link.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(logdir, link.parent)
    tmp = link.with_name(f".{link.name}.tmp-{os.getpid()}")
    try:
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(relative_target)
        os.replace(tmp, link)
    except OSError:
        tmp.unlink(missing_ok=True)
        (link.parent / f"{link.name}.txt").write_text(str(logdir) + "\n", encoding="utf-8")


def initialize(args: argparse.Namespace) -> None:
    logdir = absolute(args.logdir)
    config_dir = logdir / "config"
    for directory in (config_dir, logdir / "plots", logdir / "gifs", logdir / "evaluations"):
        directory.mkdir(parents=True, exist_ok=True)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    start = utc_now()
    parameters: dict[str, Any] = {
        "bundle_version": "v13.45",
        "run_id": args.run_id,
        "run_tag": args.run_tag,
        "variant": args.variant,
        "mode": args.mode,
        "launcher": args.launcher,
        "hyperparameters": parse_params(args.param),
        "command": command,
    }
    write_simple_yaml(config_dir / "parameters.yaml", parameters)
    atomic_json(config_dir / "parameters.json", parameters)

    status = {
        "bundle_version": "v13.45",
        "run_id": args.run_id,
        "run_tag": args.run_tag,
        "variant": args.variant,
        "mode": args.mode,
        "launcher": args.launcher,
        "state": "running",
        "started_at_utc": start,
        "started_at_epoch": time.time(),
        "finished_at_utc": None,
        "finished_at_epoch": None,
        "duration_seconds": None,
        "exit_code": None,
        "logdir": str(logdir),
    }
    atomic_json(logdir / "run_status.json", status)
    if args.latest_link is not None:
        update_latest_link(args.latest_link, logdir)
    print(f"[RUN] initialized {logdir}")


def finish(args: argparse.Namespace) -> None:
    logdir = absolute(args.logdir)
    status_path = logdir / "run_status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        status = {"logdir": str(logdir)}
    code = int(args.exit_code)
    finish_time = utc_now()
    finish_epoch = time.time()
    try:
        duration = max(finish_epoch - float(status.get("started_at_epoch")), 0.0)
    except Exception:
        duration = None
    status.update(
        {
            "state": "completed" if code == 0 else "failed_or_interrupted",
            "finished_at_utc": finish_time,
            "finished_at_epoch": finish_epoch,
            "duration_seconds": duration,
            "exit_code": code,
            "artifacts": {
                "metrics_jsonl": (logdir / "metrics.jsonl").is_file(),
                "train_log": (logdir / "train.log").is_file(),
                "plot_count": len(list((logdir / "plots").glob("*.png"))),
                "latest_checkpoint": (logdir / "latest.pt").is_file(),
                "best_success_checkpoint": (logdir / "best_success.pt").is_file(),
                "best_reward_checkpoint": (logdir / "best_reward.pt").is_file(),
                "resolved_config": (logdir / "config" / "resolved_config.yaml").is_file(),
            },
        }
    )
    atomic_json(status_path, status)
    print(f"[RUN] finalized {logdir} exit_code={code}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command_name", required=True)

    init = sub.add_parser("init")
    init.add_argument("--logdir", required=True, type=Path)
    init.add_argument("--mode", required=True)
    init.add_argument("--variant", required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--run-tag", required=True)
    init.add_argument("--launcher", required=True)
    init.add_argument("--latest-link", type=Path)
    init.add_argument("--param", action="append", default=[])
    init.add_argument("command", nargs=argparse.REMAINDER)

    done = sub.add_parser("finish")
    done.add_argument("--logdir", required=True, type=Path)
    done.add_argument("--exit-code", required=True, type=int)

    args = parser.parse_args()
    if args.command_name == "init":
        initialize(args)
    else:
        finish(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
