#!/usr/bin/env python3
"""Isaac Lab CLI launcher for NM512/r2dreamer."""
from __future__ import annotations

import argparse
import os
import pathlib
import runpy
import sys

from isaaclab.app import AppLauncher


def _find_isaaclab_root(start: pathlib.Path) -> pathlib.Path:
    for candidate in (start, *start.parents):
        if (candidate / "isaaclab.sh").is_file():
            return candidate
    raise RuntimeError("Could not find IsaacLab root containing isaaclab.sh")


def _remove_override(args: list[str], key: str) -> list[str]:
    prefix = key + "="
    return [arg for arg in args if not arg.startswith(prefix)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NM512/r2dreamer on the Isaac Lab camera-pose task.",
        conflict_handler="resolve",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    here = pathlib.Path(__file__).resolve().parent
    isaaclab_root = _find_isaaclab_root(here)
    repo = isaaclab_root / "r2dreamer"
    if not (repo / "train.py").is_file():
        raise RuntimeError(f"NM512/r2dreamer checkout not found at {repo}")

    sys.path.insert(0, str(repo))
    import isaaclab_runtime

    isaaclab_runtime.set_runtime(app_launcher, simulation_app)
    hydra_args = _remove_override(hydra_args, "env.env_num")
    hydra_args.append(f"env.env_num={int(args_cli.num_envs)}")

    os.chdir(repo)
    sys.argv = [str(repo / "train.py"), *hydra_args]
    runpy.run_path(str(repo / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
