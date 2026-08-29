#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def initialize(root: Path, logdir: Path, latest_link: Path, run_id: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(HERE / "run_manager.py"),
            "init",
            "--logdir",
            str(logdir),
            "--mode",
            "headless",
            "--variant",
            "rgb_only",
            "--run-id",
            run_id,
            "--run-tag",
            "seed3_task7",
            "--launcher",
            "run_anymal_headless.sh",
            "--latest-link",
            str(latest_link),
            "--param",
            "SEED=3",
            "--param",
            "TASK_SEED=7",
            "--",
            "/bin/echo",
            "hello world",
            "seed=3",
        ],
        check=True,
    )


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    variant_root = root / "logs" / "rgb_only"
    latest_link = variant_root / "latest_run"
    run1 = variant_root / "runs" / "run_001"
    run2 = variant_root / "runs" / "run_002"

    initialize(root, run1, latest_link, "run_001")
    assert (run1 / "config/parameters.yaml").is_file()
    assert (run1 / "config/parameters.json").is_file()
    assert (run1 / "run_status.json").is_file()
    assert (run1 / "plots").is_dir()
    assert (run1 / "gifs").is_dir()
    assert (run1 / "evaluations").is_dir()
    assert latest_link.resolve() == run1.resolve()

    for forbidden in (
        "system_info.txt",
        "python_packages.txt",
        "runtime.json",
        "runtime_environment.json",
        "r2dreamer_git.yaml",
        "r2dreamer_git_diff.patch",
        "unresolved_config.yaml",
    ):
        assert not any(path.name == forbidden for path in run1.rglob("*"))
    assert not (run1 / "source_snapshot").exists()

    params = json.loads((run1 / "config/parameters.json").read_text())
    assert params["hyperparameters"]["SEED"] == "3"
    assert params["hyperparameters"]["TASK_SEED"] == "7"
    assert params["command"][-1] == "seed=3"
    status = json.loads((run1 / "run_status.json").read_text())
    assert status["state"] == "running"

    initialize(root, run2, latest_link, "run_002")
    assert run1.is_dir()
    assert latest_link.resolve() == run2.resolve()

    subprocess.run(
        [
            sys.executable,
            str(HERE / "run_manager.py"),
            "finish",
            "--logdir",
            str(run2),
            "--exit-code",
            "0",
        ],
        check=True,
    )
    status = json.loads((run2 / "run_status.json").read_text())
    assert status["state"] == "completed"
    assert status["exit_code"] == 0

print("[RUN MANAGER SANITY] minimal parameters, atomic latest_run link, and status passed")
