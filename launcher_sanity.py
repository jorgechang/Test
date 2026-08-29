#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def run_launcher(bundle: Path, name: str, extra_env: dict[str, str]) -> list[str]:
    output = bundle.parent / f"{name}.{len(list(bundle.parent.glob(name + '.*.args')))}.args"
    env = os.environ.copy()
    env.update(
        {
            "FAKE_ARGS_OUT": str(output),
            "GOALVEC_SKIP_INSTALL": "1",
            "AUTO_PLOTS": "0",
            "SAVE_GIFS": "0",
            "NUM_ENVS": "2",
            "STEPS": "1234",
            "SEED": "11",
            "TASK_SEED": "22",
            "RUN_TAG": "fixed_tag",
        }
    )
    env.update(extra_env)
    subprocess.run(
        ["bash", str(bundle / name), "test_override=kept"],
        check=True,
        cwd=bundle.parent,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return output.read_text(encoding="utf-8").splitlines()


def require_arg(args: list[str], expected: str) -> None:
    if expected not in args:
        raise AssertionError(f"Missing launcher argument {expected!r}: {args}")


with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "IsaacLab-beta"
    root.mkdir()
    bundle = root / HERE.name
    shutil.copytree(
        HERE,
        bundle,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "BUNDLE_MANIFEST.sha256"),
    )

    fake = root / "isaaclab.sh"
    fake.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nprintf "%s\\n" "$@" > "$FAKE_ARGS_OUT"\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    repo = root / "r2dreamer"
    repo.mkdir()
    (repo / "train.py").write_text("# mock r2dreamer root\n", encoding="utf-8")

    rgb = run_launcher(bundle, "run_anymal_headless.sh", {})
    for expected in (
        "seed=11",
        "env.task_seed=22",
        "env.balanced_randomization=true",
        "++trainer.checkpoint_every=10000",
        "++trainer.best_checkpoint_window=100",
        "++trainer.best_checkpoint_min_episodes=100",
        "test_override=kept",
    ):
        require_arg(rgb, expected)
    assert "env.proprioception_enabled=true" not in rgb
    assert "env.encoder.mlp_keys=proprio" not in rgb
    assert "env.decoder.mlp_keys=proprio" not in rgb
    assert not any("archive" in arg or "best_checkpoint_check" in arg for arg in rgb)

    # The same launcher and tag still receive isolated run folders.
    run_launcher(bundle, "run_anymal_headless.sh", {})
    run_base = repo / "logdir/isaaclab_anymal_v13_45/rgb_only"
    runs_root = run_base / "runs"
    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    assert len(run_dirs) == 2, run_dirs
    assert (run_base / "latest_run").resolve() == run_dirs[-1].resolve()
    for run_dir in run_dirs:
        assert (run_dir / "config/parameters.yaml").is_file()
        assert (run_dir / "config/parameters.json").is_file()
        assert (run_dir / "run_status.json").is_file()
        assert (run_dir / "train.log").is_file()
        for forbidden in (
            "system_info.txt",
            "python_packages.txt",
            "runtime_environment.json",
            "r2dreamer_git.yaml",
            "r2dreamer_git_diff.patch",
            "unresolved_config.yaml",
        ):
            assert not (run_dir / "config" / forbidden).exists()
        assert not (run_dir / "source_snapshot").exists()

    # Concurrent launches with the same timestamp/tag reserve distinct folders.
    processes: list[subprocess.Popen[str]] = []
    for index in range(2):
        env = os.environ.copy()
        env.update(
            {
                "FAKE_ARGS_OUT": str(root / f"parallel_{index}.args"),
                "GOALVEC_SKIP_INSTALL": "1",
                "AUTO_PLOTS": "0",
                "SAVE_GIFS": "0",
                "NUM_ENVS": "1",
                "STEPS": "10",
                "SEED": "5",
                "TASK_SEED": "6",
                "RUN_TAG": "parallel",
                "RUN_TIMESTAMP": "20260101T000000Z",
            }
        )
        processes.append(
            subprocess.Popen(
                ["bash", str(bundle / "run_anymal_headless.sh")],
                cwd=bundle.parent,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    for process in processes:
        stdout, stderr = process.communicate(timeout=60)
        if process.returncode != 0:
            raise AssertionError(f"parallel launcher failed: {stdout}\n{stderr}")
    parallel_dirs = sorted(runs_root.glob("20260101T000000Z_headless_parallel*"))
    assert len(parallel_dirs) == 2 and parallel_dirs[0].name != parallel_dirs[1].name

    proprio = run_launcher(bundle, "run_anymal_proprio_headless.sh", {})
    for expected in (
        "env.proprioception_enabled=true",
        "env.encoder.mlp_keys=proprio",
        "env.decoder.mlp_keys=proprio",
        "++model.loss_scales.proprio=1.0",
    ):
        require_arg(proprio, expected)
    assert all("goal_vec" not in arg for arg in proprio)

    both = run_launcher(
        bundle,
        "run_anymal_proprio_headless.sh",
        {
            "GOALVEC_DECODER_ENABLED": "1",
            "GOALVEC_LOSS_SCALE": "77",
            "PROPRIO_LOSS_SCALE": "0.5",
        },
    )
    for expected in (
        "env.proprioception_enabled=true",
        "env.encoder.mlp_keys=proprio",
        "env.decoder.mlp_keys=proprio|goal_vec",
        "++model.loss_scales.proprio=0.5",
        "++model.loss_scales.goal_vec=77",
    ):
        require_arg(both, expected)

    # Generic evaluation reloads only architecture/task parameters from the run.
    saved_run = repo / "logdir/isaaclab_anymal_v13_45/rgb_proprio/runs/saved_config_run"
    (saved_run / "config").mkdir(parents=True)
    saved_checkpoint = saved_run / "latest.pt"
    saved_checkpoint.write_bytes(b"saved")
    parameters = {
        "variant": "rgb_proprio",
        "hyperparameters": {
            "MODEL_HORIZON": "777",
            "IMAGE_SIZE": "64",
            "SEED": "31",
            "TASK_SEED": "123456",
            "BALANCED_RANDOMIZATION": "0",
            "PROPRIOCEPTION_ENABLED": "1",
            "PROPRIO_LOSS_SCALE": "0.25",
            "GOALVEC_DECODER_ENABLED": "1",
            "GOALVEC_LOSS_SCALE": "55",
            "ARROW_TARGET_ANCHOR": "tip",
        },
    }
    (saved_run / "config/parameters.json").write_text(
        json.dumps(parameters), encoding="utf-8"
    )
    saved_args_path = root / "saved_eval.args"
    saved_env = os.environ.copy()
    for key in parameters["hyperparameters"]:
        saved_env.pop(key, None)
    saved_env.update(
        {
            "FAKE_ARGS_OUT": str(saved_args_path),
            "GOALVEC_SKIP_INSTALL": "1",
            "SAVE_GIFS": "0",
            "RUN_DIR": str(saved_run),
            "CHECKPOINT_KIND": "latest",
            "TASK_SEED": "999",  # deliberately not inherited
            "IMAGE_SIZE": "96",  # explicit override beats saved value
        }
    )
    subprocess.run(
        ["bash", str(bundle / "run_anymal_eval.sh")],
        check=True,
        cwd=bundle.parent,
        env=saved_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    saved_args = saved_args_path.read_text(encoding="utf-8").splitlines()
    for expected in (
        str(saved_checkpoint.resolve()),
        "model.horizon=777",
        "env.size=[96,96]",
        "seed=31",
        "env.task_seed=999",
        "env.balanced_randomization=false",
        "env.arrow_target_anchor=tip",
        "env.proprioception_enabled=true",
        "env.encoder.mlp_keys=proprio",
        "env.decoder.mlp_keys=proprio|goal_vec",
        "++model.loss_scales.proprio=0.25",
        "++model.loss_scales.goal_vec=55",
    ):
        require_arg(saved_args, expected)

print(
    "[LAUNCHER SANITY] unique run folders, minimal parameters, three-checkpoint "
    "configuration, saved-config recovery, RGB/proprio, seed, and eval routing passed"
)
