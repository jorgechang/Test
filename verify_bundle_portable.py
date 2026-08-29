#!/usr/bin/env python3
"""Portable offline verifier for the Git-cloned v13.45 bundle.

The original ``verify_bundle.py`` was designed for the packaged ZIP. Two details
make it unsuitable as-is on OSCAR:

1. Its per-method contract hashes use ``ast.dump()``, whose serialization differs
   between Python 3.12 (Isaac Sim 6.0.1 on OSCAR) and the Python version used to
   package the bundle. Identical source can therefore produce different hashes.
2. Its final manifest check requires that *no extra files* exist. A normal
   ``git clone`` necessarily adds ``.git/``, and this portable helper itself is
   also an extra file.

This verifier avoids both false failures without weakening the distributed-code
integrity check: it verifies the SHA-256 of every file listed in the original
bundle manifest (including the complete ANYmal environment source), then runs all
non-AST contract checks and all behavioral sanity scripts from the original
verifier. Training/environment code is not modified.
"""
from __future__ import annotations

import hashlib
import pathlib
import py_compile
import subprocess
import sys

import verify_bundle as v

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = HERE / "BUNDLE_MANIFEST.sha256"


def verify_distributed_manifest() -> None:
    """Verify every file shipped in the original v13.45 bundle byte-for-byte.

    Extra files such as ``.git/`` and this helper are intentionally allowed.
    """
    v.req(MANIFEST.is_file(), "bundle manifest exists")
    count = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        rel = rel.strip()
        path = HERE / rel
        v.req(path.is_file(), f"manifest file exists: {rel}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        v.req(actual == digest, f"manifest hash: {rel}")
        count += 1
    v.req(count > 0, "manifest contains distributed files")


def verify_exact_task_source() -> None:
    """Make the old AST contract portable by checking the entire source file."""
    expected = "9fa536020b654b3190dc86899342d1f1b9685d785ceebe48644fea57a9970a72"
    actual = hashlib.sha256(v.ENV.read_bytes()).hexdigest()
    v.req(actual == expected, "exact distributed ANYmal environment source")
    print(
        "[VERIFY] INFO: exact environment-file SHA replaces Python-version-dependent "
        "per-method ast.dump hashes"
    )


def main() -> None:
    # Integrity first: this is stronger than checking only selected methods and is
    # independent of Python's AST serialization format.
    verify_distributed_manifest()
    verify_exact_task_source()

    # All remaining static/behavioral contracts from the original verifier.
    v.check_config_contract()
    v.check_action_extension()
    v.check_training_budget()
    v.check_camera_contract()
    v.check_scene_and_sampler()
    v.check_reward_source()
    v.check_cleanup_and_logging()
    v.check_arrow_and_rgb_routing()
    v.check_proprioception_routing()
    v.check_install_and_launchers()
    v.check_run_management_and_best_checkpoints()

    # Syntax checks. Ignore Git internals; only distributed/runtime Python matters.
    for path in HERE.rglob("*.py"):
        if ".git" in path.parts:
            continue
        py_compile.compile(str(path), doraise=True)
    print("[VERIFY] OK: Python syntax")

    for path in HERE.glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    print("[VERIFY] OK: shell syntax")

    # Same behavioral/offline sanity suite as verify_bundle.py.
    for script in (
        "v13_20_reward_sanity.py",
        "visible_arrow_sanity.py",
        "object_layout_sanity.py",
        "proprioception_sanity.py",
        "camera_mount_sanity.py",
        "action_space_sanity.py",
        "installer_sanity.py",
        "launcher_sanity.py",
        "checkpoint_manager_sanity.py",
        "run_manager_sanity.py",
    ):
        subprocess.run([sys.executable, str(HERE / script)], check=True)

    print(
        "[VERIFY] OK: reward, arrow, balanced randomization, proprioception, "
        "camera mount, action-space, installer, launcher, run-manager, "
        "checkpoint-manager sanity"
    )
    print("[VERIFY] ALL OFFLINE CHECKS PASSED")


if __name__ == "__main__":
    main()
