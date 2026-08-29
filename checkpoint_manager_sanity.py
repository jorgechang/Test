#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import torch

from camera_pose_checkpoints import CheckpointManager


class FakeAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


def episode(manager: CheckpointManager, *, success: float, ret: float) -> None:
    manager.record_episode(
        score=ret,
        length=100,
        metrics={"log_success": success},
    )


with tempfile.TemporaryDirectory() as td:
    logdir = Path(td) / "run"
    cfg = SimpleNamespace(
        checkpoint_every=10_000,
        best_checkpoint_window=10,
        best_checkpoint_min_episodes=5,
    )
    manager = CheckpointManager(logdir, cfg)
    agent = FakeAgent()
    assert not manager.should_save(9_999)
    assert manager.should_save(10_000)

    for _ in range(10):
        episode(manager, success=0.0, ret=40.0)
    manager.save(agent, 10_000, reason="periodic")
    assert (logdir / "latest.pt").is_file()
    assert (logdir / "best_success.pt").is_file()
    assert (logdir / "best_reward.pt").is_file()

    # Success improves but return falls: only best_success should move.
    for _ in range(10):
        episode(manager, success=0.5, ret=30.0)
    manager.save(agent, 20_000, reason="periodic")
    metadata = json.loads((logdir / "checkpoint_summary.json").read_text())
    assert metadata["best_success"]["step"] == 20_000
    assert metadata["best_reward"]["step"] == 10_000
    assert abs(metadata["best_success"]["selection"]["success_rate"] - 0.5) < 1e-9

    # Return improves while success becomes worse: only best_reward should move.
    for _ in range(10):
        episode(manager, success=0.0, ret=100.0)
    manager.save(agent, 30_000, reason="periodic")
    metadata = json.loads((logdir / "checkpoint_summary.json").read_text())
    assert metadata["best_success"]["step"] == 20_000
    assert metadata["best_reward"]["step"] == 30_000
    assert metadata["latest"]["step"] == 30_000
    latest = torch.load(logdir / "latest.pt", map_location="cpu", weights_only=False)
    assert latest["checkpoint_metadata"]["step"] == 30_000

    checkpoint_names = sorted(path.name for path in logdir.glob("*.pt"))
    assert checkpoint_names == ["best_reward.pt", "best_success.pt", "latest.pt"]
    assert not (logdir / "checkpoints").exists()

with tempfile.TemporaryDirectory() as td:
    # A short smoke test still gets all three paths, clearly marked as fallbacks.
    logdir = Path(td) / "short"
    manager = CheckpointManager(
        logdir,
        SimpleNamespace(
            checkpoint_every=10_000,
            best_checkpoint_window=100,
            best_checkpoint_min_episodes=100,
        ),
    )
    manager.save(FakeAgent(), 123, reason="final")
    metadata = json.loads((logdir / "checkpoint_summary.json").read_text())
    assert metadata["best_success"]["fallback_to_latest"] is True
    assert metadata["best_reward"]["fallback_to_latest"] is True
    assert sorted(path.name for path in logdir.glob("*.pt")) == [
        "best_reward.pt",
        "best_success.pt",
        "latest.pt",
    ]

print("[CHECKPOINT SANITY] latest, best-success, best-reward, metadata, and fallback passed")
