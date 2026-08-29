#!/usr/bin/env python3
"""Minimal checkpoint manager for ANYmal camera-pose training.

Only three model files are retained inside each run:

* ``latest.pt``: newest periodic/final checkpoint;
* ``best_success.pt``: highest rolling success rate;
* ``best_reward.pt``: highest rolling mean episode return.

The manager observes training episode logs only. It does not change rewards,
actions, replay, optimization, or policy inputs.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import torch


def _cfg(config: Any, name: str, default: Any) -> Any:
    try:
        value = config.get(name, default)
    except Exception:
        value = getattr(config, name, default)
    return default if value is None else value


def _scalar(value: Any, default: float = math.nan) -> float:
    try:
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().item()
        value = float(value)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _mean(values: list[float], default: float) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float(default)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(dict(data), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _torch_save_atomic(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    torch.save(dict(payload), tmp)
    os.replace(tmp, destination)


def _clone_atomic(source: Path, destination: Path) -> None:
    """Copy a stable checkpoint, preferring a same-filesystem hard link."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    tmp.unlink(missing_ok=True)
    try:
        os.link(source, tmp)
    except OSError:
        shutil.copy2(source, tmp)
    os.replace(tmp, destination)


@dataclass(frozen=True)
class SelectionSummary:
    episodes_in_window: int
    completed_episodes: int
    success_rate: float
    mean_return: float

    @property
    def success_key(self) -> tuple[float, float]:
        # Success is primary; return only breaks exact success-rate ties.
        return (round(self.success_rate, 9), round(self.mean_return, 9))

    @property
    def reward_key(self) -> tuple[float, float]:
        # Return is primary; success only breaks exact return ties.
        return (round(self.mean_return, 9), round(self.success_rate, 9))

    def as_dict(self) -> dict[str, Any]:
        return {
            "episodes_in_window": self.episodes_in_window,
            "completed_episodes": self.completed_episodes,
            "success_rate": self.success_rate,
            "mean_return": self.mean_return,
        }


class CheckpointManager:
    """Retain only latest, best-success, and best-reward checkpoints."""

    def __init__(self, logdir: str | Path, config: Any):
        self.logdir = Path(logdir).expanduser().resolve()
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_every = max(int(_cfg(config, "checkpoint_every", 10_000)), 0)
        self.window = max(int(_cfg(config, "best_checkpoint_window", 100)), 1)
        self.min_episodes = max(
            int(_cfg(config, "best_checkpoint_min_episodes", self.window)), 1
        )
        self.min_episodes = min(self.min_episodes, self.window)

        self.rows: deque[dict[str, float]] = deque(maxlen=self.window)
        self.completed_episodes = 0
        self.best_success_key: tuple[float, float] | None = None
        self.best_reward_key: tuple[float, float] | None = None
        self.last_saved_step = -1
        self.next_save_step: float = (
            self.checkpoint_every if self.checkpoint_every > 0 else math.inf
        )
        self.metadata: dict[str, Any] = {
            "criteria": {
                "best_success": (
                    "maximum rolling success rate; rolling mean episode return "
                    "breaks exact ties"
                ),
                "best_reward": (
                    "maximum rolling mean episode return; rolling success rate "
                    "breaks exact ties"
                ),
                "window_episodes": self.window,
                "minimum_episodes": self.min_episodes,
            },
            "latest": None,
            "best_success": None,
            "best_reward": None,
        }
        self._restore_existing_state()

    @property
    def latest_path(self) -> Path:
        return self.logdir / "latest.pt"

    @property
    def best_success_path(self) -> Path:
        return self.logdir / "best_success.pt"

    @property
    def best_reward_path(self) -> Path:
        return self.logdir / "best_reward.pt"

    @property
    def summary_path(self) -> Path:
        return self.logdir / "checkpoint_summary.json"

    def _restore_existing_state(self) -> None:
        if self.summary_path.is_file():
            try:
                saved = json.loads(self.summary_path.read_text(encoding="utf-8"))
                if isinstance(saved, dict):
                    self.metadata.update(saved)
                latest = saved.get("latest") or {}
                self.last_saved_step = int(latest.get("step", -1))
                success = (saved.get("best_success") or {}).get("selection") or {}
                reward = (saved.get("best_reward") or {}).get("selection") or {}
                if success:
                    self.best_success_key = (
                        float(success.get("success_rate", -math.inf)),
                        float(success.get("mean_return", -math.inf)),
                    )
                if reward:
                    self.best_reward_key = (
                        float(reward.get("mean_return", -math.inf)),
                        float(reward.get("success_rate", -math.inf)),
                    )
            except Exception:
                pass

        self._restore_window_from_metrics()
        if self.checkpoint_every > 0 and self.last_saved_step >= 0:
            self.next_save_step = (
                self.last_saved_step // self.checkpoint_every + 1
            ) * self.checkpoint_every

    def _restore_window_from_metrics(self) -> None:
        metrics_path = self.logdir / "metrics.jsonl"
        if not metrics_path.is_file():
            return
        restored: list[dict[str, float]] = []
        try:
            lines = metrics_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            return
        for line in lines:
            try:
                raw = json.loads(line)
            except Exception:
                continue
            if "episode/score" not in raw:
                continue
            restored.append(
                {
                    "score": _scalar(raw.get("episode/score")),
                    "success": _scalar(raw.get("episode/log_success"), 0.0),
                }
            )
        self.completed_episodes = len(restored)
        self.rows.extend(restored[-self.window :])

    def record_episode(
        self, *, score: Any, length: Any, metrics: Mapping[str, Any]
    ) -> None:
        del length  # Selection uses only success and episode return.
        self.rows.append(
            {
                "score": _scalar(score),
                "success": _scalar(metrics.get("log_success", 0.0), 0.0),
            }
        )
        self.completed_episodes += 1

    def should_save(self, step: int) -> bool:
        if self.checkpoint_every <= 0:
            return False
        step = int(step)
        if step < self.next_save_step:
            return False
        while self.next_save_step <= step:
            self.next_save_step += self.checkpoint_every
        return True

    def summary(self) -> SelectionSummary | None:
        if not self.rows:
            return None
        rows = list(self.rows)
        successes = [
            min(max(row.get("success", 0.0), 0.0), 1.0) for row in rows
        ]
        returns = [row.get("score", math.nan) for row in rows]
        return SelectionSummary(
            episodes_in_window=len(rows),
            completed_episodes=self.completed_episodes,
            success_rate=_mean(successes, 0.0),
            mean_return=_mean(returns, -1.0e9),
        )

    def _checkpoint_payload(
        self,
        agent: Any,
        step: int,
        reason: str,
        summary: SelectionSummary | None,
    ) -> dict[str, Any]:
        try:
            import tools  # type: ignore

            optim_state = tools.recursively_collect_optim_state_dict(agent)
        except Exception:
            optim_state = {}
        metadata: dict[str, Any] = {
            "format_version": 3,
            "step": int(step),
            "reason": reason,
            "saved_at_utc": _utc_now(),
            "completed_episodes": self.completed_episodes,
            "selection_window": self.window,
            "minimum_selection_episodes": self.min_episodes,
        }
        if summary is not None:
            metadata["selection"] = summary.as_dict()
        return {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": optim_state,
            "checkpoint_metadata": metadata,
        }

    def _entry(
        self,
        *,
        kind: str,
        path: Path,
        step: int,
        reason: str,
        summary: SelectionSummary | None,
        fallback: bool = False,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "kind": kind,
            "path": str(path),
            "step": int(step),
            "reason": reason,
            "saved_at_utc": _utc_now(),
            "completed_episodes": self.completed_episodes,
            "fallback_to_latest": bool(fallback),
        }
        if summary is not None:
            entry["selection"] = summary.as_dict()
        return entry

    def save(self, agent: Any, step: int, *, reason: str) -> None:
        step = max(int(step), 0)
        summary = self.summary()
        _torch_save_atomic(
            self._checkpoint_payload(agent, step, reason, summary), self.latest_path
        )
        self.last_saved_step = step
        self.metadata["latest"] = self._entry(
            kind="latest",
            path=self.latest_path,
            step=step,
            reason=reason,
            summary=summary,
        )

        eligible = summary is not None and summary.episodes_in_window >= self.min_episodes
        if eligible and (
            self.best_success_key is None or summary.success_key > self.best_success_key
        ):
            _clone_atomic(self.latest_path, self.best_success_path)
            self.best_success_key = summary.success_key
            self.metadata["best_success"] = self._entry(
                kind="best_success",
                path=self.best_success_path,
                step=step,
                reason=reason,
                summary=summary,
            )

        if eligible and (
            self.best_reward_key is None or summary.reward_key > self.best_reward_key
        ):
            _clone_atomic(self.latest_path, self.best_reward_path)
            self.best_reward_key = summary.reward_key
            self.metadata["best_reward"] = self._entry(
                kind="best_reward",
                path=self.best_reward_path,
                step=step,
                reason=reason,
                summary=summary,
            )

        # Short smoke tests may finish before the requested window is available.
        # Keep all three paths usable, while marking the two best files as fallbacks.
        if reason == "final" and not self.best_success_path.exists():
            _clone_atomic(self.latest_path, self.best_success_path)
            self.metadata["best_success"] = self._entry(
                kind="best_success",
                path=self.best_success_path,
                step=step,
                reason="fallback_to_final_latest",
                summary=summary,
                fallback=True,
            )
        if reason == "final" and not self.best_reward_path.exists():
            _clone_atomic(self.latest_path, self.best_reward_path)
            self.metadata["best_reward"] = self._entry(
                kind="best_reward",
                path=self.best_reward_path,
                step=step,
                reason="fallback_to_final_latest",
                summary=summary,
                fallback=True,
            )

        self.metadata["criteria"] = {
            "best_success": (
                "maximum rolling success rate; rolling mean episode return "
                "breaks exact ties"
            ),
            "best_reward": (
                "maximum rolling mean episode return; rolling success rate "
                "breaks exact ties"
            ),
            "window_episodes": self.window,
            "minimum_episodes": self.min_episodes,
        }
        _write_json_atomic(self.summary_path, self.metadata)

        if summary is None:
            selection_text = "no completed episodes"
        else:
            selection_text = (
                f"success={100.0 * summary.success_rate:.2f}% "
                f"return={summary.mean_return:.3f} "
                f"window={summary.episodes_in_window}"
            )
        print(
            f"[CHECKPOINT] reason={reason} step={step} latest={self.latest_path} | "
            f"{selection_text}"
        )
