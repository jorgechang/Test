"""Shared Isaac Lab runtime created by the CLI launcher before Hydra starts."""
from __future__ import annotations

_SIMULATION_APP = None
_APP_LAUNCHER = None


def set_runtime(app_launcher, simulation_app) -> None:
    global _APP_LAUNCHER, _SIMULATION_APP
    _APP_LAUNCHER = app_launcher
    _SIMULATION_APP = simulation_app


def get_simulation_app():
    return _SIMULATION_APP


def get_app_launcher():
    return _APP_LAUNCHER
