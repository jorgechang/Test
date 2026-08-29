"""Human-only target-pose overlay for the active Isaac Sim viewport.

This module deliberately does *not* use Isaac Sim DebugDraw and does not create
USD geometry.  In the user's Isaac Sim build, DebugDraw lines were present in
the TiledCamera RGB stream, so they were not suitable for a clean learning
observation.

Instead, this implementation attaches an :mod:`omni.ui.scene` ``SceneView`` to
the active Kit viewport.  SceneUI items are viewport UI overlays: they are not
part of the USD stage, do not participate in physics, and are not rendered by
Replicator/TiledCamera render products.

The numerical target remains in the environment and is still used for reward,
success, and the goal-conditioned state vector.  Only this visualization is
human-only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TargetDebugStyle:
    """Dimensions of the human-only target camera frame, in stage units."""

    arrow_length: float
    axis_length: float
    frustum_distance: float
    frustum_half_width: float
    frustum_half_height: float
    line_width: float = 4.0
    point_size: float = 10.0
    max_envs: int = 64


class TargetPoseDebugDraw:
    """Draw target camera poses in the active viewport UI only.

    The historical class name is retained so the Jetbot and ANYmal wrappers do
    not need special-case imports.  Internally this is a SceneUI viewport
    overlay, not DebugDraw.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        camera_height: float,
        style: TargetDebugStyle,
        strict: bool = False,
    ) -> None:
        self._requested = bool(enabled)
        self._strict = bool(strict)
        self._camera_height = float(camera_height)
        self._style = style

        self._viewport_window: Any | None = None
        self._scene_view: Any | None = None
        self._frame: Any | None = None
        self._frame_id = f"r2dreamer_target_pose_overlay_{id(self):x}"
        self._warned = False

        if self._requested:
            self._ensure_overlay()

    @property
    def enabled(self) -> bool:
        """Whether the SceneUI overlay is currently attached to a viewport."""
        return self._scene_view is not None

    def _report_failure(self, exc: Exception) -> None:
        if self._strict:
            raise RuntimeError(
                "show_target_debug=true, but the viewport-only SceneUI target "
                "overlay could not be initialized."
            ) from exc
        if not self._warned:
            print(
                "[TARGET OVERLAY][WARNING] Could not attach the human-only "
                f"target overlay to the active viewport: {exc}. The numerical "
                "goal remains active and Dreamer training continues."
            )
            self._warned = True

    def _ensure_overlay(self) -> bool:
        """Lazily attach a SceneView to the active Kit viewport."""
        if not self._requested:
            return False
        if self._scene_view is not None:
            return True

        try:
            from omni.kit.viewport.utility import get_active_viewport_window
            from omni.ui import scene as sc

            viewport_window = get_active_viewport_window()
            if viewport_window is None:
                raise RuntimeError("no active Kit viewport window is available yet")

            frame = viewport_window.get_frame(self._frame_id)
            with frame:
                scene_view = sc.SceneView()

            viewport_window.viewport_api.add_scene_view(scene_view)

            self._viewport_window = viewport_window
            self._frame = frame
            self._scene_view = scene_view
            print(
                "[TARGET OVERLAY] Human-only target camera pose enabled through "
                "omni.ui.scene. It is attached to the Kit viewport and is not "
                "part of the USD/TiledCamera render."
            )
            return True
        except Exception as exc:  # pragma: no cover - Kit/version dependent
            self._report_failure(exc)
            return False

    @staticmethod
    def _add(
        a: tuple[float, float, float],
        b: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    @staticmethod
    def _scale(
        vector: tuple[float, float, float],
        scale: float,
    ) -> tuple[float, float, float]:
        return (vector[0] * scale, vector[1] * scale, vector[2] * scale)

    def redraw(
        self,
        *,
        goal_xy: torch.Tensor,
        goal_yaw: torch.Tensor,
        env_origins: torch.Tensor,
    ) -> None:
        """Rebuild the viewport overlay from the current target poses."""
        if not self._requested or not self._ensure_overlay():
            return

        xy = goal_xy.detach().to(device="cpu", dtype=torch.float32)
        yaw = goal_yaw.detach().to(device="cpu", dtype=torch.float32)
        origins = env_origins.detach().to(device="cpu", dtype=torch.float32)

        if xy.ndim != 2 or xy.shape[-1] != 2:
            raise ValueError(f"goal_xy must have shape (N,2), got {tuple(xy.shape)}")
        if yaw.ndim != 1 or yaw.shape[0] != xy.shape[0]:
            raise ValueError(
                f"goal_yaw must have shape ({xy.shape[0]},), got {tuple(yaw.shape)}"
            )
        if (
            origins.ndim != 2
            or origins.shape[0] != xy.shape[0]
            or origins.shape[-1] < 3
        ):
            raise ValueError(
                "env_origins must have shape (N,>=3) and match goal count; "
                f"got {tuple(origins.shape)} for {xy.shape[0]} goals"
            )

        from omni.ui import scene as sc

        style = self._style
        count = min(int(xy.shape[0]), max(int(style.max_envs), 0))

        green = (0.05, 1.00, 0.05, 1.00)
        red = (1.00, 0.08, 0.08, 1.00)
        blue = (0.10, 0.35, 1.00, 1.00)
        cyan = (0.05, 0.90, 1.00, 1.00)
        yellow = (1.00, 0.90, 0.05, 1.00)

        scene = self._scene_view.scene
        scene.clear()

        with scene:
            for env_id in range(count):
                x = float(origins[env_id, 0] + xy[env_id, 0])
                y = float(origins[env_id, 1] + xy[env_id, 1])
                z = float(origins[env_id, 2] + self._camera_height)
                angle = float(yaw[env_id])

                forward = (math.cos(angle), math.sin(angle), 0.0)
                left = (-math.sin(angle), math.cos(angle), 0.0)
                up = (0.0, 0.0, 1.0)
                origin = (x, y, z)

                def line(
                    start: tuple[float, float, float],
                    end: tuple[float, float, float],
                    color: tuple[float, float, float, float],
                    thickness: float | None = None,
                ) -> None:
                    sc.Line(
                        start,
                        end,
                        color=color,
                        thickness=float(
                            style.line_width if thickness is None else thickness
                        ),
                    )

                # Target point.
                sc.Points(
                    [origin],
                    colors=[yellow],
                    sizes=[float(style.point_size)],
                )

                # Green viewing-direction arrow.
                arrow_tip = self._add(
                    origin, self._scale(forward, style.arrow_length)
                )
                line(origin, arrow_tip, green, style.line_width + 1.0)

                head_length = max(0.22 * style.arrow_length, 1.0e-3)
                back = (-forward[0], -forward[1], 0.0)
                spread = math.radians(28.0)
                head_left = (
                    math.cos(spread) * back[0] + math.sin(spread) * left[0],
                    math.cos(spread) * back[1] + math.sin(spread) * left[1],
                    0.0,
                )
                head_right = (
                    math.cos(spread) * back[0] - math.sin(spread) * left[0],
                    math.cos(spread) * back[1] - math.sin(spread) * left[1],
                    0.0,
                )
                line(
                    arrow_tip,
                    self._add(arrow_tip, self._scale(head_left, head_length)),
                    green,
                )
                line(
                    arrow_tip,
                    self._add(arrow_tip, self._scale(head_right, head_length)),
                    green,
                )

                # Red lateral and blue vertical axes.
                line(
                    origin,
                    self._add(origin, self._scale(left, style.axis_length)),
                    red,
                )
                line(
                    origin,
                    self._add(origin, self._scale(up, style.axis_length)),
                    blue,
                )

                # Cyan camera-frustum wireframe.
                center = self._add(
                    origin, self._scale(forward, style.frustum_distance)
                )
                left_offset = self._scale(left, style.frustum_half_width)
                up_offset = self._scale(up, style.frustum_half_height)

                corners = (
                    self._add(self._add(center, left_offset), up_offset),
                    self._add(self._add(center, left_offset), self._scale(up_offset, -1.0)),
                    self._add(
                        self._add(center, self._scale(left_offset, -1.0)),
                        self._scale(up_offset, -1.0),
                    ),
                    self._add(
                        self._add(center, self._scale(left_offset, -1.0)),
                        up_offset,
                    ),
                )

                for corner in corners:
                    line(origin, corner, cyan)
                for first, second in zip(corners, corners[1:] + corners[:1]):
                    line(first, second, cyan)

    def close(self) -> None:
        """Detach the SceneView and release all viewport UI resources."""
        scene_view = self._scene_view
        viewport_window = self._viewport_window

        self._scene_view = None
        self._viewport_window = None

        if scene_view is not None:
            try:
                scene_view.scene.clear()
            except Exception:
                pass
            if viewport_window is not None:
                try:
                    viewport_window.viewport_api.remove_scene_view(scene_view)
                except Exception:
                    pass

        if self._frame is not None:
            try:
                self._frame.clear()
            except Exception:
                pass
        self._frame = None
