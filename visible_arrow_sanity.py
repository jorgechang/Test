#!/usr/bin/env python3
from __future__ import annotations
import ast
import math
import pathlib
import textwrap
import types
import torch

HERE = pathlib.Path(__file__).resolve().parent
ENV_PATH = HERE / "payload/envs/isaaclab_anymal_room.py"
SRC = ENV_PATH.read_text()
TREE = ast.parse(SRC)
CLS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "IsaacLabAnymalRoomVecEnv")
MOVE = next(n for n in CLS.body if isinstance(n, ast.FunctionDef) and n.name == "_move_visible_goal")
MOVE_SRC = textwrap.dedent(ast.get_source_segment(SRC, MOVE) or "")
NS = {
    "torch": torch,
    "math": math,
    "ARROW_HEAD_BASE_FRACTION": 0.36,
    "ARROW_HEAD_MID_FRACTION": 0.33,
    "ARROW_HEAD_TIP_FRACTION": 0.31,
}
exec(MOVE_SRC, NS)
move_visible_goal = NS["_move_visible_goal"]


class Dummy:
    _move_visible_goal = move_visible_goal


def run_case(anchor_mode: str, yaw_value: float) -> None:
    obj = Dummy()
    obj._device = torch.device("cpu")
    obj._show_goal_arrow = True
    obj._arrow_target_anchor = anchor_mode
    obj._arrow_length = 1.60
    obj._arrow_head_length = 0.55
    camera_goal = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    camera_offset = 0.510
    axis = torch.tensor([math.cos(yaw_value), math.sin(yaw_value)], dtype=torch.float32)
    base_goal = camera_goal - camera_offset * axis
    obj._goal_xy = camera_goal
    obj._goal_yaw = torch.tensor([yaw_value], dtype=torch.float32)
    obj._goal_base_pose = lambda: (
        base_goal.clone(), torch.tensor([yaw_value], dtype=torch.float32)
    )
    captured = {}

    def capture(self, name, env_ids, local_xy, z, yaw, forward_offset=0.0, lateral_offset=0.0):
        captured[name] = {"xy": local_xy.detach().clone(), "z": float(z), "yaw": yaw.detach().clone()}

    obj._write_marker_pose = types.MethodType(capture, obj)
    obj._move_visible_goal(torch.tensor([0], dtype=torch.long))

    required = {"goal_arrow_anchor", "goal_arrow_shaft", "goal_arrow_head_base", "goal_arrow_head_mid", "goal_arrow_head_tip"}
    assert set(captured) == required
    assert torch.allclose(captured["goal_arrow_anchor"]["xy"][0], base_goal[0], atol=1e-6)
    # Visual target must NOT be the mathematical camera goal anymore.
    assert torch.linalg.vector_norm(captured["goal_arrow_anchor"]["xy"][0] - camera_goal[0]) > 0.50

    expected_tip_offset = obj._arrow_length if anchor_mode == "base" else 0.0
    expected_tip = base_goal[0] + expected_tip_offset * axis
    tip_center = captured["goal_arrow_head_tip"]["xy"][0]
    tip_len = 0.31 * obj._arrow_head_length
    actual_tip = tip_center + 0.5 * tip_len * axis
    assert torch.linalg.vector_norm(actual_tip - expected_tip) < 1e-5, (anchor_mode, yaw_value, actual_tip, expected_tip)

    shaft = captured["goal_arrow_shaft"]["xy"][0]
    forward_projection = float((shaft - base_goal[0]) @ axis)
    if anchor_mode == "tip":
        assert forward_projection < 0.0, forward_projection
    else:
        assert forward_projection > 0.0, forward_projection

    # In base mode the camera target lies exactly 0.510 m along the arrow axis.
    if anchor_mode == "base":
        camera_forward = float((camera_goal[0] - base_goal[0]) @ axis)
        camera_lateral = float((camera_goal[0] - base_goal[0]) @ torch.tensor([-axis[1], axis[0]]))
        assert abs(camera_forward - camera_offset) < 1e-5
        assert abs(camera_lateral) < 1e-5

for mode in ("base", "tip"):
    for yaw in (0.0, math.pi / 2.0, -1.1):
        run_case(mode, yaw)

print("[ARROW SANITY] robot-base anchored arrow geometry passed for base/tip modes")
