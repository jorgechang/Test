#!/usr/bin/env python3
from __future__ import annotations

import ast
import math
import pathlib
import textwrap
import types

import torch

HERE = pathlib.Path(__file__).resolve().parent
ENV = HERE / "payload/envs/isaaclab_anymal_room.py"
SOURCE = ENV.read_text()
TREE = ast.parse(SOURCE)


def extract(name: str, namespace: dict):
    node = next(
        item
        for item in TREE.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    exec(textwrap.dedent(ast.get_source_segment(SOURCE, node) or ""), namespace)
    return namespace[name]


ns = {
    "math": math,
    "Any": object,
    "DEFAULT_CAMERA_OFFSET_X": 0.510,
    "DEFAULT_CAMERA_OFFSET_Y": 0.0,
    "DEFAULT_CAMERA_OFFSET_Z": 0.015,
    "DEFAULT_CAMERA_PITCH_DEG": 0.0,
}
_get = extract("_get", ns)
ns["_get"] = _get
_mul = extract("_quat_mul_wxyz_tuple", ns)
ns["_quat_mul_wxyz_tuple"] = _mul
_mount = extract("_camera_mount_from_config", ns)


def apply_xyzw(q, v):
    q = torch.tensor(q, dtype=torch.float64)
    v = torch.tensor(v, dtype=torch.float64)
    qvec = q[:3]
    qw = q[3]
    t = 2.0 * torch.cross(qvec, v, dim=0)
    return v + qw * t + torch.cross(qvec, t, dim=0)


level = types.SimpleNamespace(
    camera_offset_x=0.510,
    camera_offset_y=0.0,
    camera_offset_z=0.015,
    camera_pitch_deg=0.0,
)
pos, ros_wxyz, gl_xyzw, pitch = _mount(level)
assert pos == (0.510, 0.0, 0.015)
assert pitch == 0.0
assert max(abs(a - b) for a, b in zip(ros_wxyz, (0.5, -0.5, 0.5, -0.5))) < 1e-12
assert max(abs(a - b) for a, b in zip(gl_xyzw, (-0.5, 0.5, 0.5, -0.5))) < 1e-12
forward = apply_xyzw(gl_xyzw, (0.0, 0.0, -1.0))
assert torch.allclose(forward, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12)

pitched = types.SimpleNamespace(
    camera_offset_x=0.510,
    camera_offset_y=0.0,
    camera_offset_z=0.015,
    camera_pitch_deg=15.0,
)
_, _, gl_xyzw_15, _ = _mount(pitched)
forward_15 = apply_xyzw(gl_xyzw_15, (0.0, 0.0, -1.0))
expected = torch.tensor(
    [math.cos(math.radians(15.0)), 0.0, -math.sin(math.radians(15.0))],
    dtype=torch.float64,
)
assert torch.allclose(forward_15, expected, atol=1e-12)

print("[CAMERA MOUNT SANITY] shared render/reward mount passed at 0 and 15 deg pitch")
