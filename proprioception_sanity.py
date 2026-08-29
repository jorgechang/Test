#!/usr/bin/env python3
from __future__ import annotations

import ast
import pathlib
import textwrap
import types

import torch

HERE = pathlib.Path(__file__).resolve().parent
ENV_PATH = HERE / "payload/envs/isaaclab_anymal_room.py"
SRC = ENV_PATH.read_text()
TREE = ast.parse(SRC)
CLS = next(
    node
    for node in TREE.body
    if isinstance(node, ast.ClassDef) and node.name == "IsaacLabAnymalRoomVecEnv"
)


def extract_global(name: str, namespace: dict):
    node = next(
        item
        for item in TREE.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    source = ast.get_source_segment(SRC, node) or ""
    exec(source, namespace)
    return namespace[name]


namespace = {"torch": torch}
namespace["_as_torch"] = extract_global("_as_torch", namespace)
namespace["_quat_normalize_xyzw"] = extract_global(
    "_quat_normalize_xyzw", namespace
)
namespace["_quat_apply_xyzw"] = extract_global("_quat_apply_xyzw", namespace)
method_node = next(
    item
    for item in CLS.body
    if isinstance(item, ast.FunctionDef)
    and item.name == "_proprioceptive_observation"
)
exec(textwrap.dedent(ast.get_source_segment(SRC, method_node) or ""), namespace)
proprio_method = namespace["_proprioceptive_observation"]


class Dummy:
    _proprioceptive_observation = proprio_method


def make_dummy(data) -> Dummy:
    obj = Dummy()
    obj._device = torch.device("cpu")
    obj._num_envs = 2
    obj._num_joints = 12
    obj._proprio_dim = 33
    obj._proprio_linear_velocity_scale = 2.0
    obj._proprio_angular_velocity_scale = 0.25
    obj._proprio_joint_position_scale = 1.0
    obj._proprio_joint_velocity_scale = 0.05
    obj._proprio_clip = 5.0
    obj._unwrapped_env = types.SimpleNamespace(
        scene={"robot": types.SimpleNamespace(data=data)}
    )
    return obj


joint_default = torch.linspace(-0.6, 0.6, 12).repeat(2, 1)
joint_delta = torch.stack(
    [torch.linspace(-0.2, 0.2, 12), torch.linspace(0.3, -0.3, 12)]
)
joint_pos = joint_default + joint_delta
joint_vel = torch.stack(
    [torch.linspace(-10.0, 10.0, 12), torch.linspace(5.0, -5.0, 12)]
)
lin_b = torch.tensor([[0.5, -0.25, 0.1], [-0.4, 0.2, -0.05]])
ang_b = torch.tensor([[1.0, -2.0, 0.5], [-1.5, 0.25, 2.0]])
gravity_b = torch.tensor([[0.0, 0.0, -1.0], [0.1, -0.2, -0.97]])

data = types.SimpleNamespace(
    root_lin_vel_b=lin_b,
    root_ang_vel_b=ang_b,
    projected_gravity_b=gravity_b,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    default_joint_pos=joint_default,
    # Deliberately different global poses: they must not enter the vector.
    root_state_w=torch.tensor(
        [
            [100.0, -50.0, 3.0, 0.0, 0.0, 0.0, 1.0, *([0.0] * 6)],
            [-80.0, 75.0, 9.0, 0.0, 0.0, 0.0, 1.0, *([0.0] * 6)],
        ]
    ),
)
obj = make_dummy(data)
proprio = obj._proprioceptive_observation()
assert proprio.shape == (2, 33)
expected = torch.cat(
    [
        lin_b * 2.0,
        ang_b * 0.25,
        gravity_b,
        joint_delta,
        joint_vel * 0.05,
    ],
    dim=-1,
).clamp(-5.0, 5.0)
assert torch.allclose(proprio, expected, atol=1e-6)

# Changing only global position/yaw leaves direct body-frame proprio unchanged.
data.root_state_w[:, :7] = torch.tensor(
    [[-999.0, 400.0, 1.0, 0.0, 0.0, 1.0, 0.0], [333.0, -222.0, 2.0, 0.0, 0.0, -1.0, 0.0]]
)
assert torch.allclose(obj._proprioceptive_observation(), expected, atol=1e-6)

# Fallback path: identity body quaternion converts world velocities/gravity directly.
root_state = torch.zeros((2, 13), dtype=torch.float32)
root_state[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
root_state[:, 7:10] = lin_b
root_state[:, 10:13] = ang_b
fallback_data = types.SimpleNamespace(
    root_state_w=root_state,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    default_joint_pos=joint_default,
)
fallback = make_dummy(fallback_data)._proprioceptive_observation()
expected_fallback = torch.cat(
    [
        lin_b * 2.0,
        ang_b * 0.25,
        torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        joint_delta,
        joint_vel * 0.05,
    ],
    dim=-1,
).clamp(-5.0, 5.0)
assert torch.allclose(fallback, expected_fallback, atol=1e-6)

# Non-finite values are sanitized and extreme values are clipped.
fallback_data.joint_vel[0, 0] = float("nan")
fallback_data.joint_vel[0, 1] = 1e6
safe = make_dummy(fallback_data)._proprioceptive_observation()
assert torch.isfinite(safe).all()
assert float(safe.abs().max()) <= 5.0

print("[PROPRIOCEPTION SANITY] 33-D routing, normalization, no global-pose leak, fallback, and clipping passed")
