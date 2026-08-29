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
CLS = next(
    node
    for node in TREE.body
    if isinstance(node, ast.ClassDef) and node.name == "IsaacLabAnymalRoomVecEnv"
)


def extract_global(name: str, namespace: dict):
    node = next(
        item
        for item in TREE.body
        if isinstance(item, (ast.FunctionDef, ast.Assign))
        and (
            (isinstance(item, ast.FunctionDef) and item.name == name)
            or (
                isinstance(item, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in item.targets)
            )
        )
    )
    source = ast.get_source_segment(SRC, node) or ""
    exec(source, namespace)
    return namespace[name]


BASE_NS = {
    "torch": torch,
    "math": math,
    "ROOM_HALF": 4.0,
    "_as_torch": lambda value: value,
    "_wrap_to_pi": lambda angle: torch.atan2(torch.sin(angle), torch.cos(angle)),
    "PROP_NAMES": ("ball", "cube", "cone", "yellow_block"),
    "GOAL_SAMPLING_HEADING_BINS": 8,
    "OBJECT_XY": (
        (1.60, 1.70),
        (-1.80, 1.45),
        (1.65, -1.70),
        (-1.70, -1.80),
    ),
}
for name in (
    "TASK_RANDOM_STREAM_SALTS",
    "_TASK_SEED_MASK",
    "_TASK_TORCH_SEED_MASK",
):
    extract_global(name, BASE_NS)
for name in ("_mixed_task_seed", "_coprime_stride"):
    extract_global(name, BASE_NS)


def extract_method(name: str):
    node = next(
        item
        for item in CLS.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    source = textwrap.dedent(ast.get_source_segment(SRC, node) or "")
    namespace = dict(BASE_NS)
    exec(source, namespace)
    return namespace[name]


class Dummy:
    _task_random = extract_method("_task_random")
    _task_stratum_index = extract_method("_task_stratum_index")
    _advance_task_randomization = extract_method("_advance_task_randomization")
    _sample_object_layout = extract_method("_sample_object_layout")
    _sample_free_xy = extract_method("_sample_free_xy")
    _sample_goal_for_ids = extract_method("_sample_goal_for_ids")


def make_dummy(seed: int, num_envs: int, randomize: bool = True) -> Dummy:
    obj = Dummy()
    obj._device = torch.device("cpu")
    obj._num_envs = num_envs
    obj._task_seed = seed
    obj._task_episode_count = torch.zeros(num_envs, dtype=torch.int64)
    obj._randomize_objects = randomize
    obj._object_wall_clearance = 0.45
    obj._object_min_clearance = 0.70
    obj._object_sampling_attempts = 512
    obj._start_sampling_attempts = 512
    obj._balanced_randomization = True
    obj._start_grid_bins = 4
    obj._start_yaw_bins = 8
    obj._start_stratum_count = 4 * 4 * 8
    obj._start_stratum_stride = BASE_NS["_coprime_stride"](
        obj._start_stratum_count, 53
    )
    obj._goal_sampling_attempts = 1024
    obj._goal_stratified_attempts = 64
    obj._goal_distance_bins = 4
    obj._goal_bearing_bins = 8
    obj._goal_yaw_bins = 8
    obj._goal_stratum_count = 4 * 8 * 8
    obj._goal_stratum_stride = BASE_NS["_coprime_stride"](
        obj._goal_stratum_count, 73
    )
    obj._object_guard_radii = torch.tensor(
        [
            0.30,
            math.sqrt(2.0) * 0.58 / 2.0,
            0.30,
            math.sqrt((0.58 / 2.0) ** 2 + (0.74 / 2.0) ** 2),
        ],
        dtype=torch.float32,
    )
    return obj


num_envs = 512
env_ids = torch.arange(num_envs, dtype=torch.long)
obj = make_dummy(seed=123, num_envs=num_envs)
layout = obj._sample_object_layout(env_ids)
assert layout.shape == (num_envs, 4, 2)

# Every object surface respects wall and pairwise surface clearances.
for index, radius in enumerate(obj._object_guard_radii):
    allowed = 4.0 - obj._object_wall_clearance - float(radius)
    assert bool((layout[:, index].abs() <= allowed + 1e-6).all())
for i in range(4):
    for j in range(i):
        distance = torch.linalg.vector_norm(layout[:, i] - layout[:, j], dim=-1)
        required = (
            float(obj._object_guard_radii[i])
            + float(obj._object_guard_radii[j])
            + obj._object_min_clearance
        )
        assert bool((distance >= required - 1e-6).all()), (
            i,
            j,
            distance.min(),
            required,
        )

# Different environments receive different layouts and randomized placement order.
unique_first_object = torch.unique(layout[:, 0].round(decimals=4), dim=0)
assert unique_first_object.shape[0] > int(0.95 * num_envs)
order_keys = obj._task_random(env_ids, (4,), "object_order")
first_counts = torch.bincount(torch.argsort(order_keys, dim=1)[:, 0], minlength=4)
assert bool((first_counts > 0.18 * num_envs).all()), first_counts
assert bool((first_counts < 0.32 * num_envs).all()), first_counts

# Same (seed, env, episode) is reproducible. Advancing an episode changes it.
repeat = make_dummy(seed=123, num_envs=num_envs)._sample_object_layout(env_ids)
assert torch.allclose(layout, repeat)
obj._advance_task_randomization(env_ids)
next_layout = obj._sample_object_layout(env_ids)
assert not torch.allclose(layout, next_layout)
different_seed = make_dummy(seed=124, num_envs=num_envs)._sample_object_layout(env_ids)
assert not torch.allclose(layout, different_seed)

# Reset grouping/order does not couple environments: subset samples match batch samples.
subset = torch.tensor([503, 7, 128, 2, 411], dtype=torch.long)
independent = make_dummy(seed=123, num_envs=num_envs)._sample_object_layout(subset)
assert torch.allclose(independent, layout[subset])

# Balanced stratum cycles cover every joint bin exactly once before repeating.
cycle = make_dummy(seed=321, num_envs=1)
seen_start = []
for episode in range(cycle._start_stratum_count):
    cycle._task_episode_count[0] = episode
    seen_start.append(
        int(
            cycle._task_stratum_index(
                torch.tensor([0]),
                cycle._start_stratum_count,
                cycle._start_stratum_stride,
                "start_phase",
            )[0]
        )
    )
assert len(set(seen_start)) == cycle._start_stratum_count
seen_goal = []
for episode in range(cycle._goal_stratum_count):
    cycle._task_episode_count[0] = episode
    seen_goal.append(
        int(
            cycle._task_stratum_index(
                torch.tensor([0]),
                cycle._goal_stratum_count,
                cycle._goal_stratum_stride,
                "goal_phase",
            )[0]
        )
    )
assert len(set(seen_goal)) == cycle._goal_stratum_count

# Generic free-point sampling respects the live per-environment layouts.
obj = make_dummy(seed=123, num_envs=num_envs)
obj._object_xy = layout
starts = obj._sample_free_xy(
    env_ids,
    wall_clearance=1.20,
    object_clearance=1.20,
)
assert starts.shape == (num_envs, 2)
assert bool((starts.abs() <= 2.80 + 1e-6).all())
start_object_distance = torch.linalg.vector_norm(
    starts[:, None, :] - layout, dim=-1
)
assert bool((start_object_distance.amin(dim=1) >= 1.20 - 1e-6).all())

# Fixed-layout mode still gives the canonical fallback exactly.
fixed = make_dummy(seed=999, num_envs=3, randomize=False)._sample_object_layout(
    torch.arange(3, dtype=torch.long)
)
canonical = torch.tensor(
    [
        [1.60, 1.70],
        [-1.80, 1.45],
        [1.65, -1.70],
        [-1.70, -1.80],
    ],
    dtype=torch.float32,
)
assert torch.allclose(fixed, canonical.unsqueeze(0).repeat(3, 1, 1))

# Camera-goal sampler uses live layouts and satisfies all camera/base/arrow constraints.
goal_envs = 256
goal_ids = torch.arange(goal_envs, dtype=torch.long)
goal_obj = make_dummy(seed=321, num_envs=goal_envs)
goal_layout = goal_obj._sample_object_layout(goal_ids)
goal_obj._object_xy = goal_layout
base_starts = goal_obj._sample_free_xy(
    goal_ids,
    wall_clearance=1.20,
    object_clearance=1.20,
)
base_yaw_u = goal_obj._task_random(goal_ids, (1,), "start_yaw")[:, 0]
base_yaw = (base_yaw_u * 2.0 - 1.0) * math.pi
camera_offset = 0.510
camera_xy = base_starts + torch.stack(
    [camera_offset * torch.cos(base_yaw), camera_offset * torch.sin(base_yaw)],
    dim=-1,
)
camera_pos = torch.cat(
    [camera_xy, torch.full((goal_envs, 1), 0.615)], dim=-1
)

goal_obj._unwrapped_env = types.SimpleNamespace(
    scene=types.SimpleNamespace(env_origins=torch.zeros((goal_envs, 3)))
)
goal_obj._camera_pose_world_from_body = lambda: (
    camera_pos,
    torch.zeros((goal_envs, 4)),
    base_yaw,
)
goal_obj._robot_state = lambda: (
    base_starts,
    base_yaw,
    torch.full((goal_envs,), 0.60),
)
goal_obj._camera_rel_xy_planar = torch.tensor([camera_offset, 0.0])
goal_obj._camera_yaw_offset = 0.0
goal_obj._goal_min_distance = 0.50
goal_obj._goal_max_distance = 4.00
goal_obj._goal_bearing_max = math.pi
goal_obj._goal_relative_yaw_max = math.pi
goal_obj._goal_wall_clearance = 0.85
goal_obj._base_goal_wall_clearance = 1.15
goal_obj._goal_object_clearance = 0.85
goal_obj._base_goal_object_clearance = 1.20
goal_obj._arrow_target_anchor = "base"
goal_obj._arrow_length = 1.60
goal_obj._arrow_head_base_width = 0.68
goal_obj._arrow_wall_clearance = 0.10
goal_obj._arrow_object_clearance = 0.10
goal_obj._arrow_start_clearance = 0.35
goal_obj._goal_xy = torch.zeros((goal_envs, 2))
goal_obj._goal_yaw = torch.zeros(goal_envs)
goal_obj._goal_sample_attempts = torch.zeros(goal_envs, dtype=torch.int64)
goal_obj._goal_sample_reject_wall = torch.zeros(goal_envs, dtype=torch.int64)
goal_obj._goal_sample_reject_object = torch.zeros(goal_envs, dtype=torch.int64)
goal_obj._goal_sample_reject_start = torch.zeros(goal_envs, dtype=torch.int64)
goal_obj._goal_arrow_start_distance = torch.zeros(goal_envs)
goal_obj._goal_heading_attempts = torch.zeros((goal_envs, 8), dtype=torch.int64)
goal_obj._goal_heading_rejections = torch.zeros((goal_envs, 8), dtype=torch.int64)
goal_obj._goal_heading_accepts = torch.zeros((goal_envs, 8), dtype=torch.int64)
goal_obj._move_visible_goal = lambda env_ids: None
goal_obj._redraw_target_debug = lambda: None

goal_obj._sample_goal_for_ids(goal_ids)
goal_xy = goal_obj._goal_xy
relative_goal = torch.linalg.vector_norm(goal_xy - camera_xy, dim=-1)
assert bool((relative_goal >= 0.50 - 1e-5).all())
assert bool((relative_goal <= 4.00 + 1e-5).all())
assert bool((goal_xy.abs() <= 3.15 + 1e-5).all())
body_yaw = goal_obj._goal_yaw
base_goal_xy = goal_xy - torch.stack(
    [camera_offset * torch.cos(body_yaw), camera_offset * torch.sin(body_yaw)],
    dim=-1,
)
assert bool((base_goal_xy.abs() <= 2.85 + 1e-5).all())
camera_object_distance = torch.linalg.vector_norm(
    goal_xy[:, None, :] - goal_layout, dim=-1
)
base_object_distance = torch.linalg.vector_norm(
    base_goal_xy[:, None, :] - goal_layout, dim=-1
)
assert bool((camera_object_distance.amin(dim=1) >= 0.85 - 1e-5).all())
assert bool((base_object_distance.amin(dim=1) >= 1.20 - 1e-5).all())

arrow_axis = torch.stack([torch.cos(body_yaw), torch.sin(body_yaw)], dim=-1)
arrow_tail = base_goal_xy
arrow_tip = base_goal_xy + goal_obj._arrow_length * arrow_axis
arrow_center = 0.5 * (arrow_tail + arrow_tip)
half_forward = 0.5 * goal_obj._arrow_length
half_lateral = 0.5 * goal_obj._arrow_head_base_width
arrow_extent = torch.stack(
    [
        arrow_axis[:, 0].abs() * half_forward
        + arrow_axis[:, 1].abs() * half_lateral,
        arrow_axis[:, 1].abs() * half_forward
        + arrow_axis[:, 0].abs() * half_lateral,
    ],
    dim=-1,
)
assert bool(
    (
        arrow_center.abs() + arrow_extent
        <= 4.0 - goal_obj._arrow_wall_clearance + 1e-5
    ).all()
)
object_from_tail = goal_layout - arrow_tail[:, None, :]
along_arrow = (object_from_tail * arrow_axis[:, None, :]).sum(dim=-1)
along_arrow = along_arrow.clamp(0.0, goal_obj._arrow_length)
closest = arrow_tail[:, None, :] + along_arrow[..., None] * arrow_axis[:, None, :]
arrow_object_dist = torch.linalg.vector_norm(goal_layout - closest, dim=-1)
arrow_required = (
    goal_obj._object_guard_radii[None, :]
    + half_lateral
    + goal_obj._arrow_object_clearance
)
assert bool((arrow_object_dist >= arrow_required - 1e-5).all())

start_from_tail = base_starts - arrow_tail
along_start = (start_from_tail * arrow_axis).sum(dim=-1).clamp(
    0.0, goal_obj._arrow_length
)
closest_start = arrow_tail + along_start[:, None] * arrow_axis
start_centerline_dist = torch.linalg.vector_norm(base_starts - closest_start, dim=-1)
start_required = half_lateral + goal_obj._arrow_start_clearance
assert bool((start_centerline_dist >= start_required - 1e-5).all())
assert torch.allclose(
    goal_obj._goal_arrow_start_distance, start_centerline_dist, atol=1e-5
)

assert bool((goal_obj._goal_sample_attempts >= 1).all())
assert bool((goal_obj._goal_heading_accepts.sum(dim=1) == 1).all())
assert bool(
    (
        goal_obj._goal_sample_reject_wall
        + goal_obj._goal_sample_reject_object
        + goal_obj._goal_sample_reject_start
        == goal_obj._goal_sample_attempts - 1
    ).all()
)
assert bool(
    (
        goal_obj._goal_heading_rejections.sum(dim=1)
        == goal_obj._goal_sample_attempts - 1
    ).all()
)
assert bool(
    (goal_obj._goal_heading_attempts.sum(dim=1) == goal_obj._goal_sample_attempts).all()
)

print(
    "[RANDOMIZATION SANITY] independent streams, balanced cycles, 512 layouts, "
    "live starts, and 256 constrained goals passed"
)
