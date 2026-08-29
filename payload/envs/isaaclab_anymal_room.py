"""ANYmal-C v13.45: exact-v13.20 reward with visible-arrow pose navigation.

This file is derived directly from the user's v13.14 ANYmal camera-pose task.
The scalar reward and control geometry are intentionally identical to v13.20/v13.14.

The v13.20 camera-pose reward is retained, with five scene/input changes:
1. The sampled target camera pose is rendered into RGB as a bright green floor
   arrow at the exact implied robot-base goal pose. Reaching that base pose
   places the fixed camera at the original v13.14 target camera pose.
2. The 4-D relative target vector
   ``[dx_camera, dy_camera, cos(delta_yaw), sin(delta_yaw)]`` is not an
   encoder/policy input. It remains in the schema only for optional diagnostics;
   the auxiliary decoder head is structurally disabled by default.
3. The clear room interior is enlarged from 7 x 7 m to 8 x 8 m.
4. Task randomization uses independent deterministic streams for every environment
   and episode, randomized object placement order, and balanced continuous strata
   for start pose and target distance/bearing/yaw. Robot starts and camera targets
   avoid the live object poses.
5. Optional 33-D robot-only proprioception can be routed through Dreamer's MLP
   encoder/decoder without exposing global pose or the target vector.

The original v13.20 rho/alpha/beta/final-yaw geometry, dense state reward,
success criterion, actions, safety, and episode semantics are preserved.
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict


# -----------------------------------------------------------------------------
# ANYmal-C-scaled room/object geometry.
#
# Jetbot keeps the exact v12 3.2 m x 3.2 m arena in isaaclab_room.py.  This
# backend is a morphology-scaled version: room linear dimensions are ~1.875x
# larger and the landmarks are enlarged enough to remain visually salient from
# the longer navigation distances.
# -----------------------------------------------------------------------------
ROOM_HALF = 4.00             # 8.0 m x 8.0 m clear interior
WALL_H = 1.35
WALL_T = 0.40
WALL_CONTACT_OFFSET = 0.08
WALL_REST_OFFSET = 0.01

# Canonical fallback layout used only when object randomization is disabled.
BALL_XY = (1.60, 1.70)
CUBE_XY = (-1.80, 1.45)
CONE_XY = (1.65, -1.70)
YELLOW_BLOCK_XY = (-1.70, -1.80)
OBJECT_XY = (BALL_XY, CUBE_XY, CONE_XY, YELLOW_BLOCK_XY)

# Enlarged for ANYmal morphology and 1--3.5 m viewing distances.
BALL_RADIUS = 0.30
CUBE_EDGE = 0.58
CONE_RADIUS = 0.30
CONE_HEIGHT = 0.72
YELLOW_BLOCK_SIZE = (0.58, 0.74, 0.58)

# Conservative horizontal obstacle radii used by the safety guard. Cuboids use
# their circumscribed radius so the guard cannot miss a corner.
OBJECT_GUARD_RADII = (
    BALL_RADIUS,
    math.sqrt(2.0) * CUBE_EDGE / 2.0,
    CONE_RADIUS,
    math.sqrt((YELLOW_BLOCK_SIZE[0] / 2.0) ** 2 + (YELLOW_BLOCK_SIZE[1] / 2.0) ** 2),
)
OBJECT_Z = (
    BALL_RADIUS,
    CUBE_EDGE / 2.0,
    CONE_HEIGHT / 2.0,
    YELLOW_BLOCK_SIZE[2] / 2.0,
)

# ANYmal-C defaults to a base height of about 0.6 m. Camera position, pitch,
# rendering quaternion, and body-composed reward quaternion are all derived from
# the same config-driven mount helper below. Positive pitch angles point downward.
ANYMAL_ROOT_Z = 0.60
DEFAULT_CAMERA_OFFSET_X = 0.510
DEFAULT_CAMERA_OFFSET_Y = 0.0
DEFAULT_CAMERA_OFFSET_Z = 0.015
DEFAULT_CAMERA_PITCH_DEG = 0.0
CAMERA_LOCAL_FORWARD = (0.0, 0.0, -1.0)  # OpenGL optical forward used by pose math.
GOAL_SAMPLING_HEADING_BINS = 8

PROP_NAMES = ("ball", "cube", "cone", "yellow_block")
ARROW_HEAD_BASE_FRACTION = 0.36
ARROW_HEAD_MID_FRACTION = 0.33
ARROW_HEAD_TIP_FRACTION = 1.0 - ARROW_HEAD_BASE_FRACTION - ARROW_HEAD_MID_FRACTION


def _get(config: Any, name: str, default: Any) -> Any:
    value = getattr(config, name, default)
    return default if value is None else value


# Fixed salts make task randomization reproducible across Python processes.  Never
# use Python's built-in hash() here because hash randomization changes each launch.
TASK_RANDOM_STREAM_SALTS = {
    "object_order": 0x243F6A8885A308D3,
    "object_xy": 0x13198A2E03707344,
    "object_yaw": 0xA4093822299F31D0,
    "start_xy": 0x082EFA98EC4E6C89,
    "start_yaw": 0x452821E638D01377,
    "goal": 0xBE5466CF34E90C6C,
    "free_xy": 0xC0AC29B7C97C50DD,
    "object_yaw_phase": 0x3F84D5B5B5470917,
    "start_phase": 0x9216D5D98979FB1B,
    "goal_phase": 0xD1310BA698DFB5AC,
}
_TASK_SEED_MASK = (1 << 64) - 1
_TASK_TORCH_SEED_MASK = (1 << 63) - 1


def _mixed_task_seed(base_seed: int, env_id: int, episode_index: int, stream: str) -> int:
    """SplitMix64 seed for one environment, episode, and randomization stream."""
    if stream not in TASK_RANDOM_STREAM_SALTS:
        raise KeyError(f"Unknown task randomization stream: {stream}")
    x = int(base_seed) & _TASK_SEED_MASK
    x ^= (int(env_id) + 1) * 0x9E3779B97F4A7C15
    x ^= (int(episode_index) + 1) * 0xBF58476D1CE4E5B9
    x ^= TASK_RANDOM_STREAM_SALTS[stream]
    x &= _TASK_SEED_MASK
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _TASK_SEED_MASK
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _TASK_SEED_MASK
    x ^= x >> 31
    return x & _TASK_TORCH_SEED_MASK


def _coprime_stride(modulus: int, preferred: int) -> int:
    """Return a positive stride coprime to modulus for full-cycle strata."""
    if modulus <= 1:
        return 1
    stride = int(preferred) % modulus
    if stride <= 0:
        stride = 1
    while math.gcd(stride, modulus) != 1:
        stride = (stride + 1) % modulus
        if stride == 0:
            stride = 1
    return stride


def _quat_mul_wxyz_tuple(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product for scalar-first quaternions, kept dependency-free."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _camera_mount_from_config(
    config: Any,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    float,
]:
    """Return one authoritative camera mount for rendering and reward geometry.

    Returns ``(position_xyz, ros_wxyz, opengl_xyzw, pitch_deg)``. Isaac Lab's
    ``OffsetCfg`` consumes the ROS quaternion in WXYZ ordering. The body-composed
    camera pose uses the equivalent OpenGL camera quaternion in XYZW ordering.
    At zero pitch these exactly recover the validated v13.20 mount.
    """
    position = (
        float(_get(config, "camera_offset_x", DEFAULT_CAMERA_OFFSET_X)),
        float(_get(config, "camera_offset_y", DEFAULT_CAMERA_OFFSET_Y)),
        float(_get(config, "camera_offset_z", DEFAULT_CAMERA_OFFSET_Z)),
    )
    pitch_deg = float(_get(config, "camera_pitch_deg", DEFAULT_CAMERA_PITCH_DEG))
    if not -60.0 <= pitch_deg <= 60.0:
        raise ValueError("camera_pitch_deg must be within [-60, 60]")
    pitch = math.radians(pitch_deg)
    pitch_wxyz = (math.cos(0.5 * pitch), 0.0, math.sin(0.5 * pitch), 0.0)

    # Level camera frames validated in v13.20.
    level_ros_wxyz = (0.5, -0.5, 0.5, -0.5)
    level_opengl_wxyz = (-0.5, -0.5, 0.5, 0.5)
    ros_wxyz = _quat_mul_wxyz_tuple(pitch_wxyz, level_ros_wxyz)
    opengl_wxyz = _quat_mul_wxyz_tuple(pitch_wxyz, level_opengl_wxyz)
    opengl_xyzw = (
        opengl_wxyz[1],
        opengl_wxyz[2],
        opengl_wxyz[3],
        opengl_wxyz[0],
    )
    return position, ros_wxyz, opengl_xyzw, pitch_deg


def _as_torch(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    try:
        torch_view = getattr(value, "torch", None)
        if torch_view is not None:
            return torch_view() if callable(torch_view) else torch_view
    except Exception:
        pass
    try:
        import warp as wp

        return wp.to_torch(value)
    except Exception as exc:  # pragma: no cover - depends on Isaac backend
        raise TypeError(f"Cannot convert Isaac buffer {type(value)!r} to torch") from exc


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _quat_normalize_xyzw(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def _quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def _quat_apply_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qvec = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * torch.cross(qvec, v, dim=-1)
    return v + qw * t + torch.cross(qvec, t, dim=-1)


def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    """Return Isaac Lab 3.0 root quaternions in XYZW ordering."""
    quat = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=torch.float32)
    quat[:, 2] = torch.sin(0.5 * yaw)
    quat[:, 3] = torch.cos(0.5 * yaw)
    return quat


def _resize_images(images: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    source_dtype = images.dtype
    x = images.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
    x = x.permute(0, 2, 3, 1)
    if source_dtype == torch.uint8:
        return x.round().clamp(0, 255).to(torch.uint8)
    return x.to(source_dtype)


def _asset_cfg(name: str, spawn: Any, pos: tuple[float, float, float]) -> Any:
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=spawn,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _wall_cfg(name: str, pos: tuple[float, float, float], size: tuple[float, float, float]) -> Any:
    import isaaclab.sim as sim_utils

    return _asset_cfg(
        name,
        sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=WALL_CONTACT_OFFSET,
                rest_offset=WALL_REST_OFFSET,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
        ),
        pos,
    )


def _static_obstacle_props() -> dict[str, Any]:
    """Physics properties for kinematic, collidable room landmarks.

    Isaac Lab documents ``kinematic_enabled=True`` as the way to make a rigid
    object static.  The collider remains active, while gravity and forces can
    no longer move the landmark.
    """
    import isaaclab.sim as sim_utils

    return {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        "collision_props": sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.06,
            rest_offset=0.005,
        ),
        "physics_material": sim_utils.RigidBodyMaterialCfg(
            static_friction=0.9,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
    }

def _marker_cfg(
    name: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> Any:
    """Create a collision-free kinematic marker from standard Isaac Lab assets."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def build_anymal_navigation_cfg(config: Any) -> Any:
    """Build official ANYmal navigation plus the exact v12 room assets."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab_tasks.manager_based.navigation.config.anymal_c.navigation_env_cfg import (
        NavigationEnvCfg,
    )

    # Isaac Lab 3.0's official ANYmal configs contain backend PresetCfg
    # objects (physics backend, contact sensor, ANYmal armature, ...).
    # The normal Isaac Lab task launcher resolves these through its own Hydra
    # layer.  r2dreamer instantiates this config directly, so we must resolve
    # them explicitly before creating ManagerBasedRLEnv.
    from isaaclab_tasks.utils.hydra import resolve_presets

    cfg = resolve_presets(NavigationEnvCfg(), selected=("physx",))
    cfg.scene.num_envs = int(config.env_num)
    cfg.scene.env_spacing = 10.0
    cfg.seed = int(config.seed)
    cfg.sim.device = str(config.device)

    # Official ANYmal navigation normally exposes one high-level action every
    # 40 physics steps. The v12 Jetbot acts at 30 Hz. Eight steps at the
    # official 0.005 s physics timestep gives a nearby 25 Hz navigation rate,
    # while the pretrained low-level policy still updates every four steps.
    agent_decimation = int(_get(config, "agent_decimation", 8))
    low_level_decimation = int(cfg.actions.pre_trained_policy_action.low_level_decimation)
    if agent_decimation < low_level_decimation or agent_decimation % low_level_decimation:
        raise ValueError(
            "agent_decimation must be a positive multiple of the pretrained "
            f"policy low_level_decimation={low_level_decimation}; got {agent_decimation}."
        )
    cfg.decimation = agent_decimation
    cfg.sim.render_interval = agent_decimation
    cfg.sim.render = sim_utils.RenderCfg(
        antialiasing_mode="DLAA",
        enable_dl_denoiser=True,
    )
    # Extra collision robustness for a large articulated robot near static walls.
    # The thicker walls are the primary fix; CCD is an additional safeguard.
    if hasattr(cfg.sim, "physx") and hasattr(cfg.sim.physx, "enable_ccd"):
        cfg.sim.physx.enable_ccd = True

    # CCD must also be enabled on the moving rigid bodies.  Keep these
    # assignments guarded because Isaac Lab asset configs changed names across
    # releases.  Increasing solver position iterations makes wall/obstacle
    # contacts harder to violate without changing the low-level policy.
    robot_spawn = cfg.scene.robot.spawn
    robot_rigid_props = getattr(robot_spawn, "rigid_props", None)
    if robot_rigid_props is not None:
        if hasattr(robot_rigid_props, "enable_ccd"):
            robot_rigid_props.enable_ccd = True
        if hasattr(robot_rigid_props, "max_depenetration_velocity"):
            robot_rigid_props.max_depenetration_velocity = 1.0
    robot_articulation_props = getattr(robot_spawn, "articulation_props", None)
    if robot_articulation_props is not None:
        if hasattr(robot_articulation_props, "solver_position_iteration_count"):
            robot_articulation_props.solver_position_iteration_count = max(
                int(robot_articulation_props.solver_position_iteration_count or 0), 8
            )
        if hasattr(robot_articulation_props, "solver_velocity_iteration_count"):
            robot_articulation_props.solver_velocity_iteration_count = max(
                int(robot_articulation_props.solver_velocity_iteration_count or 0), 1
            )

    # The external adapter owns task resets, reward, success, and time limits.
    # Native navigation termination must not auto-reset before Dreamer records
    # the transition. Root/joint/prop states are explicitly reset below.
    cfg.terminations.time_out = None
    cfg.terminations.base_contact = None
    cfg.events.reset_base = None
    cfg.episode_length_s = 3600.0
    cfg.commands.pose_command.debug_vis = False
    cfg.commands.pose_command.resampling_time_range = (3600.0, 3600.0)
    cfg.observations.policy.enable_corruption = False
    cfg.actions.pre_trained_policy_action.debug_vis = bool(
        _get(config, "anymal_velocity_debug", False)
    )

    image_h, image_w = map(int, config.size)
    camera_pos_xyz, camera_ros_wxyz, _, camera_pitch_deg = _camera_mount_from_config(config)
    camera_hfov_deg = float(_get(config, "camera_horizontal_fov_deg", 90.0))
    camera_aperture = float(_get(config, "camera_horizontal_aperture", 20.955))
    if not 1.0 < camera_hfov_deg < 179.0:
        raise ValueError("camera_horizontal_fov_deg must be within (1, 179)")
    if camera_aperture <= 0.0:
        raise ValueError("camera_horizontal_aperture must be > 0")
    camera_focal_length = camera_aperture / (
        2.0 * math.tan(math.radians(0.5 * camera_hfov_deg))
    )
    cfg.scene.head_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=camera_pos_xyz,
            rot=camera_ros_wxyz,
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            # v13.45 defaults: 128x128, 90-degree HFOV, level camera.
            focal_length=camera_focal_length,
            horizontal_aperture=camera_aperture,
            clipping_range=(0.05, 20.0),
        ),
        data_types=["rgb"],
        width=image_w,
        height=image_h,
        # The target is defined in the pose of this exact rendered sensor.
        # Keep the live pose buffer current even during training.
        update_latest_camera_pose=True,
    )

    # ANYmal-specific 7 m clear interior. Walls are centered *outside* the
    # requested interior boundary, and their lengths overlap at the corners.
    # This avoids both corner gaps and the very thin-collider penetration seen
    # in the previous 6 m room.
    wall_center = ROOM_HALF + 0.5 * WALL_T
    wall_span = 2.0 * (ROOM_HALF + WALL_T)
    cfg.scene.wall_n = _wall_cfg(
        "WallN", (0.0, wall_center, WALL_H / 2), (wall_span, WALL_T, WALL_H)
    )
    cfg.scene.wall_s = _wall_cfg(
        "WallS", (0.0, -wall_center, WALL_H / 2), (wall_span, WALL_T, WALL_H)
    )
    cfg.scene.wall_e = _wall_cfg(
        "WallE", (wall_center, 0.0, WALL_H / 2), (WALL_T, wall_span, WALL_H)
    )
    cfg.scene.wall_w = _wall_cfg(
        "WallW", (-wall_center, 0.0, WALL_H / 2), (WALL_T, wall_span, WALL_H)
    )

    # Scaled v12 landmarks. They are kinematic colliders repositioned at resets.
    cfg.scene.ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
            **_static_obstacle_props(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*BALL_XY, BALL_RADIUS)),
    )
    cfg.scene.cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_EDGE, CUBE_EDGE, CUBE_EDGE),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.9)),
            **_static_obstacle_props(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*CUBE_XY, CUBE_EDGE / 2.0)),
    )
    cfg.scene.cone = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cone",
        spawn=sim_utils.ConeCfg(
            radius=CONE_RADIUS,
            height=CONE_HEIGHT,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.3)),
            **_static_obstacle_props(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*CONE_XY, CONE_HEIGHT / 2.0)),
    )
    cfg.scene.yellow_block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/YellowBlock",
        spawn=sim_utils.CuboidCfg(
            size=YELLOW_BLOCK_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.75, 0.10)),
            **_static_obstacle_props(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(*YELLOW_BLOCK_XY, YELLOW_BLOCK_SIZE[2] / 2.0)
        ),
    )

    # Visible green floor arrow. These are collision-free kinematic USD assets,
    # so they are rendered by TiledCamera but never affect ANYmal physics.
    # _goal_xy/_goal_yaw remain the exact v13.14 mathematical camera target.
    arrow_green = (0.04, 0.95, 0.08)
    arrow_length = float(_get(config, "arrow_length", 1.60))
    arrow_head_length = float(_get(config, "arrow_head_length", 0.55))
    arrow_shaft_width = float(_get(config, "arrow_shaft_width", 0.26))
    arrow_head_base_width = float(_get(config, "arrow_head_base_width", 0.68))
    arrow_head_mid_width = float(_get(config, "arrow_head_mid_width", 0.44))
    arrow_head_tip_width = float(_get(config, "arrow_head_tip_width", 0.22))
    if arrow_length <= 0.0 or not 0.0 < arrow_head_length < arrow_length:
        raise ValueError("arrow dimensions require 0 < head length < total length")
    if not (
        arrow_head_base_width >= arrow_head_mid_width >= arrow_head_tip_width > 0.0
    ):
        raise ValueError("arrow head widths must taper from base to tip")
    if arrow_shaft_width <= 0.0:
        raise ValueError("arrow_shaft_width must be > 0")
    arrow_shaft_length = max(0.18, arrow_length - arrow_head_length)
    arrow_head_base_length = ARROW_HEAD_BASE_FRACTION * arrow_head_length
    arrow_head_mid_length = ARROW_HEAD_MID_FRACTION * arrow_head_length
    arrow_head_tip_length = ARROW_HEAD_TIP_FRACTION * arrow_head_length
    cfg.scene.goal_arrow_anchor = _marker_cfg(
        "GoalArrowAnchor", (0.28, 0.28, 0.025), (0.0, 0.0, 0.035), arrow_green
    )
    cfg.scene.goal_arrow_shaft = _marker_cfg(
        "GoalArrowShaft", (arrow_shaft_length, arrow_shaft_width, 0.025),
        (0.5 * arrow_shaft_length, 0.0, 0.035), arrow_green
    )
    cfg.scene.goal_arrow_head_base = _marker_cfg(
        "GoalArrowHeadBase", (arrow_head_base_length, arrow_head_base_width, 0.025),
        (arrow_shaft_length + 0.5 * arrow_head_base_length, 0.0, 0.035), arrow_green
    )
    cfg.scene.goal_arrow_head_mid = _marker_cfg(
        "GoalArrowHeadMid", (arrow_head_mid_length, arrow_head_mid_width, 0.025),
        (arrow_shaft_length + arrow_head_base_length + 0.5 * arrow_head_mid_length, 0.0, 0.035), arrow_green
    )
    cfg.scene.goal_arrow_head_tip = _marker_cfg(
        "GoalArrowHeadTip", (arrow_head_tip_length, arrow_head_tip_width, 0.025),
        (arrow_length - 0.5 * arrow_head_tip_length, 0.0, 0.035), arrow_green
    )

    cfg.viewer.eye = (9.0, 9.0, 6.0)
    cfg.viewer.lookat = (0.0, 0.0, 0.70)
    return cfg


def create_isaaclab_anymal_room_env(config: Any) -> Any:
    """Create the ManagerBasedRLEnv after AppLauncher is already running."""
    from isaaclab.envs import ManagerBasedRLEnv

    return ManagerBasedRLEnv(cfg=build_anymal_navigation_cfg(config))


class IsaacLabAnymalRoomVecEnv:
    """r2dreamer vector adapter for v13.45 camera-pose navigation on ANYmal-C."""

    def __init__(
        self,
        env: Any,
        config: Any,
        simulation_app: Any | None = None,
        image_size: tuple[int, int] | None = None,
    ):
        self._env = env
        self._unwrapped_env = env.unwrapped
        self._app = simulation_app
        self._config = config
        self._image_size = tuple(image_size) if image_size is not None else None
        self._num_envs = int(self._unwrapped_env.num_envs)
        self._device = torch.device(self._unwrapped_env.device)
        # Task randomness is independent of Dreamer's global RNG.  Each draw is
        # keyed by (task_seed, local env id, episode index, stream), so asynchronous
        # resets cannot change another environment's sequence.
        self._task_seed = int(_get(config, "task_seed", _get(config, "seed", 0)))
        self._task_episode_count = torch.zeros(
            self._num_envs, dtype=torch.int64, device=self._device
        )
        (
            self._camera_rel_pos_xyz,
            self._camera_ros_quat_wxyz,
            self._camera_rel_quat_xyzw,
            self._camera_pitch_deg,
        ) = _camera_mount_from_config(config)
        self._camera_world_z = ANYMAL_ROOT_Z + self._camera_rel_pos_xyz[2]
        self._camera_rel_xy_planar, self._camera_yaw_offset = self._camera_mount_planar()
        self._action_count = int(_get(config, "action_count", 11))
        self._time_limit = int(_get(config, "time_limit", 320))
        self._show_goal_arrow = bool(_get(config, "show_goal_arrow", True))
        self._arrow_target_anchor = str(
            _get(config, "arrow_target_anchor", "base")
        ).strip().lower()
        if self._arrow_target_anchor not in {"base", "tip"}:
            raise ValueError("arrow_target_anchor must be either 'base' or 'tip'")
        self._arrow_length = float(_get(config, "arrow_length", 1.60))
        self._arrow_head_length = float(_get(config, "arrow_head_length", 0.55))
        self._arrow_head_base_width = float(
            _get(config, "arrow_head_base_width", 0.68)
        )
        self._arrow_wall_clearance = float(
            _get(config, "arrow_wall_clearance", 0.10)
        )
        self._arrow_object_clearance = float(
            _get(config, "arrow_object_clearance", 0.10)
        )
        self._arrow_start_clearance = float(
            _get(config, "arrow_start_clearance", 0.35)
        )
        if self._arrow_length <= 0.0 or self._arrow_head_length <= 0.0:
            raise ValueError("arrow_length and arrow_head_length must be > 0")
        if self._arrow_head_length >= self._arrow_length:
            raise ValueError("arrow_head_length must be smaller than arrow_length")
        if self._arrow_head_base_width <= 0.0:
            raise ValueError("arrow_head_base_width must be > 0")
        if (
            self._arrow_wall_clearance < 0.0
            or self._arrow_object_clearance < 0.0
            or self._arrow_start_clearance < 0.0
        ):
            raise ValueError("arrow clearances must be non-negative")
        self._show_target_debug = bool(_get(config, "show_target_debug", False))
        self._camera_debug = bool(_get(config, "camera_debug", False))

        # Per-episode, per-environment landmark randomization. Clearances below
        # are measured from object surfaces, not object centers.
        self._randomize_objects = bool(_get(config, "randomize_objects", True))
        self._randomize_object_yaw = bool(_get(config, "randomize_object_yaw", True))
        self._object_wall_clearance = float(
            _get(config, "object_wall_clearance", 0.45)
        )
        self._object_min_clearance = float(
            _get(config, "object_min_clearance", 0.70)
        )
        self._object_sampling_attempts = int(
            _get(config, "object_sampling_attempts", 512)
        )
        if self._object_wall_clearance < 0.0 or self._object_min_clearance < 0.0:
            raise ValueError("object clearances must be non-negative")
        if self._object_sampling_attempts <= 0:
            raise ValueError("object_sampling_attempts must be positive")

        # Balanced randomization preserves continuous jitter but systematically
        # covers the joint start/goal ranges before repeating. Rejection fallback
        # keeps every reset feasible near walls and dense object layouts.
        self._balanced_randomization = bool(
            _get(config, "balanced_randomization", True)
        )
        self._start_sampling_attempts = int(
            _get(config, "start_sampling_attempts", 512)
        )
        self._start_stratified_attempts = int(
            _get(config, "start_stratified_attempts", 64)
        )
        self._start_grid_bins = int(_get(config, "start_grid_bins", 4))
        self._start_yaw_bins = int(_get(config, "start_yaw_bins", 8))
        self._goal_sampling_attempts = int(
            _get(config, "goal_sampling_attempts", 1024)
        )
        self._goal_stratified_attempts = int(
            _get(config, "goal_stratified_attempts", 64)
        )
        self._goal_distance_bins = int(_get(config, "goal_distance_bins", 4))
        self._goal_bearing_bins = int(_get(config, "goal_bearing_bins", 8))
        self._goal_yaw_bins = int(_get(config, "goal_yaw_bins", 8))
        self._object_yaw_bins = int(_get(config, "object_yaw_bins", 8))
        for name, value in (
            ("start_sampling_attempts", self._start_sampling_attempts),
            ("start_grid_bins", self._start_grid_bins),
            ("start_yaw_bins", self._start_yaw_bins),
            ("goal_sampling_attempts", self._goal_sampling_attempts),
            ("goal_distance_bins", self._goal_distance_bins),
            ("goal_bearing_bins", self._goal_bearing_bins),
            ("goal_yaw_bins", self._goal_yaw_bins),
            ("object_yaw_bins", self._object_yaw_bins),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self._start_stratified_attempts <= self._start_sampling_attempts:
            raise ValueError(
                "start_stratified_attempts must be in [0, start_sampling_attempts]"
            )
        if not 0 <= self._goal_stratified_attempts <= self._goal_sampling_attempts:
            raise ValueError(
                "goal_stratified_attempts must be in [0, goal_sampling_attempts]"
            )
        self._start_stratum_count = (
            self._start_grid_bins * self._start_grid_bins * self._start_yaw_bins
        )
        self._goal_stratum_count = (
            self._goal_distance_bins
            * self._goal_bearing_bins
            * self._goal_yaw_bins
        )
        self._start_stratum_stride = _coprime_stride(
            self._start_stratum_count, 53
        )
        self._goal_stratum_stride = _coprime_stride(
            self._goal_stratum_count, 73
        )

        # Proprioception contains only robot-internal/body-frame quantities:
        # base linear/angular velocity, projected gravity, relative joint angles,
        # and joint velocities. It never includes global XY/yaw or target geometry.
        self._proprioception_enabled = bool(
            _get(config, "proprioception_enabled", False)
        )
        self._proprio_linear_velocity_scale = float(
            _get(config, "proprio_linear_velocity_scale", 2.0)
        )
        self._proprio_angular_velocity_scale = float(
            _get(config, "proprio_angular_velocity_scale", 0.25)
        )
        self._proprio_joint_position_scale = float(
            _get(config, "proprio_joint_position_scale", 1.0)
        )
        self._proprio_joint_velocity_scale = float(
            _get(config, "proprio_joint_velocity_scale", 0.05)
        )
        self._proprio_clip = float(_get(config, "proprio_clip", 5.0))
        if self._proprio_clip <= 0.0:
            raise ValueError("proprio_clip must be positive")

        if self._action_count != 11:
            raise ValueError("The precise-pose ANYmal adapter defines exactly eleven discrete actions.")

        # Eleven high-level commands. The original v13.40 first nine actions
        # remain byte-for-byte equivalent in meaning and order; pure body-frame
        # left/right strafes are appended as actions 9 and 10. This preserves
        # forward arcs for efficient navigation while adding direct lateral
        # correction without changing heading. Positive body-frame y is left.
        forward_speed = float(_get(config, "forward_speed", 0.45))
        backward_speed = float(_get(config, "backward_speed", 0.25))
        arc_forward_speed = float(_get(config, "arc_forward_speed", 0.32))
        arc_yaw_rate = float(_get(config, "arc_yaw_rate", 0.35))
        coarse_yaw_rate = float(_get(config, "coarse_yaw_rate", 0.45))
        fine_yaw_rate = float(_get(config, "fine_yaw_rate", 0.20))
        lateral_speed = float(_get(config, "lateral_speed", 0.25))
        if lateral_speed <= 0.0:
            raise ValueError("lateral_speed must be positive")
        self._action_table = torch.tensor(
            [
                [forward_speed, 0.00, 0.00],
                [-backward_speed, 0.00, 0.00],
                [arc_forward_speed, 0.00, arc_yaw_rate],
                [arc_forward_speed, 0.00, -arc_yaw_rate],
                [0.00, 0.00, coarse_yaw_rate],
                [0.00, 0.00, -coarse_yaw_rate],
                [0.00, 0.00, 0.00],
                [0.00, 0.00, fine_yaw_rate],
                [0.00, 0.00, -fine_yaw_rate],
                [0.00, lateral_speed, 0.00],
                [0.00, -lateral_speed, 0.00],
            ],
            dtype=torch.float32,
            device=self._device,
        )

        # Camera-pose task values; distances/clearances come from the ANYmal YAML.
        self._goal_min_distance = float(_get(config, "goal_min_distance", 1.00))
        self._goal_max_distance = float(_get(config, "goal_max_distance", 3.50))
        self._goal_bearing_max = math.radians(float(_get(config, "goal_bearing_max_deg", 180.0)))
        self._goal_relative_yaw_max = math.radians(float(_get(config, "goal_relative_yaw_max_deg", 180.0)))
        self._goal_wall_clearance = float(_get(config, "goal_wall_clearance", 0.60))
        self._goal_object_clearance = float(_get(config, "goal_object_clearance", 0.70))
        self._position_tolerance = float(_get(config, "position_tolerance", 0.12))
        self._yaw_tolerance = math.radians(float(_get(config, "yaw_tolerance_deg", 5.0)))
        self._success_hold_steps = int(_get(config, "success_hold_steps", 5))

        # v13.14 SIMPLE STATE REWARD: no PBRS and no progress deltas.
        # Keep the validated v13.14 approach geometry, but pay a bounded,
        # non-negative reward for occupying better camera-pose states.
        self._geometry_yaw_metres_per_rad = float(
            _get(config, "geometry_yaw_metres_per_rad", 3.0)
        )
        self._geometry_rho0 = float(_get(config, "geometry_rho0", 1.0))
        self._geometry_k_alpha = float(_get(config, "geometry_k_alpha", 0.5))
        self._geometry_k_beta = float(_get(config, "geometry_k_beta", 0.25))
        self._geometry_k_yaw = float(_get(config, "geometry_k_yaw", 1.0))
        self._proximity_scale = float(_get(config, "proximity_scale", 0.5))
        self._proximity_sigma = float(_get(config, "proximity_sigma", 2.0))
        self._in_tolerance_reward = float(_get(config, "in_tolerance_reward", 0.5))

        if self._geometry_yaw_metres_per_rad <= 0.0:
            raise ValueError("geometry_yaw_metres_per_rad must be > 0")
        if self._geometry_rho0 <= 0.0:
            raise ValueError("geometry_rho0 must be > 0")
        if self._proximity_scale <= 0.0 or self._proximity_sigma <= 0.0:
            raise ValueError("proximity_scale and proximity_sigma must be > 0")
        if self._in_tolerance_reward < 0.0:
            raise ValueError("in_tolerance_reward must be >= 0")

        self._start_wall_clearance = float(_get(config, "start_wall_clearance", 0.90))
        self._start_object_clearance = float(_get(config, "start_object_clearance", 1.00))
        self._base_goal_wall_clearance = float(
            _get(config, "base_goal_wall_clearance", 0.90)
        )
        self._base_goal_object_clearance = float(
            _get(config, "base_goal_object_clearance", 1.00)
        )
        self._fall_height = float(_get(config, "fall_height", 0.32))
        self._upright_threshold = float(_get(config, "upright_threshold", 0.25))
        self._room_exit_margin = float(_get(config, "room_exit_margin", 0.10))

        # Conservative geometric guard for the outer walls only.
        # PhysX remains responsible for all normal wall/object contact. The
        # colored landmarks are intentionally *not* part of this guard, so
        # ANYmal may touch, step on, and climb over them without an artificial
        # reward penalty or episode reset.
        self._wall_guard_enabled = bool(
            _get(config, "wall_guard_enabled", True)
        )
        # The artificial wall guard can be diagnostic-only. Physical walls, fall,
        # room exit, success, and timeout remain independent termination signals.
        self._wall_guard_terminal = bool(
            _get(config, "wall_guard_terminal", False)
        )
        self._wall_guard_clearance = float(
            _get(config, "wall_guard_clearance", 0.70)
        )

        self._episode_steps = torch.zeros(
            self._num_envs, dtype=torch.int64, device=self._device
        )
        self._success_counter = torch.zeros_like(self._episode_steps)
        self._episode_ever_success = torch.zeros(
            self._num_envs, dtype=torch.bool, device=self._device
        )
        self._episode_in_tolerance_steps = torch.zeros_like(self._episode_steps)
        self._episode_max_hold_steps = torch.zeros_like(self._episode_steps)
        self._episode_first_success_step = torch.zeros_like(self._episode_steps)
        self._goal_xy = torch.zeros((self._num_envs, 2), device=self._device)
        self._goal_yaw = torch.zeros(self._num_envs, device=self._device)
        # Goal-sampler diagnostics are latched once per episode and emitted only
        # on the terminal step, so r2dreamer's sum-based episode logger preserves
        # their true values. Heading bins use desired robot-base yaw.
        self._goal_sample_attempts = torch.zeros_like(self._episode_steps)
        self._goal_sample_reject_wall = torch.zeros_like(self._episode_steps)
        self._goal_sample_reject_object = torch.zeros_like(self._episode_steps)
        self._goal_sample_reject_start = torch.zeros_like(self._episode_steps)
        self._goal_arrow_start_distance = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.float32
        )
        self._goal_heading_attempts = torch.zeros(
            (self._num_envs, GOAL_SAMPLING_HEADING_BINS),
            device=self._device,
            dtype=torch.int64,
        )
        self._goal_heading_rejections = torch.zeros_like(self._goal_heading_attempts)
        self._goal_heading_accepts = torch.zeros_like(self._goal_heading_attempts)
        canonical_object_xy = torch.tensor(
            OBJECT_XY, device=self._device, dtype=torch.float32
        )
        self._object_xy = canonical_object_xy.unsqueeze(0).repeat(
            self._num_envs, 1, 1
        )
        self._object_yaw = torch.zeros(
            (self._num_envs, len(PROP_NAMES)), device=self._device
        )
        self._object_guard_radii = torch.tensor(
            OBJECT_GUARD_RADII, device=self._device, dtype=torch.float32
        )
        self._episode_initial_position_error = torch.zeros(self._num_envs, device=self._device)
        self._episode_initial_yaw_error = torch.zeros(self._num_envs, device=self._device)
        self._episode_initial_bearing_error = torch.zeros(self._num_envs, device=self._device)
        self._episode_min_position_error = torch.full(
            (self._num_envs,), float("inf"), device=self._device
        )
        self._episode_min_yaw_error = torch.full(
            (self._num_envs,), float("inf"), device=self._device
        )
        self._episode_min_bearing_error = torch.full(
            (self._num_envs,), float("inf"), device=self._device
        )
        # Coupled pose diagnostics: these record the OTHER error at the instant
        # each component is best. They directly reveal whether position and yaw are
        # solved at different times in the episode.
        self._episode_yaw_at_min_position = torch.full(
            (self._num_envs,), float("inf"), device=self._device
        )
        self._episode_position_at_min_yaw = torch.full(
            (self._num_envs,), float("inf"), device=self._device
        )

        # Initialize manager/action-policy state once before custom placement.
        self._env.reset(seed=int(config.seed))
        robot_data = self._unwrapped_env.scene["robot"].data
        self._num_joints = int(_as_torch(robot_data.joint_pos).shape[-1])
        self._proprio_dim = 9 + 2 * self._num_joints

        # Human-only target camera visualization. This is an omni.ui.scene
        # overlay attached to the Kit viewport; it is not USD geometry and is
        # therefore absent from the TiledCamera RGB stream. The ordinary scene
        # green USD arrow remains independently visible in the TiledCamera.
        from envs.target_pose_debug import TargetDebugStyle, TargetPoseDebugDraw

        self._target_debug = TargetPoseDebugDraw(
            enabled=self._show_target_debug,
            camera_height=self._camera_world_z,
            style=TargetDebugStyle(
                arrow_length=float(_get(config, "target_debug_arrow_length", 0.85)),
                axis_length=float(_get(config, "target_debug_axis_length", 0.35)),
                frustum_distance=float(
                    _get(config, "target_debug_frustum_distance", 0.38)
                ),
                frustum_half_width=float(
                    _get(config, "target_debug_frustum_half_width", 0.20)
                ),
                frustum_half_height=float(
                    _get(config, "target_debug_frustum_half_height", 0.14)
                ),
                line_width=float(_get(config, "target_debug_line_width", 5.0)),
                point_size=float(_get(config, "target_debug_point_size", 12.0)),
                max_envs=int(_get(config, "target_debug_max_envs", 64)),
            ),
            strict=bool(_get(config, "target_debug_strict", False)),
        )

        # Validate that the official pretrained 3-D command action term exists.
        try:
            action_term = self._unwrapped_env.action_manager.get_term(
                "pre_trained_policy_action"
            )
            action_dim = int(action_term.action_dim)
        except Exception as exc:
            raise RuntimeError(
                "Could not access Isaac Lab's pretrained ANYmal-C action term. "
                "Check that Isaac-Navigation-Flat-Anymal-C-v0 is installed."
            ) from exc
        if action_dim != 3:
            raise RuntimeError(
                f"Expected pretrained ANYmal command action_dim=3, got {action_dim}."
            )

        all_ids = torch.arange(self._num_envs, device=self._device)
        self._reset_props(all_ids)
        self._place_robot_randomly(all_ids)
        self._sample_goal_for_ids(all_ids)
        self._advance_task_randomization(all_ids)
        pos_err, yaw_err, bearing_err = self._goal_metrics_from_body_pose()
        self._episode_initial_position_error.copy_(pos_err)
        self._episode_initial_yaw_error.copy_(yaw_err)
        self._episode_initial_bearing_error.copy_(bearing_err)
        self._episode_min_position_error.copy_(pos_err)
        self._episode_min_yaw_error.copy_(yaw_err)
        self._episode_min_bearing_error.copy_(bearing_err)
        self._episode_yaw_at_min_position.copy_(yaw_err)
        self._episode_position_at_min_yaw.copy_(pos_err)

        self._observation_space = self._build_observation_space()
        self._action_space = self._build_action_space()
        self._printed_camera_info = False
        self._last_debug_camera_pos: torch.Tensor | None = None

        rel_xy, yaw_offset = self._camera_rel_xy_planar, self._camera_yaw_offset
        print(
            "[ANYMAL CAMERA POSE] body-composed pose authority enabled: "
            f"fallback body-relative xy=({float(rel_xy[0]):.3f}, {float(rel_xy[1]):.3f}) m, "
            f"optical yaw offset={math.degrees(yaw_offset):.3f} deg; "
            "reward/goal/eval use live body pose + fixed camera mount; "
            "TiledCamera pos_w/quat_w_world are diagnostic-only because they may be stale."
        )

        print(
            "[ANYMAL CONTROL] official pretrained locomotion active; "
            f"Dreamer command dim={action_dim}, agent_decimation={self._unwrapped_env.cfg.decimation}"
        )
        print(
            "[ANYMAL RANDOMIZATION] "
            f"task_seed={self._task_seed}, per-env/per-episode streams=on, "
            f"balanced={self._balanced_randomization}, "
            f"start strata={self._start_stratum_count}, "
            f"goal strata={self._goal_stratum_count}, randomized object order=on"
        )
        print(
            "[ANYMAL PROPRIO] "
            f"enabled={self._proprioception_enabled}, dim={self._proprio_dim}; "
            "body velocities + gravity + relative joints only (no global pose/goal leak)"
        )

    @property
    def env_num(self) -> int:
        return self._num_envs

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self._action_space


    def _build_observation_space(self) -> gym.spaces.Dict:
        h, w = self._image_size or tuple(map(int, self._config.size))
        spaces: dict[str, gym.Space] = {
            "image": gym.spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8),
            # Optional auxiliary target retained in the observation schema:
            # [dx_camera, dy_camera, cos(delta_yaw), sin(delta_yaw)].
            # It is excluded from the encoder unless explicitly requested.
            "goal_vec": gym.spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32
            ),
        }
        if self._proprioception_enabled:
            spaces["proprio"] = gym.spaces.Box(
                -self._proprio_clip,
                self._proprio_clip,
                shape=(self._proprio_dim,),
                dtype=np.float32,
            )
        return gym.spaces.Dict(spaces)

    def _build_action_space(self) -> gym.spaces.Discrete:
        space = gym.spaces.Discrete(self._action_count)
        space.discrete = True
        return space

    def _decode_action(self, action: torch.Tensor) -> torch.Tensor:
        action = action.to(self._device)
        if action.ndim == 2 and action.shape[-1] == self._action_count:
            return torch.argmax(action, dim=-1)
        if action.ndim == 2 and action.shape[-1] == 1:
            return action[:, 0].long()
        if action.ndim == 1:
            return action.long()
        raise ValueError(
            f"Expected (N,{self._action_count}) one-hot or (N,1) integer actions, "
            f"received {tuple(action.shape)}"
        )

    def _camera(self) -> Any:
        sensors = getattr(self._unwrapped_env.scene, "sensors", {})
        if "head_camera" in sensors:
            return sensors["head_camera"]
        return self._unwrapped_env.scene["head_camera"]

    def _robot_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        robot = self._unwrapped_env.scene["robot"]
        state = _as_torch(robot.data.root_state_w)
        origins = _as_torch(self._unwrapped_env.scene.env_origins)
        base_xy = state[:, :2] - origins[:, :2]
        # Isaac Lab 3.0 stores quaternions as XYZW.
        qx, qy, qz, qw = state[:, 3], state[:, 4], state[:, 5], state[:, 6]
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy.square() + qz.square()),
        )
        return base_xy.float(), yaw.float(), (state[:, 2] - origins[:, 2]).float()

    def _camera_pose_world_reported_sensor(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pose reported by Isaac Lab's TiledCamera buffers (diagnostic only).

        Isaac Lab 3.0 / Isaac Sim 6.0.1 can leave ``data.pos_w`` and
        ``data.quat_w_world`` frozen for cameras attached below an articulation.
        Keep these values only so the diagnostics can expose that bug; they are
        never used for reward, success, goal observations, or evaluation.
        """
        camera = self._camera()
        camera_pos_w = _as_torch(camera.data.pos_w).to(self._device, dtype=torch.float32)
        camera_quat_w = _quat_normalize_xyzw(
            _as_torch(camera.data.quat_w_world).to(self._device, dtype=torch.float32)
        )
        if (
            camera_pos_w.shape[0] != self._num_envs
            or camera_quat_w.shape[0] != self._num_envs
            or not torch.isfinite(camera_pos_w).all()
            or not torch.isfinite(camera_quat_w).all()
        ):
            raise RuntimeError("TiledCamera reported pose buffers are unavailable or non-finite.")
        qx, qy, qz, qw = camera_quat_w.unbind(-1)
        camera_yaw_w = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy.square() + qz.square()),
        )
        return camera_pos_w, camera_quat_w, camera_yaw_w

    def _camera_pose_world(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Authoritative camera pose from the moving robot body + fixed mount.

        Do NOT use ``TiledCamera.data.pos_w`` here.  On the user's Isaac Lab 3.0
        stack that buffer was measured to remain exactly frozen while ANYmal's
        base translated ~0.4 m and rotated >20 degrees.  The rigid composition
        below uses the live articulation root pose, which the physical action
        benchmark confirmed updates correctly.

        This single pose source is used by reward, success, the 4-D relative-goal
        observation, evaluation, and diagnostics.  RGB still comes from the
        TiledCamera and is checked separately by ``rgb_motion_benchmark.py``.
        """
        return self._camera_pose_world_from_body()

    def _camera_pose_world_from_body(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Authoritative rigid-body composition of live robot pose and fixed camera mount."""
        robot = self._unwrapped_env.scene["robot"]
        state = _as_torch(robot.data.root_state_w).to(self._device, dtype=torch.float32)
        body_pos_w = state[:, :3]
        body_quat_w = _quat_normalize_xyzw(state[:, 3:7])
        rel_pos = torch.tensor(
            self._camera_rel_pos_xyz, device=self._device, dtype=torch.float32
        ).expand(self._num_envs, -1)
        rel_quat = _quat_normalize_xyzw(
            torch.tensor(
                self._camera_rel_quat_xyzw, device=self._device, dtype=torch.float32
            ).expand(self._num_envs, -1)
        )
        camera_pos_w = body_pos_w + _quat_apply_xyzw(body_quat_w, rel_pos)
        camera_quat_w = _quat_normalize_xyzw(_quat_mul_xyzw(body_quat_w, rel_quat))
        optical_forward = torch.tensor(
            CAMERA_LOCAL_FORWARD, device=self._device, dtype=torch.float32
        ).expand(self._num_envs, -1)
        forward_w = _quat_apply_xyzw(camera_quat_w, optical_forward)
        camera_yaw_w = torch.atan2(forward_w[:, 1], forward_w[:, 0])
        return camera_pos_w, camera_quat_w, camera_yaw_w

    def _camera_pose_local(self) -> tuple[torch.Tensor, torch.Tensor]:
        camera_pos_w, _, camera_yaw_w = self._camera_pose_world()
        origins = _as_torch(self._unwrapped_env.scene.env_origins).to(
            self._device, dtype=torch.float32
        )
        camera_xy = camera_pos_w[:, :2] - origins[:, :2]
        return camera_xy.float(), camera_yaw_w.float()

    def _camera_mount_planar(self) -> tuple[torch.Tensor, float]:
        rel_pos = torch.tensor(
            self._camera_rel_pos_xyz, device=self._device, dtype=torch.float32
        )
        rel_quat = _quat_normalize_xyzw(
            torch.tensor(
                self._camera_rel_quat_xyzw, device=self._device, dtype=torch.float32
            )
        )
        optical_forward = torch.tensor(
            CAMERA_LOCAL_FORWARD, device=self._device, dtype=torch.float32
        )
        forward_body = _quat_apply_xyzw(rel_quat, optical_forward)
        yaw_offset = math.atan2(float(forward_body[1]), float(forward_body[0]))
        return rel_pos[:2], yaw_offset

    def _goal_base_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Desired planar base pose implied by the target camera pose."""
        camera_rel_xy, camera_yaw_offset = self._camera_rel_xy_planar, self._camera_yaw_offset
        goal_base_yaw = _wrap_to_pi(self._goal_yaw - camera_yaw_offset)
        c, sn = torch.cos(goal_base_yaw), torch.sin(goal_base_yaw)
        offset_w = torch.stack([c * camera_rel_xy[0] - sn * camera_rel_xy[1], sn * camera_rel_xy[0] + c * camera_rel_xy[1]], dim=-1)
        return self._goal_xy - offset_w, goal_base_yaw

    def _pose_geometry(self) -> dict[str, torch.Tensor]:
        """Smooth metre-equivalent error used by the dense state reward.

        The goal-base pose is privileged reward geometry only. Dreamer receives
        RGB only; the 4-D relative target vector is not fed to the policy.
        """
        base_xy, base_yaw, _ = self._robot_state()
        goal_base_xy, goal_base_yaw = self._goal_base_pose()
        delta = goal_base_xy - base_xy
        rho = torch.linalg.vector_norm(delta, dim=-1)
        line_yaw = torch.atan2(delta[:, 1], delta[:, 0])
        alpha = _wrap_to_pi(line_yaw - base_yaw)
        beta = _wrap_to_pi(goal_base_yaw - line_yaw)
        _, camera_yaw = self._camera_pose_local()
        camera_yaw_error_signed = _wrap_to_pi(self._goal_yaw - camera_yaw)

        rho0_sq = max(self._geometry_rho0, 1e-6) ** 2
        gate = rho.square() / (rho.square() + rho0_sq)
        yaw_m = max(self._geometry_yaw_metres_per_rad, 1e-6)
        e_rho = rho
        e_alpha = gate * self._geometry_k_alpha * yaw_m * alpha
        e_beta = gate * self._geometry_k_beta * yaw_m * beta
        e_yaw = (1.0 - gate) * self._geometry_k_yaw * yaw_m * camera_yaw_error_signed
        error = torch.sqrt(
            e_rho.square() + e_alpha.square() + e_beta.square() + e_yaw.square() + 1e-8
        )
        return {
            "rho": rho,
            "alpha": alpha,
            "beta": beta,
            "camera_yaw_signed": camera_yaw_error_signed,
            "gate": gate,
            "error": error,
        }

    def _goal_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return position, final-yaw, and target-bearing errors for reward shaping."""
        camera_xy, camera_yaw = self._camera_pose_local()
        delta_w = self._goal_xy - camera_xy
        position_error = torch.linalg.vector_norm(delta_w, dim=-1)
        yaw_error = torch.abs(_wrap_to_pi(self._goal_yaw - camera_yaw))
        cos_yaw, sin_yaw = torch.cos(camera_yaw), torch.sin(camera_yaw)
        dx_camera = cos_yaw * delta_w[:, 0] + sin_yaw * delta_w[:, 1]
        dy_camera = -sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]
        bearing_error = torch.abs(torch.atan2(dy_camera, dx_camera))
        return position_error, yaw_error, bearing_error

    def _goal_errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        position_error, yaw_error, _ = self._goal_metrics()
        return position_error, yaw_error

    def _goal_metrics_from_body_pose(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reset-time pose metrics before a fresh rendered camera frame is guaranteed."""
        camera_pos_w, _, camera_yaw = self._camera_pose_world_from_body()
        origins = _as_torch(self._unwrapped_env.scene.env_origins).to(
            self._device, dtype=torch.float32
        )
        camera_xy = camera_pos_w[:, :2] - origins[:, :2]
        delta_w = self._goal_xy - camera_xy
        position_error = torch.linalg.vector_norm(delta_w, dim=-1)
        yaw_error = torch.abs(_wrap_to_pi(self._goal_yaw - camera_yaw))
        cos_yaw, sin_yaw = torch.cos(camera_yaw), torch.sin(camera_yaw)
        dx_camera = cos_yaw * delta_w[:, 0] + sin_yaw * delta_w[:, 1]
        dy_camera = -sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]
        bearing_error = torch.abs(torch.atan2(dy_camera, dx_camera))
        return position_error, yaw_error, bearing_error

    def _goal_errors_from_body_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        position_error, yaw_error, _ = self._goal_metrics_from_body_pose()
        return position_error, yaw_error

    def _goal_conditioned_observation(self) -> torch.Tensor:
        """Target camera pose expressed entirely in the current camera frame."""
        camera_xy, camera_yaw = self._camera_pose_local()
        delta_w = self._goal_xy - camera_xy
        cos_yaw, sin_yaw = torch.cos(camera_yaw), torch.sin(camera_yaw)
        dx_camera = cos_yaw * delta_w[:, 0] + sin_yaw * delta_w[:, 1]
        dy_camera = -sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]
        delta_yaw = _wrap_to_pi(self._goal_yaw - camera_yaw)
        return torch.stack(
            [
                dx_camera,
                dy_camera,
                torch.cos(delta_yaw),
                torch.sin(delta_yaw),
            ],
            dim=-1,
        ).float()

    def _read_camera(self) -> torch.Tensor:
        camera = self._camera()
        if "rgb" not in camera.data.output:
            raise RuntimeError(
                f"Camera outputs are {list(camera.data.output.keys())}; expected 'rgb'."
            )
        raw = _as_torch(camera.data.output["rgb"]).clone()
        if raw.shape[-1] == 4:
            raw = raw[..., :3]

        if raw.dtype != torch.uint8:
            if not raw.dtype.is_floating_point:
                raise TypeError(f"Unexpected RGB dtype: {raw.dtype}")
            # Detect old [0,1] versus [0,255] float-camera conventions once.
            maximum = float(raw.detach().amax().item())
            if maximum <= 1.5:
                raw = raw * 255.0
            elif maximum > 255.5:
                raise RuntimeError(f"Unexpected floating RGB maximum: {maximum:.3f}")
            raw = raw.round().clamp(0, 255).to(torch.uint8)

        if self._image_size is not None:
            h, w = self._image_size
            if raw.shape[1] != h or raw.shape[2] != w:
                raw = _resize_images(raw, self._image_size)

        if not self._printed_camera_info:
            print(
                "[ANYMAL CAMERA]",
                "shape=", tuple(raw.shape),
                "dtype=", raw.dtype,
                "configured_size=", self._image_size,
            )
            self._printed_camera_info = True
        return raw


    def _make_data(self) -> dict[str, torch.Tensor]:
        # goal_vec remains in replay for optional diagnostics/decoder loss, but it is
        # never included in the policy path unless the caller explicitly changes the
        # encoder regex. Proprioception is included only in the dedicated variant.
        data = {
            "image": self._read_camera(),
            "goal_vec": self._goal_conditioned_observation(),
        }
        if self._proprioception_enabled:
            data["proprio"] = self._proprioceptive_observation()
        return data

    def _task_random(
        self,
        env_ids: torch.Tensor,
        shape_tail: tuple[int, ...],
        stream: str,
    ) -> torch.Tensor:
        """Stateless random pool keyed independently for each env and episode."""
        env_ids = env_ids.to(self._device, dtype=torch.long).reshape(-1)
        if env_ids.numel() == 0:
            return torch.empty((0, *shape_tail), device=self._device)
        env_list = env_ids.detach().cpu().tolist()
        episode_list = (
            self._task_episode_count[env_ids].detach().cpu().tolist()
        )
        rows = []
        for env_id, episode_index in zip(env_list, episode_list):
            # Generate on CPU, then transfer once. This reset-only path is small,
            # avoids CUDA-generator/version quirks, and gives the task sampler the
            # same deterministic sequence on CPU and GPU simulator devices.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _mixed_task_seed(
                    self._task_seed,
                    int(env_id),
                    int(episode_index),
                    stream,
                )
            )
            rows.append(
                torch.rand(
                    shape_tail,
                    device="cpu",
                    dtype=torch.float32,
                    generator=generator,
                )
            )
        return torch.stack(rows, dim=0).to(self._device)


    def _task_stratum_index(
        self,
        env_ids: torch.Tensor,
        total: int,
        stride: int,
        phase_stream: str,
    ) -> torch.Tensor:
        """Full-cycle joint stratum index with a different phase per environment."""
        env_ids = env_ids.to(self._device, dtype=torch.long).reshape(-1)
        if total <= 1:
            return torch.zeros_like(env_ids)
        phases = torch.tensor(
            [
                _mixed_task_seed(self._task_seed, int(env_id), 0, phase_stream) % total
                for env_id in env_ids.detach().cpu().tolist()
            ],
            dtype=torch.int64,
            device=self._device,
        )
        episodes = self._task_episode_count[env_ids]
        return (phases + episodes * int(stride)).remainder(total)


    def _advance_task_randomization(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(self._device, dtype=torch.long).reshape(-1)
        if env_ids.numel():
            self._task_episode_count[env_ids] += 1


    def _proprioceptive_observation(self) -> torch.Tensor:
        """Normalized robot-only proprioception with no task/global-pose leakage."""
        robot = self._unwrapped_env.scene["robot"]
        data = robot.data

        root_lin_vel_b = getattr(data, "root_lin_vel_b", None)
        root_ang_vel_b = getattr(data, "root_ang_vel_b", None)
        projected_gravity_b = getattr(data, "projected_gravity_b", None)

        if root_lin_vel_b is None or root_ang_vel_b is None or projected_gravity_b is None:
            state = _as_torch(data.root_state_w).to(self._device, dtype=torch.float32)
            body_quat = _quat_normalize_xyzw(state[:, 3:7])
            inverse_quat = torch.cat([-body_quat[:, :3], body_quat[:, 3:4]], dim=-1)
            root_lin_vel_b_t = _quat_apply_xyzw(inverse_quat, state[:, 7:10])
            root_ang_vel_b_t = _quat_apply_xyzw(inverse_quat, state[:, 10:13])
            gravity_w = torch.tensor(
                [0.0, 0.0, -1.0], device=self._device, dtype=torch.float32
            ).expand(self._num_envs, -1)
            projected_gravity_b_t = _quat_apply_xyzw(inverse_quat, gravity_w)
        else:
            root_lin_vel_b_t = _as_torch(root_lin_vel_b).to(
                self._device, dtype=torch.float32
            )
            root_ang_vel_b_t = _as_torch(root_ang_vel_b).to(
                self._device, dtype=torch.float32
            )
            projected_gravity_b_t = _as_torch(projected_gravity_b).to(
                self._device, dtype=torch.float32
            )

        joint_pos = _as_torch(data.joint_pos).to(self._device, dtype=torch.float32)
        joint_vel = _as_torch(data.joint_vel).to(self._device, dtype=torch.float32)
        default_joint_pos = _as_torch(data.default_joint_pos).to(
            self._device, dtype=torch.float32
        )
        if default_joint_pos.ndim == 1:
            default_joint_pos = default_joint_pos.unsqueeze(0).expand_as(joint_pos)

        proprio = torch.cat(
            [
                root_lin_vel_b_t * self._proprio_linear_velocity_scale,
                root_ang_vel_b_t * self._proprio_angular_velocity_scale,
                projected_gravity_b_t,
                (joint_pos - default_joint_pos) * self._proprio_joint_position_scale,
                joint_vel * self._proprio_joint_velocity_scale,
            ],
            dim=-1,
        )
        if proprio.shape[-1] != self._proprio_dim:
            raise RuntimeError(
                f"Expected proprio dim {self._proprio_dim}, got {proprio.shape[-1]}"
            )
        proprio = torch.nan_to_num(
            proprio,
            nan=0.0,
            posinf=self._proprio_clip,
            neginf=-self._proprio_clip,
        )
        return proprio.clamp(-self._proprio_clip, self._proprio_clip).float()


    def _sample_object_layout(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample unbiased four-object layouts with random placement order per env."""
        env_ids = env_ids.to(self._device, dtype=torch.long).reshape(-1)
        n = int(env_ids.numel())
        object_count = len(PROP_NAMES)
        if n == 0:
            return torch.empty((0, object_count, 2), device=self._device)

        canonical = torch.tensor(
            OBJECT_XY, device=self._device, dtype=torch.float32
        ).unsqueeze(0).repeat(n, 1, 1)
        if not self._randomize_objects:
            return canonical

        # A different order for every environment/episode removes the fixed-order
        # bias where the first object always receives the easiest placement.
        order_keys = self._task_random(env_ids, (object_count,), "object_order")
        placement_order = torch.argsort(order_keys, dim=1)
        candidate_u = self._task_random(
            env_ids,
            (object_count, self._object_sampling_attempts, 2),
            "object_xy",
        )

        selected_by_object = torch.zeros_like(canonical)
        placed_xy = torch.zeros_like(canonical)
        placed_radii = torch.zeros((n, object_count), device=self._device)
        batch = torch.arange(n, device=self._device)

        for slot in range(object_count):
            object_index = placement_order[:, slot]
            radius = self._object_guard_radii[object_index]
            interior = ROOM_HALF - self._object_wall_clearance - radius
            if bool((interior <= 0.0).any()):
                raise ValueError(
                    "object_wall_clearance leaves no valid room interior for at least one object"
                )

            candidates = (
                candidate_u[:, slot] * 2.0 - 1.0
            ) * interior[:, None, None]
            filled = torch.zeros(n, dtype=torch.bool, device=self._device)
            for attempt in range(self._object_sampling_attempts):
                candidate = candidates[:, attempt]
                valid = torch.ones(n, dtype=torch.bool, device=self._device)
                if slot:
                    distances = torch.linalg.vector_norm(
                        candidate[:, None, :] - placed_xy[:, :slot, :], dim=-1
                    )
                    required = (
                        radius[:, None]
                        + placed_radii[:, :slot]
                        + self._object_min_clearance
                    )
                    valid &= (distances >= required).all(dim=1)
                take = (~filled) & valid
                if bool(take.any()):
                    take_batch = batch[take]
                    take_object = object_index[take]
                    selected_by_object[take_batch, take_object] = candidate[take]
                    placed_xy[take, slot] = candidate[take]
                    placed_radii[take, slot] = radius[take]
                filled |= take
                if bool(filled.all()):
                    break

            if not bool(filled.all()):
                missing = int((~filled).sum().item())
                raise RuntimeError(
                    "Could not sample randomized landmark layouts for "
                    f"{missing}/{n} environments at placement slot {slot}. "
                    "Reduce object clearances."
                )
        return selected_by_object


    def _sample_free_xy(
        self,
        env_ids: torch.Tensor,
        wall_clearance: float,
        object_clearance: float,
    ) -> torch.Tensor:
        """Sample generic free points with per-env/per-episode deterministic streams."""
        env_ids = env_ids.to(self._device, dtype=torch.long).reshape(-1)
        n = int(env_ids.numel())
        objects = self._object_xy[env_ids]
        interior = ROOM_HALF - wall_clearance
        if interior <= 0:
            raise ValueError("wall_clearance leaves no valid room interior")

        pool = self._task_random(
            env_ids, (self._start_sampling_attempts, 2), "free_xy"
        )
        selected = torch.zeros((n, 2), device=self._device)
        filled = torch.zeros(n, dtype=torch.bool, device=self._device)
        for attempt in range(self._start_sampling_attempts):
            candidate = (pool[:, attempt] * 2.0 - 1.0) * interior
            distances = torch.linalg.vector_norm(
                candidate[:, None, :] - objects, dim=-1
            )
            valid = distances.amin(dim=1) >= object_clearance
            take = (~filled) & valid
            selected[take] = candidate[take]
            filled |= take
            if bool(filled.all()):
                break

        if not bool(filled.all()):
            raise RuntimeError(
                "Could not sample collision-free points against the randomized object "
                "layouts. Reduce clearances."
            )
        return selected


    def _place_robot_randomly(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(self._device, dtype=torch.long).reshape(-1)
        n = int(env_ids.numel())
        if n == 0:
            return

        robot = self._unwrapped_env.scene["robot"]
        origins = _as_torch(self._unwrapped_env.scene.env_origins)[env_ids]
        default = _as_torch(robot.data.default_root_state)[env_ids].clone()
        objects = self._object_xy[env_ids]
        interior = ROOM_HALF - self._start_wall_clearance
        if interior <= 0.0:
            raise ValueError("start_wall_clearance leaves no valid room interior")

        xy_pool = self._task_random(
            env_ids, (self._start_sampling_attempts, 2), "start_xy"
        )
        if self._balanced_randomization:
            joint_index = self._task_stratum_index(
                env_ids,
                self._start_stratum_count,
                self._start_stratum_stride,
                "start_phase",
            )
            grid_count = self._start_grid_bins * self._start_grid_bins
            grid_index = joint_index.remainder(grid_count)
            yaw_bin = torch.div(joint_index, grid_count, rounding_mode="floor")
            x_bin = grid_index.remainder(self._start_grid_bins)
            y_bin = torch.div(grid_index, self._start_grid_bins, rounding_mode="floor")
        else:
            x_bin = y_bin = yaw_bin = None

        local_xy = torch.zeros((n, 2), device=self._device)
        filled = torch.zeros(n, dtype=torch.bool, device=self._device)
        for attempt in range(self._start_sampling_attempts):
            unit = xy_pool[:, attempt]
            if (
                self._balanced_randomization
                and attempt < self._start_stratified_attempts
            ):
                unit = torch.stack(
                    [
                        (x_bin.float() + unit[:, 0]) / self._start_grid_bins,
                        (y_bin.float() + unit[:, 1]) / self._start_grid_bins,
                    ],
                    dim=-1,
                )
            candidate = (unit * 2.0 - 1.0) * interior
            distances = torch.linalg.vector_norm(
                candidate[:, None, :] - objects, dim=-1
            )
            valid = distances.amin(dim=1) >= self._start_object_clearance
            take = (~filled) & valid
            local_xy[take] = candidate[take]
            filled |= take
            if bool(filled.all()):
                break
        if not bool(filled.all()):
            raise RuntimeError(
                "Could not sample collision-free ANYmal starts against randomized "
                "objects. Reduce start clearances."
            )

        yaw_u = self._task_random(env_ids, (1,), "start_yaw")[:, 0]
        if self._balanced_randomization:
            yaw_unit = (yaw_bin.float() + yaw_u) / self._start_yaw_bins
        else:
            yaw_unit = yaw_u
        yaw = (yaw_unit * 2.0 - 1.0) * math.pi

        default[:, 0] = origins[:, 0] + local_xy[:, 0]
        default[:, 1] = origins[:, 1] + local_xy[:, 1]
        default[:, 2] = origins[:, 2] + ANYMAL_ROOT_Z
        default[:, 3:7] = _yaw_quaternion(yaw)
        default[:, 7:] = 0.0

        robot.write_root_pose_to_sim(default[:, :7], env_ids=env_ids)
        robot.write_root_velocity_to_sim(default[:, 7:], env_ids=env_ids)

        joint_pos = _as_torch(robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = _as_torch(robot.data.default_joint_vel)[env_ids].clone()
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        robot.set_joint_position_target(joint_pos, env_ids=env_ids)


    def _reset_props(self, env_ids: torch.Tensor) -> None:
        """Place all four kinematic landmarks in a fresh random layout."""
        env_ids = env_ids.to(self._device, dtype=torch.long).reshape(-1)
        n = int(env_ids.numel())
        if n == 0:
            return

        local_xy = self._sample_object_layout(env_ids)
        if self._randomize_object_yaw:
            yaw_u = self._task_random(
                env_ids, (len(PROP_NAMES),), "object_yaw"
            )
            if self._balanced_randomization:
                env_list = env_ids.detach().cpu().tolist()
                phases = torch.tensor(
                    [
                        [
                            _mixed_task_seed(
                                self._task_seed,
                                int(env_id),
                                object_index,
                                "object_yaw_phase",
                            )
                            % self._object_yaw_bins
                            for object_index in range(len(PROP_NAMES))
                        ]
                        for env_id in env_list
                    ],
                    dtype=torch.int64,
                    device=self._device,
                )
                bins = (
                    phases
                    + self._task_episode_count[env_ids, None]
                ).remainder(self._object_yaw_bins)
                yaw_unit = (bins.float() + yaw_u) / self._object_yaw_bins
            else:
                yaw_unit = yaw_u
            yaw = (yaw_unit * 2.0 - 1.0) * math.pi
        else:
            yaw = torch.zeros(
                (n, len(PROP_NAMES)), device=self._device, dtype=torch.float32
            )

        self._object_xy[env_ids] = local_xy
        self._object_yaw[env_ids] = yaw
        origins = _as_torch(self._unwrapped_env.scene.env_origins)[env_ids]

        for object_index, name in enumerate(PROP_NAMES):
            asset = self._unwrapped_env.scene[name]
            state = _as_torch(asset.data.default_root_state)[env_ids].clone()
            state[:, 0] = origins[:, 0] + local_xy[:, object_index, 0]
            state[:, 1] = origins[:, 1] + local_xy[:, object_index, 1]
            state[:, 2] = origins[:, 2] + OBJECT_Z[object_index]
            state[:, 3:7] = _yaw_quaternion(yaw[:, object_index])
            state[:, 7:] = 0.0
            asset.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
            asset.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)

    def _sample_goal_for_ids(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(self._device, dtype=torch.long).reshape(-1)
        n = int(env_ids.numel())
        if n == 0:
            return

        camera_pos_w, _, camera_yaw_all = self._camera_pose_world_from_body()
        origins = _as_torch(self._unwrapped_env.scene.env_origins).to(
            self._device, dtype=torch.float32
        )
        start_cam_xy = (camera_pos_w[:, :2] - origins[:, :2])[env_ids]
        start_cam_yaw = camera_yaw_all[env_ids]
        start_base_xy_all, _, _ = self._robot_state()
        start_base_xy = start_base_xy_all[env_ids]

        selected_xy = torch.zeros((n, 2), device=self._device)
        selected_yaw = torch.zeros(n, device=self._device)
        selected_start_distance = torch.zeros(n, device=self._device)
        filled = torch.zeros(n, dtype=torch.bool, device=self._device)
        objects = self._object_xy[env_ids]
        camera_interior = ROOM_HALF - self._goal_wall_clearance
        base_interior = ROOM_HALF - self._base_goal_wall_clearance
        r2_min = self._goal_min_distance ** 2
        r2_span = self._goal_max_distance ** 2 - r2_min
        if r2_span < 0:
            raise ValueError("goal_max_distance must be >= goal_min_distance")

        attempts = torch.zeros(n, dtype=torch.int64, device=self._device)
        reject_wall = torch.zeros_like(attempts)
        reject_object = torch.zeros_like(attempts)
        reject_start = torch.zeros_like(attempts)
        heading_attempts = torch.zeros(
            (n, GOAL_SAMPLING_HEADING_BINS), dtype=torch.int64, device=self._device
        )
        heading_rejections = torch.zeros_like(heading_attempts)
        heading_accepts = torch.zeros_like(heading_attempts)

        random_pool = self._task_random(
            env_ids, (self._goal_sampling_attempts, 3), "goal"
        )
        if self._balanced_randomization:
            joint_index = self._task_stratum_index(
                env_ids,
                self._goal_stratum_count,
                self._goal_stratum_stride,
                "goal_phase",
            )
            distance_bin = joint_index.remainder(self._goal_distance_bins)
            remainder = torch.div(
                joint_index, self._goal_distance_bins, rounding_mode="floor"
            )
            bearing_bin = remainder.remainder(self._goal_bearing_bins)
            yaw_bin = torch.div(
                remainder, self._goal_bearing_bins, rounding_mode="floor"
            )
        else:
            distance_bin = bearing_bin = yaw_bin = None

        for attempt_index in range(self._goal_sampling_attempts):
            active = ~filled
            attempts += active.long()

            u_r = random_pool[:, attempt_index, 0]
            u_bearing = random_pool[:, attempt_index, 1]
            u_yaw = random_pool[:, attempt_index, 2]
            if (
                self._balanced_randomization
                and attempt_index < self._goal_stratified_attempts
            ):
                u_r = (distance_bin.float() + u_r) / self._goal_distance_bins
                u_bearing = (
                    bearing_bin.float() + u_bearing
                ) / self._goal_bearing_bins
                u_yaw = (yaw_bin.float() + u_yaw) / self._goal_yaw_bins

            # Squared-radius strata retain the original area-uniform marginal.
            radius = torch.sqrt(r2_min + u_r * r2_span)
            bearing = (u_bearing * 2.0 - 1.0) * self._goal_bearing_max
            direction = start_cam_yaw + bearing
            candidate = start_cam_xy + torch.stack(
                [radius * torch.cos(direction), radius * torch.sin(direction)], dim=-1
            )
            yaw_delta = (u_yaw * 2.0 - 1.0) * self._goal_relative_yaw_max
            yaw = _wrap_to_pi(start_cam_yaw + yaw_delta)

            camera_rel_xy = self._camera_rel_xy_planar
            camera_yaw_offset = self._camera_yaw_offset
            body_yaw = _wrap_to_pi(yaw - camera_yaw_offset)
            offset_x = (
                torch.cos(body_yaw) * camera_rel_xy[0]
                - torch.sin(body_yaw) * camera_rel_xy[1]
            )
            offset_y = (
                torch.sin(body_yaw) * camera_rel_xy[0]
                + torch.cos(body_yaw) * camera_rel_xy[1]
            )
            base_xy = candidate - torch.stack([offset_x, offset_y], dim=-1)

            camera_wall_ok = (candidate.abs() <= camera_interior).all(dim=-1)
            base_wall_ok = (base_xy.abs() <= base_interior).all(dim=-1)

            camera_object_dist = torch.linalg.vector_norm(
                candidate[:, None, :] - objects, dim=-1
            )
            base_object_dist = torch.linalg.vector_norm(
                base_xy[:, None, :] - objects, dim=-1
            )
            camera_object_ok = (
                camera_object_dist.amin(dim=1) >= self._goal_object_clearance
            )
            base_object_ok = (
                base_object_dist.amin(dim=1) >= self._base_goal_object_clearance
            )

            # The RGB arrow depicts the implied robot-base goal while reward and
            # success still evaluate the unchanged target camera pose.
            arrow_axis = torch.stack(
                [torch.cos(body_yaw), torch.sin(body_yaw)], dim=-1
            )
            if self._arrow_target_anchor == "base":
                arrow_tail = base_xy
                arrow_tip = base_xy + self._arrow_length * arrow_axis
            else:
                arrow_tail = base_xy - self._arrow_length * arrow_axis
                arrow_tip = base_xy
            arrow_center = 0.5 * (arrow_tail + arrow_tip)
            half_forward = 0.5 * self._arrow_length
            half_lateral = 0.5 * self._arrow_head_base_width
            arrow_extent = torch.stack(
                [
                    arrow_axis[:, 0].abs() * half_forward
                    + arrow_axis[:, 1].abs() * half_lateral,
                    arrow_axis[:, 1].abs() * half_forward
                    + arrow_axis[:, 0].abs() * half_lateral,
                ],
                dim=-1,
            )
            arrow_limit = ROOM_HALF - self._arrow_wall_clearance
            arrow_wall_ok = (
                (arrow_center.abs() + arrow_extent) <= arrow_limit
            ).all(dim=-1)

            object_from_tail = objects - arrow_tail[:, None, :]
            along_arrow = (object_from_tail * arrow_axis[:, None, :]).sum(dim=-1)
            along_arrow = along_arrow.clamp(0.0, self._arrow_length)
            closest_arrow_xy = (
                arrow_tail[:, None, :]
                + along_arrow[..., None] * arrow_axis[:, None, :]
            )
            arrow_object_dist = torch.linalg.vector_norm(
                objects - closest_arrow_xy, dim=-1
            )
            arrow_required = (
                self._object_guard_radii[None, :]
                + half_lateral
                + self._arrow_object_clearance
            )
            arrow_object_ok = (arrow_object_dist >= arrow_required).all(dim=1)

            # Prevent reset states in which ANYmal begins on top of the arrow.
            # The configured margin is measured from the arrow's outer footprint
            # to the initial robot-base center; the widest head width is used
            # conservatively for the full segment.
            start_from_tail = start_base_xy - arrow_tail
            along_start = (start_from_tail * arrow_axis).sum(dim=-1)
            along_start = along_start.clamp(0.0, self._arrow_length)
            closest_to_start = arrow_tail + along_start[:, None] * arrow_axis
            start_arrow_centerline_dist = torch.linalg.vector_norm(
                start_base_xy - closest_to_start, dim=-1
            )
            start_required = half_lateral + self._arrow_start_clearance
            arrow_start_ok = start_arrow_centerline_dist >= start_required

            wall_ok = camera_wall_ok & base_wall_ok & arrow_wall_ok
            object_ok = camera_object_ok & base_object_ok & arrow_object_ok
            valid = wall_ok & object_ok & arrow_start_ok

            # Rejection statistics use exclusive reason precedence
            # wall -> object -> initial-robot overlap. Heading is desired base yaw.
            heading_bin = torch.floor(
                (body_yaw + math.pi)
                * (GOAL_SAMPLING_HEADING_BINS / (2.0 * math.pi))
            ).long().remainder(GOAL_SAMPLING_HEADING_BINS)
            heading_attempts.scatter_add_(
                1, heading_bin[:, None], active.long()[:, None]
            )
            rejected = active & (~valid)
            heading_rejections.scatter_add_(
                1, heading_bin[:, None], rejected.long()[:, None]
            )
            accepted = active & valid
            heading_accepts.scatter_add_(
                1, heading_bin[:, None], accepted.long()[:, None]
            )
            reject_wall += (active & (~wall_ok)).long()
            reject_object += (active & wall_ok & (~object_ok)).long()
            reject_start += (active & wall_ok & object_ok & (~arrow_start_ok)).long()

            take = active & valid
            selected_xy[take] = candidate[take]
            selected_yaw[take] = yaw[take]
            selected_start_distance[take] = start_arrow_centerline_dist[take]
            filled |= take
            if bool(filled.all()):
                break

        if not bool(filled.all()):
            failed = torch.nonzero(~filled, as_tuple=False).flatten().tolist()
            raise RuntimeError(
                "Could not sample a reachable camera goal with a valid visible "
                f"arrow for local env indices {failed}. Reduce clearances or "
                "adjust the goal distribution."
            )

        self._goal_xy[env_ids] = selected_xy
        self._goal_yaw[env_ids] = selected_yaw
        self._goal_sample_attempts[env_ids] = attempts
        self._goal_sample_reject_wall[env_ids] = reject_wall
        self._goal_sample_reject_object[env_ids] = reject_object
        self._goal_sample_reject_start[env_ids] = reject_start
        self._goal_arrow_start_distance[env_ids] = selected_start_distance
        self._goal_heading_attempts[env_ids] = heading_attempts
        self._goal_heading_rejections[env_ids] = heading_rejections
        self._goal_heading_accepts[env_ids] = heading_accepts
        self._move_visible_goal(env_ids)
        self._redraw_target_debug()

    def _redraw_target_debug(self) -> None:
        """Refresh the human-only target pose overlay for all environments."""
        self._target_debug.redraw(
            goal_xy=self._goal_xy,
            goal_yaw=self._goal_yaw,
            env_origins=_as_torch(self._unwrapped_env.scene.env_origins),
        )

    def _write_marker_pose(
        self,
        asset_name: str,
        env_ids: torch.Tensor,
        local_xy: torch.Tensor,
        z: float,
        yaw: torch.Tensor,
        forward_offset: float = 0.0,
        lateral_offset: float = 0.0,
    ) -> None:
        origins = _as_torch(self._unwrapped_env.scene.env_origins)[env_ids]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        x = local_xy[:, 0] + cos_yaw * forward_offset - sin_yaw * lateral_offset
        y = local_xy[:, 1] + sin_yaw * forward_offset + cos_yaw * lateral_offset
        pose = torch.zeros((env_ids.numel(), 7), device=self._device)
        pose[:, 0] = origins[:, 0] + x
        pose[:, 1] = origins[:, 1] + y
        pose[:, 2] = origins[:, 2] + z
        pose[:, 3:7] = _yaw_quaternion(yaw)
        asset = self._unwrapped_env.scene[asset_name]
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(
            torch.zeros((env_ids.numel(), 6), device=self._device),
            env_ids=env_ids,
        )

    def _move_visible_goal(self, env_ids: torch.Tensor) -> None:
        """Render the unchanged v13.14 target camera pose as a green floor arrow.

        ``_goal_xy`` and ``_goal_yaw`` remain the mathematical target CAMERA pose
        used by the unchanged v13.14 reward. The visible arrow instead depicts the
        implied target ROBOT-BASE pose returned by ``_goal_base_pose()``.

        ``base`` (default): arrow base/origin = desired robot-base XY and arrow
        direction = desired robot-base yaw. With the fixed camera mount, reaching
        this base pose places the camera exactly at ``_goal_xy/_goal_yaw``.
        ``tip`` remains an optional visualization-only alternative.
        """
        env_ids = env_ids.to(self._device, dtype=torch.long).reshape(-1)
        if env_ids.numel() == 0:
            return

        if not self._show_goal_arrow:
            hidden_xy = torch.zeros((env_ids.numel(), 2), device=self._device)
            hidden_yaw = torch.zeros(env_ids.numel(), device=self._device)
            for name in (
                "goal_arrow_anchor",
                "goal_arrow_shaft",
                "goal_arrow_head_base",
                "goal_arrow_head_mid",
                "goal_arrow_head_tip",
            ):
                self._write_marker_pose(name, env_ids, hidden_xy, -100.0, hidden_yaw)
            return

        goal_base_xy, goal_base_yaw = self._goal_base_pose()
        xy = goal_base_xy[env_ids]
        yaw = goal_base_yaw[env_ids]
        floor_z = 0.035

        if self._arrow_target_anchor == "base":
            visual_tip = self._arrow_length
            visual_tail = 0.0
        else:  # optional alternate drawing: arrow tip == desired camera XY
            visual_tip = 0.0
            visual_tail = -self._arrow_length

        shaft_end = visual_tip - self._arrow_head_length
        shaft_center_forward = 0.5 * (visual_tail + shaft_end)
        head_base_len = ARROW_HEAD_BASE_FRACTION * self._arrow_head_length
        head_mid_len = ARROW_HEAD_MID_FRACTION * self._arrow_head_length
        head_tip_len = ARROW_HEAD_TIP_FRACTION * self._arrow_head_length
        head_base_center = shaft_end + 0.5 * head_base_len
        head_mid_center = shaft_end + head_base_len + 0.5 * head_mid_len
        head_tip_center = visual_tip - 0.5 * head_tip_len

        c, s = torch.cos(yaw), torch.sin(yaw)

        def goal_frame_xy(forward: float, lateral: float = 0.0) -> torch.Tensor:
            return xy + torch.stack(
                [c * forward - s * lateral, s * forward + c * lateral], dim=-1
            )

        # Target-point tile at the implied target robot-base XY.
        self._write_marker_pose("goal_arrow_anchor", env_ids, xy, floor_z, yaw)
        self._write_marker_pose(
            "goal_arrow_shaft",
            env_ids,
            goal_frame_xy(shaft_center_forward),
            floor_z,
            yaw,
        )
        # Tapered three-segment head: visually reads as a single triangle while
        # remaining robust Isaac Lab cuboid geometry.
        self._write_marker_pose(
            "goal_arrow_head_base",
            env_ids,
            goal_frame_xy(head_base_center),
            floor_z,
            yaw,
        )
        self._write_marker_pose(
            "goal_arrow_head_mid",
            env_ids,
            goal_frame_xy(head_mid_center),
            floor_z,
            yaw,
        )
        self._write_marker_pose(
            "goal_arrow_head_tip",
            env_ids,
            goal_frame_xy(head_tip_center),
            floor_z,
            yaw,
        )

    def _reset_base_env(self, env_ids: torch.Tensor) -> None:
        try:
            self._env.reset(env_ids=env_ids)
        except TypeError:
            # Compatibility fallback for older Isaac Lab versions.
            self._unwrapped_env._reset_idx(env_ids)

    def _reset_ids(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(self._device, dtype=torch.long).reshape(-1)
        if env_ids.numel() == 0:
            return
        self._reset_base_env(env_ids)
        self._reset_props(env_ids)
        self._place_robot_randomly(env_ids)
        self._sample_goal_for_ids(env_ids)
        self._advance_task_randomization(env_ids)
        pos_err, yaw_err, bearing_err = self._goal_metrics_from_body_pose()
        self._episode_initial_position_error[env_ids] = pos_err[env_ids]
        self._episode_initial_yaw_error[env_ids] = yaw_err[env_ids]
        self._episode_initial_bearing_error[env_ids] = bearing_err[env_ids]
        self._episode_min_position_error[env_ids] = pos_err[env_ids]
        self._episode_min_yaw_error[env_ids] = yaw_err[env_ids]
        self._episode_min_bearing_error[env_ids] = bearing_err[env_ids]
        self._episode_yaw_at_min_position[env_ids] = yaw_err[env_ids]
        self._episode_position_at_min_yaw[env_ids] = pos_err[env_ids]
        self._episode_steps[env_ids] = 0
        self._success_counter[env_ids] = 0
        self._episode_ever_success[env_ids] = False
        self._episode_in_tolerance_steps[env_ids] = 0
        self._episode_max_hold_steps[env_ids] = 0
        self._episode_first_success_step[env_ids] = 0

    def _fall_mask(self, done: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, root_z = self._robot_state()
        state = _as_torch(self._unwrapped_env.scene["robot"].data.root_state_w)
        # Quaternion is XYZW in Isaac Lab 3.0.
        qx, qy = state[:, 3], state[:, 4]
        # World-z component of the robot body +Z axis. Upright is close to +1,
        # sideways is 0, and upside down is -1.
        body_up_z = 1.0 - 2.0 * (qx.square() + qy.square())
        fall = (root_z < self._fall_height) | (body_up_z < self._upright_threshold)
        return fall & (~done), root_z

    def _wall_guard_mask(self, done: torch.Tensor) -> torch.Tensor:
        """Last-resort wall guard; ordinary object contact remains unrestricted."""
        if not self._wall_guard_enabled:
            return torch.zeros_like(done)
        base_xy, _, _ = self._robot_state()
        safe_half = ROOM_HALF - self._wall_guard_clearance
        near_wall = (base_xy.abs() >= safe_half).any(dim=-1)
        return near_wall & (~done)

    def _room_exit_mask(self, done: torch.Tensor) -> torch.Tensor:
        """Last-resort check if the base crossed the nominal room boundary."""
        base_xy, _, _ = self._robot_state()
        outside = (base_xy.abs() > (ROOM_HALF + self._room_exit_margin)).any(dim=-1)
        return outside & (~done)

    def step(self, action: torch.Tensor, done: torch.Tensor) -> tuple[TensorDict, torch.Tensor]:
        done = done.to(self._device, dtype=torch.bool)
        action_index = self._decode_action(action)

        if done.any():
            self._reset_ids(done.nonzero(as_tuple=False).squeeze(-1))

        # Diagnostic only: capture the target direction *before* applying the
        # selected action. This lets the plotter measure whether the actor is
        # learning target-conditioned choices rather than merely wandering.
        # It is not a policy input. The same 4-D quantity is retained only for
        # privileged action-alignment diagnostics and optional post-hoc probes.
        pre_goal_observation = self._goal_conditioned_observation()
        pre_dx_camera = pre_goal_observation[:, 0]
        pre_dy_camera = pre_goal_observation[:, 1]
        pre_position_error = torch.sqrt(
            pre_dx_camera.square() + pre_dy_camera.square()
        )
        pre_signed_bearing = torch.atan2(pre_dy_camera, pre_dx_camera)
        # Decoder target convention is [dx, dy, cos(delta_yaw), sin(delta_yaw)].
        # This is privileged logging only; it is not routed into Dreamer's encoder.
        pre_signed_yaw_error = torch.atan2(
            pre_goal_observation[:, 3], pre_goal_observation[:, 2]
        )

        velocity_command = self._action_table[action_index].clone()
        # A neutral high-level step after reset gives the locomotion policy one
        # rendered settling transition and prevents reward across episodes.
        velocity_command[done] = 0.0

        # Native reward/termination outputs are intentionally ignored.
        #
        # The pretrained locomotion policy must run without autograd because
        # Warp/PhysX cannot consume a CUDA tensor with requires_grad=True.
        # Use no_grad rather than inference_mode here: ManagerBasedRLEnv keeps
        # persistent command/reward metric buffers created or updated during
        # env.step(), and inference tensors cannot later be mutated by reset()
        # outside an InferenceMode context. no_grad prevents the locomotion
        # output from requiring gradients while keeping Isaac Lab's persistent
        # tensors as ordinary mutable tensors. Dreamer itself still trains with
        # gradients outside this simulator-only block.
        with torch.no_grad():
            self._env.step(velocity_command)
        data = self._make_data()

        self._episode_steps = torch.where(
            done, torch.zeros_like(self._episode_steps), self._episode_steps + 1
        )

        position_error, yaw_error, bearing_error = self._goal_metrics()
        # For freshly reset slots, replace the body-composed reset estimate with
        # the actual live rendered-camera metrics after the neutral settling step.
        # This makes d_initial and the logged initial pose match the post-settle live body-composed camera pose.
        self._episode_initial_position_error = torch.where(
            done, position_error, self._episode_initial_position_error
        )
        self._episode_initial_yaw_error = torch.where(
            done, yaw_error, self._episode_initial_yaw_error
        )
        self._episode_initial_bearing_error = torch.where(
            done, bearing_error, self._episode_initial_bearing_error
        )

        position_improved = done | (position_error < self._episode_min_position_error)
        yaw_improved = done | (yaw_error < self._episode_min_yaw_error)
        self._episode_yaw_at_min_position = torch.where(
            position_improved, yaw_error, self._episode_yaw_at_min_position
        )
        self._episode_position_at_min_yaw = torch.where(
            yaw_improved, position_error, self._episode_position_at_min_yaw
        )

        self._episode_min_position_error = torch.where(
            done, position_error, self._episode_min_position_error
        )
        self._episode_min_yaw_error = torch.where(
            done, yaw_error, self._episode_min_yaw_error
        )
        self._episode_min_bearing_error = torch.where(
            done, bearing_error, self._episode_min_bearing_error
        )

        self._episode_min_position_error = torch.minimum(
            self._episode_min_position_error, position_error
        )
        self._episode_min_yaw_error = torch.minimum(
            self._episode_min_yaw_error, yaw_error
        )
        self._episode_min_bearing_error = torch.minimum(
            self._episode_min_bearing_error, bearing_error
        )

        within_pose = (
            (position_error < self._position_tolerance)
            & (yaw_error < self._yaw_tolerance)
            & (~done)
        )
        self._success_counter = torch.where(
            within_pose,
            self._success_counter + 1,
            torch.zeros_like(self._success_counter),
        )
        self._episode_in_tolerance_steps += within_pose.long()
        self._episode_max_hold_steps = torch.maximum(
            self._episode_max_hold_steps, self._success_counter
        )
        success_event = self._success_counter >= self._success_hold_steps
        first_success = success_event & (~self._episode_ever_success)
        self._episode_first_success_step = torch.where(
            first_success,
            self._episode_steps,
            self._episode_first_success_step,
        )
        self._episode_ever_success |= success_event

        fall, root_z = self._fall_mask(done)
        wall_guard = self._wall_guard_mask(done)
        room_exit = self._room_exit_mask(done)
        external_timeout = self._episode_steps >= self._time_limit
        wall_terminal = wall_guard if self._wall_guard_terminal else torch.zeros_like(wall_guard)

        # Success is NON-TERMINAL: training means reach and maintain the desired
        # viewpoint. Fall/true room exit (and optional wall-terminal mode) remain
        # true terminals; timeout remains a truncation.
        true_terminal = fall | wall_terminal | room_exit
        episode_done = true_terminal | external_timeout
        timeout_only = external_timeout & (~true_terminal)

        geometry_post = self._pose_geometry()
        r_proximity = self._proximity_scale * torch.exp(
            -geometry_post["error"] / max(self._proximity_sigma, 1e-6)
        )
        r_in_tolerance = self._in_tolerance_reward * within_pose.float()

        # Exact v13.20 scalar reward: bounded proximity-state reward plus
        # per-step precise-tolerance occupancy reward. There are no progress,
        # joint-pose, step-cost, collision, or terminal-bonus terms.
        reward = torch.where(
            done | true_terminal,
            torch.zeros_like(r_proximity),
            r_proximity + r_in_tolerance,
        )

        # Physical terminal states receive no hand-coded negative reward; they
        # are undesirable because they forfeit future positive state reward.

        # r2dreamer's episode logger sums every ``log_*`` channel.  Therefore
        # final/min/initial quantities are emitted only on the terminal step,
        # while reward components and one-hot actions are intentionally summed.
        terminal_f = episode_done.float()
        active_f = (~done).float()

        # Privileged diagnostics use the exact robot-base pose depicted by the
        # visible arrow. These quantities never enter the observation, actor, or
        # RSSM posterior and do not contribute to reward.
        base_xy_diag, _, _ = self._robot_state()
        goal_base_xy_diag, goal_base_yaw_diag = self._goal_base_pose()
        base_delta_diag = goal_base_xy_diag - base_xy_diag
        base_position_error_diag = torch.linalg.vector_norm(base_delta_diag, dim=-1)
        cb, sb = torch.cos(goal_base_yaw_diag), torch.sin(goal_base_yaw_diag)
        longitudinal_error_diag = cb * base_delta_diag[:, 0] + sb * base_delta_diag[:, 1]
        lateral_error_diag = -sb * base_delta_diag[:, 0] + cb * base_delta_diag[:, 1]
        data["log_initial_position_error"] = (
            self._episode_initial_position_error * terminal_f
        ).unsqueeze(-1)
        data["log_final_position_error"] = (position_error * terminal_f).unsqueeze(-1)
        data["log_min_position_error"] = (
            self._episode_min_position_error * terminal_f
        ).unsqueeze(-1)
        data["log_initial_yaw_error_deg"] = (
            torch.rad2deg(self._episode_initial_yaw_error) * terminal_f
        ).unsqueeze(-1)
        data["log_final_yaw_error_deg"] = (
            torch.rad2deg(yaw_error) * terminal_f
        ).unsqueeze(-1)
        data["log_min_yaw_error_deg"] = (
            torch.rad2deg(self._episode_min_yaw_error) * terminal_f
        ).unsqueeze(-1)
        data["log_initial_bearing_error_deg"] = (
            torch.rad2deg(self._episode_initial_bearing_error) * terminal_f
        ).unsqueeze(-1)
        data["log_final_bearing_error_deg"] = (
            torch.rad2deg(bearing_error) * terminal_f
        ).unsqueeze(-1)
        data["log_min_bearing_error_deg"] = (
            torch.rad2deg(self._episode_min_bearing_error) * terminal_f
        ).unsqueeze(-1)
        data["log_yaw_at_min_position_deg"] = (
            torch.rad2deg(self._episode_yaw_at_min_position) * terminal_f
        ).unsqueeze(-1)
        data["log_position_at_min_yaw"] = (
            self._episode_position_at_min_yaw * terminal_f
        ).unsqueeze(-1)
        data["log_final_base_position_error"] = (
            base_position_error_diag * terminal_f
        ).unsqueeze(-1)
        data["log_final_abs_longitudinal_error"] = (
            longitudinal_error_diag.abs() * terminal_f
        ).unsqueeze(-1)
        data["log_final_abs_lateral_error"] = (
            lateral_error_diag.abs() * terminal_f
        ).unsqueeze(-1)
        data["log_timeout"] = timeout_only.float().unsqueeze(-1)
        data["log_success"] = (
            self._episode_ever_success.float() * terminal_f
        ).unsqueeze(-1)
        data["log_reward_proximity"] = (r_proximity * active_f).unsqueeze(-1)
        data["log_reward_in_tolerance"] = (r_in_tolerance * active_f).unsqueeze(-1)
        data["log_geometric_error"] = (geometry_post["error"] * active_f).unsqueeze(-1)
        data["log_base_goal_distance"] = (geometry_post["rho"] * active_f).unsqueeze(-1)
        data["log_in_tolerance_steps"] = within_pose.float().unsqueeze(-1)
        data["log_max_hold_steps"] = (
            self._episode_max_hold_steps.float() * terminal_f
        ).unsqueeze(-1)
        data["log_first_success_step"] = (
            self._episode_first_success_step.float() * terminal_f
        ).unsqueeze(-1)
        data["log_goal_sample_attempts"] = (
            self._goal_sample_attempts.float() * terminal_f
        ).unsqueeze(-1)
        data["log_goal_sample_reject_wall"] = (
            self._goal_sample_reject_wall.float() * terminal_f
        ).unsqueeze(-1)
        data["log_goal_sample_reject_object"] = (
            self._goal_sample_reject_object.float() * terminal_f
        ).unsqueeze(-1)
        data["log_goal_sample_reject_start"] = (
            self._goal_sample_reject_start.float() * terminal_f
        ).unsqueeze(-1)
        data["log_goal_arrow_start_distance"] = (
            self._goal_arrow_start_distance * terminal_f
        ).unsqueeze(-1)
        for heading_bin in range(GOAL_SAMPLING_HEADING_BINS):
            suffix = f"bin_{heading_bin}"
            data[f"log_goal_heading_attempts_{suffix}"] = (
                self._goal_heading_attempts[:, heading_bin].float() * terminal_f
            ).unsqueeze(-1)
            data[f"log_goal_heading_rejections_{suffix}"] = (
                self._goal_heading_rejections[:, heading_bin].float() * terminal_f
            ).unsqueeze(-1)
            data[f"log_goal_heading_accepts_{suffix}"] = (
                self._goal_heading_accepts[:, heading_bin].float() * terminal_f
            ).unsqueeze(-1)
        # Target-action alignment diagnostics. These are episode-summed counts;
        # the plotter turns them into percentages. Restrict to far-field
        # navigation (>0.5 m) so final-yaw behavior near the goal is not mislabeled.
        far_navigation = pre_position_error > 0.50
        bearing_threshold = math.radians(20.0)
        turn_opportunity = (
            far_navigation
            & (pre_signed_bearing.abs() > bearing_threshold)
            & (~done)
        )
        yaw_command = velocity_command[:, 2]
        signed_turn = yaw_command * pre_signed_bearing
        turn_toward = turn_opportunity & (signed_turn > 1e-6)
        turn_away = turn_opportunity & (signed_turn < -1e-6)
        no_yaw_when_turn_needed = turn_opportunity & (yaw_command.abs() <= 1e-6)

        ahead_opportunity = (
            far_navigation
            & (pre_dx_camera > 0.0)
            & (pre_signed_bearing.abs() <= bearing_threshold)
            & (~done)
        )
        forward_when_ahead = ahead_opportunity & (velocity_command[:, 0] > 0.05)

        data["log_target_turn_opportunities"] = turn_opportunity.float().unsqueeze(-1)
        data["log_turn_toward_target"] = turn_toward.float().unsqueeze(-1)
        data["log_turn_away_from_target"] = turn_away.float().unsqueeze(-1)
        data["log_no_yaw_when_turn_needed"] = no_yaw_when_turn_needed.float().unsqueeze(-1)
        data["log_target_ahead_opportunities"] = ahead_opportunity.float().unsqueeze(-1)
        data["log_forward_when_target_ahead"] = forward_when_ahead.float().unsqueeze(-1)

        # Diagnostic-only terminal-yaw action alignment. This exactly mirrors
        # the offline diagnostic: camera position < 1 m and |final yaw error| > 30 deg.
        # It never contributes to reward or observations.
        near_yaw_opportunity = (
            (pre_position_error < 1.0)
            & (pre_signed_yaw_error.abs() > math.radians(30.0))
            & (~done)
        )
        yaw_signed_alignment = yaw_command * pre_signed_yaw_error
        yaw_correcting_near = near_yaw_opportunity & (yaw_signed_alignment > 1e-6)
        yaw_away_near = near_yaw_opportunity & (yaw_signed_alignment < -1e-6)
        no_yaw_near = near_yaw_opportunity & (yaw_command.abs() <= 1e-6)
        data["log_near_yaw_opportunities"] = near_yaw_opportunity.float().unsqueeze(-1)
        data["log_yaw_correcting_near"] = yaw_correcting_near.float().unsqueeze(-1)
        data["log_yaw_away_near"] = yaw_away_near.float().unsqueeze(-1)
        data["log_no_yaw_near"] = no_yaw_near.float().unsqueeze(-1)

        action_oh = torch.nn.functional.one_hot(action_index, num_classes=self._action_count).float() * active_f[:, None]
        for action_i, action_name in enumerate((
            "forward", "backward", "left_arc", "right_arc", "coarse_left",
            "coarse_right", "stop", "fine_left", "fine_right",
            "strafe_left", "strafe_right",
        )):
            data[f"log_action_{action_name}"] = action_oh[:, action_i : action_i + 1]

        data["is_first"] = done.unsqueeze(-1)
        data["is_terminal"] = true_terminal.unsqueeze(-1)
        data["is_last"] = episode_done.unsqueeze(-1)
        data["reward"] = reward.unsqueeze(-1)
        data["log_fall"] = fall.float().unsqueeze(-1)
        data["log_wall_collision"] = wall_guard.float().unsqueeze(-1)
        # Kept for evaluator compatibility. Object contact is intentionally
        # allowed and is not treated as a collision/terminal event.
        data["log_obstacle_collision"] = torch.zeros_like(reward).unsqueeze(-1)
        data["log_collision"] = wall_guard.float().unsqueeze(-1)
        data["log_room_exit"] = room_exit.float().unsqueeze(-1)
        # These compatibility/debug values are terminal-masked because the
        # trainer sums every log_* channel over the episode.
        data["log_position_error"] = (position_error * terminal_f).unsqueeze(-1)
        data["log_yaw_error_deg"] = (
            torch.rad2deg(yaw_error) * terminal_f
        ).unsqueeze(-1)
        data["log_goal_x"] = (self._goal_xy[:, 0] * terminal_f).unsqueeze(-1)
        data["log_goal_y"] = (self._goal_xy[:, 1] * terminal_f).unsqueeze(-1)
        data["log_goal_yaw_deg"] = (
            torch.rad2deg(self._goal_yaw) * terminal_f
        ).unsqueeze(-1)
        data["log_root_z"] = (root_z * terminal_f).unsqueeze(-1)

        if self._camera_debug:
            camera_pos = getattr(self._camera().data, "pos_w", None)
            if camera_pos is not None:
                camera_pos_t = _as_torch(camera_pos)
                if self._last_debug_camera_pos is not None:
                    movement = torch.linalg.vector_norm(
                        camera_pos_t - self._last_debug_camera_pos, dim=-1
                    ).mean()
                    data["log_camera_pose_delta"] = movement.expand(
                        self._num_envs, 1
                    )
                self._last_debug_camera_pos = camera_pos_t.clone()

        transition = TensorDict(data, batch_size=(self._num_envs,), device=self._device)
        return transition, episode_done

    def close(self) -> None:
        try:
            self._target_debug.close()
            self._env.close()
        finally:
            if self._app is not None:
                self._app.close()
