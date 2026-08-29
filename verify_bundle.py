#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import math
import pathlib
import py_compile
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ENV = HERE / "payload/envs/isaaclab_anymal_room.py"
CFG = HERE / "payload/configs/env/isaaclab_anymal_room.yaml"

# Methods that remain executable-AST-identical to the validated v13.20/v13.14
# task. Camera mount construction and goal sampling are checked separately
# because v13.45 retains the validated camera/reward logic while changing only
# task randomization and optional observation routing.
EXPECTED_V13_20_AST = {
    "_build_action_space": "497764d7efc3f244cae4a6c5fa302fad155586e8fc4e4668fb495f77ffb70e8b",
    "_decode_action": "dbcacf6f9258b0b97e632f0149c62f706f49b0183f7fd2dea202c4bbf02f65f3",
    "_robot_state": "b0d8ec53e8e155af5ccac5f3a1b9938af7379f7e3a439db811fad5af1b72e7c2",
    "_camera_pose_world": "26a0f0783e7e06e9672b49b1432a118656057490685a188a4133985d7a515495",
    "_camera_pose_local": "b7674768dc45d1a0cbf1a7f6f2482ec9ed859a21a366709c89b028d7d5524520",
    "_goal_base_pose": "3ec2f8070813126d4205880bd81c0b092b207c56bf1ef0693e2c9684ed7c98f5",
    "_pose_geometry": "1ddaf15c947910788a0608d123a0a33902eeed8abac6a15e888c02a9223fe018",
    "_goal_metrics": "2f6b71d33cd240192e27ded909bac49ccd382bc944ef5a649c5e2854930a75a1",
    "_goal_errors": "c5c30d907daaa6283fc1e210c6e5e0b94613be6fefef239b06a2a45a3a1f71c9",
    "_goal_metrics_from_body_pose": "cb1eee42fea82484eebccbea0c7e86c1dc05eb0746e552b79bab62f951e63755",
    "_goal_errors_from_body_pose": "dadc51f89b311d60548492fef147edeee5bdb0439445360374811ea890b1aae7",
    "_goal_conditioned_observation": "2b536af87ad8a28c9454d97a68ab83b5d33bb1ef9dbecfb2d7f6f1e3209d8dba",
    "_reset_base_env": "ff0fec6b83cdd4a36488ca4fe75175bb76d9465c84055485a8c44aa1787f07e1",
    "_fall_mask": "b386095376f5486c08d1a08c51030bebfb63de11e59e0b4dc505e2325284ed9e",
    "_wall_guard_mask": "cfce695d9b07fe4726cedec17f6c1885404a77376ca9b9fd47d3d0255d31e674",
    "_room_exit_mask": "f0469323ec654d319311cee1950f55fbd33a46a13e70d3d098b03e240b7185d0",
}


def req(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[VERIFY] OK: {message}")


def method_logic_hashes(path: pathlib.Path) -> dict[str, str]:
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "IsaacLabAnymalRoomVecEnv"
    )
    hashes: dict[str, str] = {}
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        body = list(fn.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        clone = ast.FunctionDef(
            name=fn.name,
            args=fn.args,
            body=body,
            decorator_list=fn.decorator_list,
            returns=fn.returns,
            type_comment=fn.type_comment,
        )
        dumped = ast.dump(clone, annotate_fields=True, include_attributes=False)
        hashes[fn.name] = hashlib.sha256(dumped.encode()).hexdigest()
    return hashes


def method_source(name: str) -> str:
    source = ENV.read_text()
    tree = ast.parse(source)
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


def check_v13_20_logic_contract() -> None:
    hashes = method_logic_hashes(ENV)
    for name, expected in EXPECTED_V13_20_AST.items():
        req(hashes.get(name) == expected, f"v13.20 logic preserved: {name}")


def check_config_contract() -> None:
    cfg = CFG.read_text()
    required = (
        'task: "isaaclabanymalroom_anymal_visible_arrow_random_objects_8m_v13_45_balanced_randomization_proprio"',
        "train_ratio: 256",
        "size: [128, 128]",
        "camera_horizontal_fov_deg: 90.0",
        "camera_horizontal_aperture: 20.955",
        "camera_pitch_deg: 0.0",
        "camera_offset_x: 0.510",
        "camera_offset_y: 0.0",
        "camera_offset_z: 0.015",
        "time_limit: 500",
        "action_count: 11",
        "lateral_speed: 0.25",
        "goal_min_distance: 0.50",
        "goal_max_distance: 4.00",
        "goal_bearing_max_deg: 180.0",
        "goal_relative_yaw_max_deg: 180.0",
        "position_tolerance: 0.12",
        "yaw_tolerance_deg: 5.0",
        "success_hold_steps: 5",
        "geometry_yaw_metres_per_rad: 3.0",
        "geometry_rho0: 1.00",
        "geometry_k_alpha: 0.5",
        "geometry_k_beta: 0.25",
        "geometry_k_yaw: 1.0",
        "proximity_scale: 0.5",
        "proximity_sigma: 2.0",
        "in_tolerance_reward: 0.5",
        "arrow_start_clearance: 0.35",
        "balanced_randomization: true",
        "start_sampling_attempts: 512",
        "start_stratified_attempts: 64",
        "start_grid_bins: 4",
        "start_yaw_bins: 8",
        "goal_sampling_attempts: 1024",
        "goal_stratified_attempts: 64",
        "goal_distance_bins: 4",
        "goal_bearing_bins: 8",
        "goal_yaw_bins: 8",
        "object_yaw_bins: 8",
        "proprioception_enabled: false",
        "proprio_linear_velocity_scale: 2.0",
        "proprio_angular_velocity_scale: 0.25",
        "proprio_joint_position_scale: 1.0",
        "proprio_joint_velocity_scale: 0.05",
        "proprio_clip: 5.0",
        "agent_decimation: 20",
        "wall_guard_terminal: false",
    )
    for literal in required:
        req(literal in cfg, f"required config: {literal}")

    for field in (
        "fall_penalty",
        "room_exit_penalty",
        "wall_guard_penalty",
        "progress_scale",
        "step_penalty",
        "success_bonus",
        "yaw_progress_scale",
        "joint_pose_scale",
        "collision_penalty",
    ):
        req(
            re.search(rf"^{re.escape(field)}\s*:", cfg, flags=re.MULTILINE) is None,
            f"unused/extra reward config absent: {field}",
        )



def check_action_extension() -> None:
    env = ENV.read_text()
    cfg = CFG.read_text()
    req("action_count: 11" in cfg, "action space has 11 actions")
    req("lateral_speed: 0.25" in cfg, "lateral speed is 0.25 m/s")
    req('self._action_count = int(_get(config, "action_count", 11))' in env, "11-action default")
    req('if self._action_count != 11:' in env, "11-action runtime contract")

    table = method_source("__init__")
    original_order = (
        '[forward_speed, 0.00, 0.00]',
        '[-backward_speed, 0.00, 0.00]',
        '[arc_forward_speed, 0.00, arc_yaw_rate]',
        '[arc_forward_speed, 0.00, -arc_yaw_rate]',
        '[0.00, 0.00, coarse_yaw_rate]',
        '[0.00, 0.00, -coarse_yaw_rate]',
        '[0.00, 0.00, 0.00]',
        '[0.00, 0.00, fine_yaw_rate]',
        '[0.00, 0.00, -fine_yaw_rate]',
        '[0.00, lateral_speed, 0.00]',
        '[0.00, -lateral_speed, 0.00]',
    )
    positions = [table.index(token) for token in original_order]
    req(positions == sorted(positions), "original nine actions preserved; strafes appended")
    req('num_classes=self._action_count' in method_source("step"), "one-hot logging follows action count")
    for name in (
        'left_arc', 'right_arc', 'strafe_left', 'strafe_right',
    ):
        req(f'"{name}"' in method_source("step"), f"action log present: {name}")


def check_training_budget() -> None:
    cfg = CFG.read_text()
    req("train_ratio: 256" in cfg, "train ratio increased to 256")
    for name in ("run_anymal_headless.sh", "run_anymal_gui.sh"):
        text = (HERE / name).read_text()
        req(
            'BATCH_LENGTH="${BATCH_LENGTH:-64}"' in text,
            f"{name}: batch length defaults to 64",
        )


def check_camera_contract() -> None:
    env = ENV.read_text()
    cfg = CFG.read_text()
    common = (HERE / "common.sh").read_text()
    req('IMAGE_SIZE="${IMAGE_SIZE:-128}"' in common, "launcher defaults to 128x128")
    req("camera_horizontal_fov_deg: 90.0" in cfg, "default HFOV is 90 deg")
    req("camera_pitch_deg: 0.0" in cfg, "default camera remains level")
    req("def _camera_mount_from_config(" in env, "single camera-mount helper exists")
    req(
        "camera_pos_xyz, camera_ros_wxyz, _, camera_pitch_deg = _camera_mount_from_config(config)"
        in env,
        "render camera uses shared mount helper",
    )
    req(
        ") = _camera_mount_from_config(config)" in method_source("__init__"),
        "reward geometry uses shared mount helper",
    )
    req("CAMERA_REL_QUAT_XYZW" not in env, "duplicate fixed camera quaternion removed")
    req("CAMERA_REL_POS_XYZ" not in env, "duplicate fixed camera position removed")

    aperture = 20.955
    focal = aperture / (2.0 * math.tan(math.radians(45.0)))
    recovered = math.degrees(2.0 * math.atan(aperture / (2.0 * focal)))
    req(abs(recovered - 90.0) < 1e-9, "90-deg focal-length calculation closes")


def check_scene_and_sampler() -> None:
    env = ENV.read_text()
    cfg = CFG.read_text()
    req("ROOM_HALF = 4.00" in env, "room interior is 8 x 8 m")
    for literal in (
        "randomize_objects: true",
        "randomize_object_yaw: true",
        "object_wall_clearance: 0.45",
        "object_min_clearance: 0.70",
        "object_sampling_attempts: 512",
        "arrow_start_clearance: 0.35",
        "balanced_randomization: true",
        "start_grid_bins: 4",
        "start_yaw_bins: 8",
        "goal_distance_bins: 4",
        "goal_bearing_bins: 8",
        "goal_yaw_bins: 8",
    ):
        req(literal in cfg, f"scene/sampler config: {literal}")

    sample_goal = method_source("_sample_goal_for_ids")
    for token in (
        "start_base_xy_all, _, _ = self._robot_state()",
        "start_arrow_centerline_dist",
        "start_required = half_lateral + self._arrow_start_clearance",
        "arrow_start_ok = start_arrow_centerline_dist >= start_required",
        "heading_attempts.scatter_add_",
        "heading_rejections.scatter_add_",
        "heading_accepts.scatter_add_",
        "reject_wall +=",
        "reject_object +=",
        "reject_start +=",
    ):
        req(token in sample_goal, f"goal sampler logic: {token}")

    for token in (
        "def _mixed_task_seed(",
        "def _task_random(",
        "self._task_episode_count",
        '"object_order"',
        "placement_order = torch.argsort",
        "self._task_stratum_index(",
        "self._goal_stratified_attempts",
        "self._start_stratified_attempts",
    ):
        req(token in env, f"balanced independent randomization: {token}")
    req("self._task_rng" not in env, "shared task RNG removed")

    reset_ids = method_source("_reset_ids")
    req(
        reset_ids.index("self._reset_props(env_ids)")
        < reset_ids.index("self._place_robot_randomly(env_ids)")
        < reset_ids.index("self._sample_goal_for_ids(env_ids)"),
        "reset order is objects -> robot -> goal/arrow",
    )
    req(
        reset_ids.index("self._sample_goal_for_ids(env_ids)")
        < reset_ids.index("self._advance_task_randomization(env_ids)"),
        "episode randomization counter advances only after the complete reset",
    )


def check_reward_source() -> None:
    step = method_source("step")
    req('r_proximity = self._proximity_scale * torch.exp(' in step, "v13.20 proximity reward")
    req(
        '-geometry_post["error"] / max(self._proximity_sigma, 1e-6)' in step,
        "v13.20 geometric error drives reward",
    )
    req(
        "r_in_tolerance = self._in_tolerance_reward * within_pose.float()" in step,
        "precise occupancy reward",
    )
    req(
        '''reward = torch.where(
            done | true_terminal,
            torch.zeros_like(r_proximity),
            r_proximity + r_in_tolerance,
        )'''
        in step,
        "scalar reward is exactly proximity plus precise occupancy",
    )
    for forbidden in (
        "r_joint_pose",
        "yaw_progress",
        "r_position_progress",
        "r_potential",
        "success_bonus",
        "step_penalty",
        "collision_penalty",
        "r_fall_penalty",
        "r_wall_penalty",
        "r_room_exit_penalty",
    ):
        req(forbidden not in step, f"no extra/dead reward logic: {forbidden}")


def check_cleanup_and_logging() -> None:
    env = ENV.read_text()
    step = method_source("step")
    for dead in (
        "_previous_position_error",
        "_previous_yaw_error",
        "_previous_bearing_error",
        "_fall_penalty",
        "_room_exit_penalty",
        "_wall_guard_penalty",
    ):
        req(dead not in env, f"dead field removed: {dead}")

    for line in (
        'data["log_position_error"] = (position_error * terminal_f).unsqueeze(-1)',
        'data["log_goal_x"] = (self._goal_xy[:, 0] * terminal_f).unsqueeze(-1)',
        'data["log_goal_y"] = (self._goal_xy[:, 1] * terminal_f).unsqueeze(-1)',
        'data["log_root_z"] = (root_z * terminal_f).unsqueeze(-1)',
    ):
        req(line in step, f"sum-based logger receives terminal-masked value: {line}")

    for key in (
        "log_goal_sample_attempts",
        "log_goal_sample_reject_wall",
        "log_goal_sample_reject_object",
        "log_goal_sample_reject_start",
        "log_goal_arrow_start_distance",
        "log_goal_heading_attempts_",
        "log_goal_heading_rejections_",
        "log_goal_heading_accepts_",
    ):
        req(key in step, f"sampler diagnostic emitted: {key}")

    plot = (HERE / "plot_training.py").read_text()
    req("goal_sampling_counts.png" in plot, "sampler count plot exists")
    req(
        "goal_sampling_rejection_by_heading.png" in plot,
        "heading-dependent rejection plot exists",
    )
    req("log_reward_fall_penalty" not in plot, "dead fall-penalty plot removed")
    req("log_reward_wall_penalty" not in plot, "dead wall-penalty plot removed")
    req("log_reward_room_exit_penalty" not in plot, "dead exit-penalty plot removed")


def check_arrow_and_rgb_routing() -> None:
    env = ENV.read_text()
    cfg = CFG.read_text()
    for literal in (
        "show_goal_arrow: true",
        'arrow_target_anchor: "base"',
        "arrow_length: 1.60",
        "arrow_head_length: 0.55",
        "arrow_shaft_width: 0.26",
        "arrow_head_base_width: 0.68",
        "arrow_head_mid_width: 0.44",
        "arrow_head_tip_width: 0.22",
    ):
        req(literal in cfg, f"arrow config: {literal}")
    req(
        "goal_base_xy, goal_base_yaw = self._goal_base_pose()" in env,
        "visible arrow uses desired robot-base pose",
    )
    req(
        'encoder:\n  cnn_keys: "image"\n  mlp_keys: "^$"' in cfg,
        "encoder is RGB-only",
    )
    req(
        'decoder:\n  cnn_keys: "image"\n  mlp_keys: "^$"' in cfg,
        "auxiliary goal decoder is off by default",
    )


def check_proprioception_routing() -> None:
    env = ENV.read_text()
    cfg = CFG.read_text()
    build_space = method_source("_build_observation_space")
    make_data = method_source("_make_data")
    proprio = method_source("_proprioceptive_observation")
    for literal in (
        "proprioception_enabled: false",
        "proprio_linear_velocity_scale: 2.0",
        "proprio_angular_velocity_scale: 0.25",
        "proprio_joint_position_scale: 1.0",
        "proprio_joint_velocity_scale: 0.05",
        "proprio_clip: 5.0",
    ):
        req(literal in cfg, f"proprio config: {literal}")
    req('spaces["proprio"]' in build_space, "proprio observation space is conditional")
    req('data["proprio"] = self._proprioceptive_observation()' in make_data, "proprio data routing is conditional")
    for token in (
        'getattr(data, "root_lin_vel_b", None)',
        'getattr(data, "root_ang_vel_b", None)',
        'getattr(data, "projected_gravity_b", None)',
        "joint_pos - default_joint_pos",
        "joint_vel * self._proprio_joint_velocity_scale",
    ):
        req(token in proprio, f"proprio component: {token}")
    for forbidden in ("_goal_xy", "_goal_yaw", "_goal_conditioned_observation", "base_xy"):
        req(forbidden not in proprio, f"no goal/global-pose leak in proprio: {forbidden}")


def check_install_and_launchers() -> None:
    install = (HERE / "install.py").read_text()
    common = (HERE / "common.sh").read_text()
    req(
        'decoder_overrides+=(env.decoder.mlp_keys="^$")' not in common,
        "Hydra-hostile disabled-decoder override absent",
    )
    req("env.decoder.mlp_keys=goal_vec" in common, "optional goal decoder override")
    req("env.encoder.mlp_keys=proprio" in common, "proprio routed into MultiEncoder")
    req("env.decoder.mlp_keys=proprio" in common, "proprio reconstructed by MultiDecoder")
    req('env.task_seed="$TASK_SEED"' in common, "task seed is explicit")
    req('TASK_SEED="${TASK_SEED:-$SEED}"' in common, "task seed defaults reproducibly")
    req("_preload_saved_run_hyperparameters" in common, "evaluation can recover run parameters")
    req("parameters.json" in common, "minimal parameter file is evaluation source")
    req("TASK_SEED is never inherited" in common, "held-out task seed is not inherited")
    req("patch_dreamer_decoder_key_scales" in install, "optional decoder patch retained")
    req("patch_trainer_run_checkpoints" in install, "three-checkpoint patch installed")
    req("patch_train_run_artifacts" in install, "resolved config/final checkpoint patch installed")
    req('CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"' in common, "checkpoint interval configurable")
    req('BEST_WINDOW_EPISODES="${BEST_WINDOW_EPISODES:-100}"' in common, "rolling selection window configurable")
    req('BEST_MIN_EPISODES="${BEST_MIN_EPISODES:-100}"' in common, "selection warmup configurable")
    for removed in ("CHECKPOINT_ARCHIVE", "BEST_CHECK_EVERY", "BEST_POSITION_SCALE", "BEST_YAW_SCALE"):
        req(removed not in common, f"removed checkpoint option absent: {removed}")

    for name in (
        "run_anymal_headless.sh",
        "run_anymal_gui.sh",
        "run_anymal_eval.sh",
        "run_anymal_eval_gui.sh",
    ):
        launcher = (HERE / name).read_text()
        req('"${decoder_overrides[@]}"' in launcher, f"{name}: decoder overrides forwarded")
        req('"${encoder_overrides[@]}"' in launcher, f"{name}: encoder overrides forwarded")
        req('"${randomization_overrides[@]}"' in launcher, f"{name}: randomization overrides forwarded")
        req('"${arrow_overrides[@]}"' in launcher, f"{name}: arrow overrides forwarded")
    for name in ("run_anymal_eval.sh", "run_anymal_eval_gui.sh"):
        req("R2DREAMER_LOAD_RUN_CONFIG=1" in (HERE / name).read_text(), f"{name}: saved run defaults enabled")
    for name in ("run_anymal_headless.sh", "run_anymal_gui.sh"):
        launcher = (HERE / name).read_text()
        req('"${checkpoint_overrides[@]}"' in launcher, f"{name}: checkpoint overrides forwarded")
        req("prepare_training_run" in launcher, f"{name}: unique run directory prepared")
        req("initialize_training_run" in launcher, f"{name}: parameters recorded")
        req("execute_training_run" in launcher, f"{name}: final status/plots handled")

    for name in (
        "run_anymal_proprio_headless.sh",
        "run_anymal_proprio_gui.sh",
        "run_anymal_proprio_eval.sh",
        "run_anymal_proprio_eval_gui.sh",
    ):
        req("PROPRIOCEPTION_ENABLED=1" in (HERE / name).read_text(), f"{name}: dedicated proprio launcher")
    req("run_anymal_proprio_gui.sh" in (HERE / "run_anymal_proprio_smoke_test.sh").read_text(), "proprio runtime smoke launcher exists")


def check_run_management_and_best_checkpoints() -> None:
    common = (HERE / "common.sh").read_text()
    manager = (HERE / "run_manager.py").read_text()
    checkpoints = (HERE / "camera_pose_checkpoints.py").read_text()
    plots = (HERE / "plot_training.py").read_text()
    readme = (HERE / "README.md").read_text()

    for token in (
        'LOG_ROOT="${LOG_ROOT:-$R2DREAMER_REPO/logdir/isaaclab_anymal_v13_45}"',
        'stamp="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"',
        'mkdir "$candidate"',
        '"$VARIANT_ROOT/latest_run"',
        "ALLOW_EXISTING_RUN",
        "prepare_training_run",
        'run_manager.py" init',
        'run_manager.py" finish',
        'plot_training.py" --logdir "$LOGDIR"',
        "default_eval_output_dir",
    ):
        req(token in common, f"run-directory management: {token}")

    for token in (
        "parameters.yaml",
        "parameters.json",
        "run_status.json",
        "latest_link",
        'logdir / "plots"',
        'logdir / "gifs"',
        'logdir / "evaluations"',
    ):
        req(token in manager, f"minimal run artifact: {token}")
    for removed in (
        "system_info.txt",
        "python_packages.txt",
        "runtime_environment.json",
        "r2dreamer_git.yaml",
        "r2dreamer_git_diff.patch",
        "source_snapshot",
        "unresolved_config.yaml",
    ):
        req(removed not in manager, f"removed metadata absent from manager: {removed}")

    for token in (
        "best_success.pt",
        "best_reward.pt",
        "latest.pt",
        "checkpoint_summary.json",
        "summary.success_key",
        "summary.reward_key",
        "_torch_save_atomic",
        "_clone_atomic",
        "maximum rolling success rate",
        "maximum rolling mean episode return",
    ):
        req(token in checkpoints, f"three-checkpoint contract: {token}")
    for removed in (
        "best_task.pt",
        "best_validation.pt",
        "archive",
        "selection_history.jsonl",
        "manager_state.json",
    ):
        req(removed not in checkpoints, f"removed checkpoint artifact absent: {removed}")

    for token in (
        '++trainer.checkpoint_every="$CHECKPOINT_EVERY"',
        '++trainer.best_checkpoint_window="$BEST_WINDOW_EPISODES"',
        '++trainer.best_checkpoint_min_episodes="$BEST_MIN_EPISODES"',
    ):
        req(token in common, f"checkpoint Hydra override: {token}")
    req('kind="${CHECKPOINT_KIND:-best_success}"' in common, "evaluation defaults to best success")
    req('candidate="$run_dir/best_success.pt"' in common, "best-success resolver")
    req('candidate="$run_dir/best_reward.pt"' in common, "best-reward resolver")
    req('candidate="$run_dir/latest.pt"' in common, "latest resolver")
    req("checkpoint_summary.json" in plots, "plots summarize the retained checkpoints")

    for removed_file in (
        "rank_checkpoint_evaluations.py",
        "run_checkpoint_validation_sweep.sh",
        "run_rank_checkpoint_evaluations.sh",
        "run_select_best_validation.sh",
    ):
        req(not (HERE / removed_file).exists(), f"removed helper absent: {removed_file}")
    req((HERE / "run_list_runs.sh").is_file(), "run listing helper exists")
    req((HERE / "checkpoint_manager_sanity.py").is_file(), "checkpoint sanity exists")
    req((HERE / "run_manager_sanity.py").is_file(), "run manager sanity exists")

    for token in ("latest.pt", "best_success.pt", "best_reward.pt"):
        req(token in readme, f"README documents {token}")
    req("parameters.yaml" in readme and "resolved_config.yaml" in readme, "README documents minimal parameter files")
    for removed in ("python_packages.txt", "system_info.txt", "best_validation.pt", "periodic archive"):
        req(removed not in readme, f"README omits removed feature: {removed}")


def check_manifest() -> None:
    manifest = HERE / "BUNDLE_MANIFEST.sha256"
    req(manifest.is_file(), "bundle manifest exists")
    listed: set[str] = set()
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        rel = rel.strip()
        listed.add(rel)
        path = HERE / rel
        req(path.is_file(), f"manifest file exists: {rel}")
        req(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"manifest hash: {rel}")

    expected = {
        path.relative_to(HERE).as_posix()
        for path in HERE.rglob("*")
        if path.is_file()
        and path != manifest
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    req(listed == expected, "manifest covers every distributed file exactly once")


def main() -> None:
    check_v13_20_logic_contract()
    check_config_contract()
    check_action_extension()
    check_training_budget()
    check_camera_contract()
    check_scene_and_sampler()
    check_reward_source()
    check_cleanup_and_logging()
    check_arrow_and_rgb_routing()
    check_proprioception_routing()
    check_install_and_launchers()
    check_run_management_and_best_checkpoints()

    for path in HERE.rglob("*.py"):
        py_compile.compile(str(path), doraise=True)
    print("[VERIFY] OK: Python syntax")
    for path in HERE.glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    print("[VERIFY] OK: shell syntax")

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
    print("[VERIFY] OK: reward, arrow, balanced randomization, proprioception, camera mount, action-space, installer, launcher, run-manager, checkpoint-manager sanity")

    check_manifest()
    print("[VERIFY] ALL OFFLINE CHECKS PASSED")


if __name__ == "__main__":
    main()
