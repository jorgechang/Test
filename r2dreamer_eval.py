#!/usr/bin/env python3
"""Evaluate a trained NM512/r2dreamer DreamerV3 checkpoint on Isaac Lab.

This is evaluation-only: no replay buffer is created and no optimizer step is
performed. The policy uses ``eval=True`` so Dreamer selects the mode of its
action distribution instead of sampling exploratory actions.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime, timezone
import hashlib
import math

from isaaclab.app import AppLauncher


def _find_isaaclab_root(start: pathlib.Path) -> pathlib.Path:
    for candidate in (start, *start.parents):
        if (candidate / "isaaclab.sh").is_file():
            return candidate
    raise RuntimeError("Could not find IsaacLab root containing isaaclab.sh")


def _remove_override(args: list[str], key: str) -> list[str]:
    prefix = key + "="
    return [arg for arg in args if not arg.startswith(prefix)]


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DreamerV3 checkpoint on the visible-arrow camera-pose task.",
        conflict_handler="resolve",
    )
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--video_steps", type=int, default=64)
    parser.add_argument("--output_dir", type=pathlib.Path, default=None)
    parser.add_argument("--no_video", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    if args_cli.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be > 0")

    # Isaac Lab must be launched before importing the patched environment.
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    here = pathlib.Path(__file__).resolve().parent
    isaaclab_root = _find_isaaclab_root(here)
    repo = isaaclab_root / "r2dreamer"
    if not (repo / "train.py").is_file():
        raise RuntimeError(f"NM512/r2dreamer checkout not found at {repo}")

    sys.path.insert(0, str(repo))

    import hydra
    import torch
    from omegaconf import OmegaConf

    import isaaclab_runtime

    isaaclab_runtime.set_runtime(app_launcher, simulation_app)

    # Compose exactly the same upstream config family as train.py, with the
    # environment/model overrides supplied by the shell launcher.
    hydra_args = _remove_override(hydra_args, "env.env_num")
    hydra_args = _remove_override(hydra_args, "env.eval_episode_num")
    hydra_args = _remove_override(hydra_args, "model.compile")
    hydra_args.extend(
        [
            f"env.env_num={int(args_cli.num_envs)}",
            "env.eval_episode_num=0",
            "model.compile=false",
        ]
    )

    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str((repo / "configs").resolve()),
    ):
        config = hydra.compose(config_name="configs", overrides=hydra_args)

    import tools
    from dreamer import Dreamer
    from envs import make_envs

    tools.set_seed_everywhere(config.seed)
    if bool(config.deterministic_run):
        tools.enable_deterministic_run()

    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args_cli.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        task_seed = int(config.env.task_seed)
        checkpoint_parent = checkpoint_path.parent
        run_dir = checkpoint_parent.parent if checkpoint_parent.name == "checkpoints" else checkpoint_parent
        base = run_dir / "evaluations" / f"{checkpoint_path.stem}_task{task_seed}_{stamp}"
        base.parent.mkdir(parents=True, exist_ok=True)
        output_dir = base
        suffix = 2
        while True:
            try:
                output_dir.mkdir()
                break
            except FileExistsError:
                output_dir = base.with_name(f"{base.name}_{suffix}")
                suffix += 1
    else:
        output_dir = args_cli.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True), encoding="utf-8"
    )
    (output_dir / "hydra_overrides.txt").write_text(
        "\n".join(hydra_args) + ("\n" if hydra_args else ""), encoding="utf-8"
    )
    (output_dir / "evaluation_request.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "episodes": int(args_cli.episodes),
                "num_envs": int(args_cli.num_envs),
                "task_seed": int(config.env.task_seed),
                "seed": int(config.seed),
                "video_steps": int(args_cli.video_steps),
                "deterministic_actor": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # The GIF logger patch installed by the visible-arrow camera-pose bundle understands
    # eval_video and eval_open_loop.
    if not args_cli.no_video:
        os.environ.setdefault("R2DREAMER_SAVE_GIFS", "1")
        os.environ.setdefault("R2DREAMER_GIF_FPS", "8")
        os.environ.setdefault("R2DREAMER_GIF_SCALE", "2")
        os.environ.setdefault("R2DREAMER_GIF_MAX_BATCH", "1")
        os.environ.setdefault("R2DREAMER_GIF_NAMES", "eval_video,eval_open_loop")

    print(f"[EVAL] checkpoint: {checkpoint_path}")
    print(f"[EVAL] output:     {output_dir}")
    print(f"[EVAL] episodes:   {args_cli.episodes}")
    print(f"[EVAL] envs:       {args_cli.num_envs}")
    print("[EVAL] policy:     deterministic actor mode (eval=True)")

    envs = None
    try:
        print("[EVAL] Creating environment...")
        envs, _, obs_space, act_space = make_envs(config.env)

        print("[EVAL] Building DreamerV3...")
        agent = Dreamer(config.model, obs_space, act_space).to(config.device)

        # This is the format saved by current NM512/r2dreamer/train.py.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=config.device,
            weights_only=False,
        )
        state_dict = checkpoint.get("agent_state_dict", checkpoint)
        agent.load_state_dict(state_dict, strict=True)

        # act() intentionally uses frozen/shared inference copies. Recreate the
        # aliases from the just-loaded live modules so evaluation cannot use a
        # stale copy even if the checkpoint format changes in the future.
        agent.clone_and_freeze()
        agent.eval()
        print("[EVAL] Checkpoint loaded successfully.")

        device = agent.device
        num_envs = int(envs.env_num)

        done = torch.ones(num_envs, dtype=torch.bool, device=device)
        state = agent.get_initial_state(num_envs)
        action = state["prev_action"].clone()

        running_return = torch.zeros(num_envs, dtype=torch.float32, device=device)
        running_length = torch.zeros(num_envs, dtype=torch.int64, device=device)
        running_in_tolerance_steps = torch.zeros(num_envs, dtype=torch.float32, device=device)
        running_wall_steps = torch.zeros(num_envs, dtype=torch.float32, device=device)

        returns: list[float] = []
        lengths: list[int] = []
        successes: list[float] = []
        final_position_errors: list[float] = []
        min_position_errors: list[float] = []
        final_yaw_errors: list[float] = []
        min_yaw_errors: list[float] = []
        falls: list[float] = []
        collisions: list[float] = []
        wall_collisions: list[float] = []
        obstacle_collisions: list[float] = []
        room_exits: list[float] = []
        timeouts: list[float] = []
        in_tolerance_steps: list[float] = []
        max_hold_steps: list[float] = []
        wall_steps_per_episode: list[float] = []

        action_counts = torch.zeros(agent.act_dim, dtype=torch.int64)

        # Save the first completed evaluation trajectory for both normal video
        # and posterior/open-loop world-model visualization.
        video_cache = []
        video_finished = False

        while len(returns) < args_cli.episodes:
            prev_done = done.clone()
            action_for_transition = action.detach().clone()

            trans, step_done = envs.step(action_for_transition, prev_done)
            trans = trans.to(device, non_blocking=True)
            done = step_done.to(device)

            # Keep action aligned with the observation exactly as trainer.eval().
            trans["action"] = action_for_transition

            active = ~prev_done
            running_return += trans["reward"][:, 0]
            running_length += active.to(torch.int64)
            if "log_in_tolerance_steps" in trans:
                running_in_tolerance_steps += trans["log_in_tolerance_steps"][:, 0]
            if "log_wall_collision" in trans:
                running_wall_steps += trans["log_wall_collision"][:, 0]

            if active.any():
                action_index = torch.argmax(action_for_transition[active], dim=-1).cpu()
                action_counts += torch.bincount(action_index, minlength=agent.act_dim)

            if not video_finished and len(video_cache) < int(args_cli.video_steps):
                video_cache.append(trans[:1].clone())

            completed_ids = done.nonzero(as_tuple=False).squeeze(-1)
            for idx_t in completed_ids:
                idx = int(idx_t.item())
                if len(returns) >= args_cli.episodes:
                    break

                success = float(trans["log_success"][idx, 0].item()) if "log_success" in trans else 0.0
                fall = float(trans["log_fall"][idx, 0].item()) if "log_fall" in trans else 0.0
                wall_step_count = float(running_wall_steps[idx].item())
                wall_collision = float(wall_step_count > 0.0)
                collision = wall_collision
                obstacle_collision = float(trans["log_obstacle_collision"][idx, 0].item()) if "log_obstacle_collision" in trans else 0.0
                room_exit = float(trans["log_room_exit"][idx, 0].item()) if "log_room_exit" in trans else 0.0
                timeout = (
                    float(trans["log_timeout"][idx, 0].item())
                    if "log_timeout" in trans
                    else float((fall < 0.5) and (room_exit < 0.5))
                )

                returns.append(float(running_return[idx].item()))
                lengths.append(int(running_length[idx].item()))
                successes.append(success)
                falls.append(fall)
                collisions.append(collision)
                wall_collisions.append(wall_collision)
                obstacle_collisions.append(obstacle_collision)
                room_exits.append(room_exit)
                timeouts.append(timeout)
                in_tolerance_steps.append(float(running_in_tolerance_steps[idx].item()))
                wall_steps_per_episode.append(wall_step_count)
                max_hold_steps.append(
                    float(trans["log_max_hold_steps"][idx, 0].item())
                    if "log_max_hold_steps" in trans else 0.0
                )

                if "log_final_position_error" in trans:
                    final_position_errors.append(float(trans["log_final_position_error"][idx, 0].item()))
                elif "log_position_error" in trans:
                    final_position_errors.append(float(trans["log_position_error"][idx, 0].item()))
                if "log_min_position_error" in trans:
                    min_position_errors.append(float(trans["log_min_position_error"][idx, 0].item()))
                if "log_final_yaw_error_deg" in trans:
                    final_yaw_errors.append(float(trans["log_final_yaw_error_deg"][idx, 0].item()))
                elif "log_yaw_error_deg" in trans:
                    final_yaw_errors.append(float(trans["log_yaw_error_deg"][idx, 0].item()))
                if "log_min_yaw_error_deg" in trans:
                    min_yaw_errors.append(float(trans["log_min_yaw_error_deg"][idx, 0].item()))

                print(
                    f"[EVAL] episode {len(returns):4d}/{args_cli.episodes}: "
                    f"return={returns[-1]:8.3f} length={lengths[-1]:4d} "
                    f"success={int(success > 0.5)}"
                )

                running_return[idx] = 0.0
                running_length[idx] = 0
                running_in_tolerance_steps[idx] = 0.0
                running_wall_steps[idx] = 0.0

                if idx == 0 and not video_finished:
                    video_finished = True

            # act(eval=True) selects the mode of the actor distribution.
            action, state = agent.act(trans, state, eval=True)

        position_scale_m = max(float(getattr(config.env, "position_tolerance", 0.12)), 1.0e-9)
        yaw_scale_deg = max(float(getattr(config.env, "yaw_tolerance_deg", 5.0)), 1.0e-9)
        paired_pose_errors = [
            math.sqrt((pos / position_scale_m) ** 2 + (yaw / yaw_scale_deg) ** 2)
            for pos, yaw in zip(final_position_errors, final_yaw_errors)
        ]
        failure_rate = _mean(
            [min(fall + room_exit, 1.0) for fall, room_exit in zip(falls, room_exits)]
        )
        total_episode_steps = max(float(sum(lengths)), 1.0)
        wall_exposure_percent = 100.0 * sum(wall_steps_per_episode) / total_episode_steps
        summary = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "episodes": len(returns),
            "num_envs": num_envs,
            "seed": int(config.seed),
            "task_seed": int(config.env.task_seed),
            "deterministic_actor": True,
            "mean_return": _mean(returns),
            "mean_episode_length": _mean([float(x) for x in lengths]),
            "success_rate": _mean(successes),
            "precise_tolerance_occupancy_percent": (
                100.0 * sum(in_tolerance_steps) / max(float(sum(lengths)), 1.0)
            ),
            "mean_max_hold_steps": _mean(max_hold_steps),
            "mean_normalized_final_pose_error": (
                _mean(paired_pose_errors) if paired_pose_errors else 1.0e9
            ),
            "selection_position_scale_m": position_scale_m,
            "selection_yaw_scale_deg": yaw_scale_deg,
            "failure_rate": failure_rate,
            "wall_exposure_percent": wall_exposure_percent,
            "timeout_rate": _mean(timeouts),
            "fall_rate": _mean(falls),
            "collision_rate": _mean(collisions),
            "wall_collision_rate": _mean(wall_collisions),
            "obstacle_collision_rate": _mean(obstacle_collisions),
            "room_exit_rate": _mean(room_exits),
            "mean_final_position_error_m": _mean(final_position_errors) if final_position_errors else None,
            "mean_min_position_error_m": _mean(min_position_errors) if min_position_errors else None,
            "mean_final_yaw_error_deg": _mean(final_yaw_errors) if final_yaw_errors else None,
            "mean_min_yaw_error_deg": _mean(min_yaw_errors) if min_yaw_errors else None,
            "action_counts": [int(x) for x in action_counts.tolist()],
        }
        if isinstance(checkpoint, dict) and "checkpoint_metadata" in checkpoint:
            summary["source_checkpoint_metadata"] = checkpoint["checkpoint_metadata"]

        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

        print("\n[EVAL] ==================== SUMMARY ====================")
        print(f"[EVAL] success rate:           {summary['success_rate'] * 100:7.2f}%")
        print(f"[EVAL] mean return:            {summary['mean_return']:7.3f}")
        print(f"[EVAL] precise occupancy:      {summary['precise_tolerance_occupancy_percent']:7.3f}%")
        print(f"[EVAL] mean max hold:          {summary['mean_max_hold_steps']:7.2f} steps")
        print(f"[EVAL] normalized pose error:  {summary['mean_normalized_final_pose_error']:7.3f}")
        print(f"[EVAL] wall exposure:          {summary['wall_exposure_percent']:7.3f}%")
        print(f"[EVAL] mean episode length:    {summary['mean_episode_length']:7.2f}")
        if summary["mean_final_position_error_m"] is not None:
            print(f"[EVAL] final position error:   {summary['mean_final_position_error_m']:7.3f} m")
        if summary["mean_min_position_error_m"] is not None:
            print(f"[EVAL] min position reached:   {summary['mean_min_position_error_m']:7.3f} m")
        if summary["mean_final_yaw_error_deg"] is not None:
            print(f"[EVAL] final yaw error:        {summary['mean_final_yaw_error_deg']:7.3f} deg")
        if summary["mean_min_yaw_error_deg"] is not None:
            print(f"[EVAL] min yaw reached:        {summary['mean_min_yaw_error_deg']:7.3f} deg")
        if any(falls):
            print(f"[EVAL] fall rate:              {summary['fall_rate'] * 100:7.2f}%")
        if any(collisions):
            print(f"[EVAL] collision rate:         {summary['collision_rate'] * 100:7.2f}%")
            print(f"[EVAL]   wall collisions:      {summary['wall_collision_rate'] * 100:7.2f}%")
            print("[EVAL] object-contact reset:    disabled (contact/climbing allowed)")
        if any(room_exits):
            print(f"[EVAL] room-exit rate:         {summary['room_exit_rate'] * 100:7.2f}%")
        print(f"[EVAL] timeout rate:           {summary['timeout_rate'] * 100:7.2f}%")
        print(f"[EVAL] action counts:          {summary['action_counts']}")
        print(f"[EVAL] summary saved:          {summary_path}")

        if not args_cli.no_video and len(video_cache) >= 6:
            cache = torch.stack(video_cache, dim=1)
            logger = tools.Logger(output_dir)

            # Save the true camera trajectory first.
            logger.video("eval_video", tools.to_np(cache["image"]))

            if agent.rep_loss == "dreamer":
                initial = agent.get_initial_state(1)
                pred = agent.video_pred(
                    cache.clone(),
                    (initial["stoch"], initial["deter"]),
                )
                logger.video("eval_open_loop", tools.to_np(pred))
            else:
                print(
                    "[EVAL] Skipping eval_open_loop: this checkpoint has no "
                    "Dreamer pixel decoder."
                )
            logger.write(0)
            print(f"[EVAL] evaluation GIFs:        {output_dir / 'gifs'}")

    finally:
        if envs is not None:
            try:
                envs.close()
            except Exception as exc:
                print(f"[EVAL][WARNING] env close failed: {exc}")


if __name__ == "__main__":
    main()
