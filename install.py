#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import re
import shutil

HERE = pathlib.Path(__file__).resolve().parent
PAYLOAD = HERE / "payload"


def find_root(start: pathlib.Path) -> pathlib.Path:
    for candidate in (start, *start.parents):
        if (candidate / "isaaclab.sh").is_file():
            return candidate
    raise RuntimeError("Could not find IsaacLab root containing isaaclab.sh")


def find_repo(root: pathlib.Path, explicit: str | None) -> pathlib.Path:
    repo = pathlib.Path(explicit).expanduser().resolve() if explicit else root / "r2dreamer"
    if (repo / "train.py").is_file() and (repo / "envs" / "__init__.py").is_file():
        return repo
    raise RuntimeError(f"Not an NM512/r2dreamer checkout: {repo}")


def _insert_suite_route(text: str, suite: str, helper: str) -> str:
    if f'suite == "{suite}"' in text or f"suite == '{suite}'" in text:
        return text
    anchor = re.search(
        r'(?m)^(\s*)if suite == ["\']isaaclab["\']:\s*\n\s+return _make_isaaclab_envs\(config\)\s*$',
        text,
    )
    if not anchor:
        raise RuntimeError("Could not find upstream IsaacLab suite route in envs/__init__.py")
    indent = anchor.group(1)
    route = f'{indent}if suite == "{suite}":\n{indent}    return {helper}(config)'
    return text[: anchor.end()] + "\n" + route + text[anchor.end():]


def _replace_or_insert_function(text: str, name: str, function: str) -> str:
    pattern = re.compile(rf'(?ms)^def {re.escape(name)}\(config\):\n.*?(?=^def |\Z)')
    replacement = function.rstrip() + "\n\n"
    if pattern.search(text):
        return pattern.sub(replacement, text, count=1)
    marker = "\ndef make_env(config, id):"
    if marker in text:
        return text.replace(marker, "\n\n" + replacement + "def make_env(config, id):", 1)
    return text.rstrip() + "\n\n" + replacement


def patch_env_init(path: pathlib.Path) -> None:
    text = path.read_text()
    if "import atexit" not in text:
        future = "from __future__ import annotations\n"
        text = text.replace(future, future + "\nimport atexit\n", 1) if future in text else "import atexit\n" + text
    backup = path.with_suffix(".py.before_anymal_goalvec_v13_20")
    if not backup.exists():
        shutil.copy2(path, backup)
    text = _insert_suite_route(text, "isaaclabanymalroom", "_make_isaaclab_anymal_room_envs")
    anymal_function = '''def _make_isaaclab_anymal_room_envs(config):
    if int(config.eval_episode_num) > 0:
        raise ValueError(
            "The ANYmal-C IsaacLab backend does not create separate eval envs yet; "
            "set eval_episode_num: 0."
        )

    from isaaclab_runtime import get_simulation_app

    simulation_app = get_simulation_app()
    if simulation_app is None:
        raise RuntimeError("Isaac Lab was not launched. Run through r2dreamer_train.py.")

    from envs.isaaclab_anymal_room import IsaacLabAnymalRoomVecEnv, create_isaaclab_anymal_room_env

    raw_env = create_isaaclab_anymal_room_env(config)
    train_envs = IsaacLabAnymalRoomVecEnv(
        raw_env,
        config=config,
        simulation_app=simulation_app,
        image_size=tuple(config.size),
    )
    atexit.register(train_envs.close)
    return train_envs, None, train_envs.observation_space, train_envs.action_space
'''
    text = _replace_or_insert_function(text, "_make_isaaclab_anymal_room_envs", anymal_function)
    text = re.sub(r"\n{3,}", "\n\n", text)
    path.write_text(text)


def patch_discrete_action_compat(path: pathlib.Path) -> None:
    """Only compatibility casts for Gymnasium Discrete.n; no Dreamer objective changes."""
    text = path.read_text()
    backup = path.with_suffix(".py.before_anymal_goalvec_v13_20")
    if not backup.exists():
        shutil.copy2(path, backup)
    replacements = {
        'self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)':
            'self.act_dim = int(act_space.n) if hasattr(act_space, "n") else int(sum(map(int, act_space.shape)))',
        'config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))':
            'config.actor.shape = (int(act_space.n),) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))',
    }
    for original, patched in replacements.items():
        if original in text:
            text = text.replace(original, patched, 1)
        elif patched not in text:
            raise RuntimeError("Could not find expected DreamerV3 discrete-action line; repo revision is incompatible")
    path.write_text(text)


def patch_dreamer_decoder_key_scales(path: pathlib.Path) -> None:
    """Support per-key scales for optional goal_vec and proprio decoder heads.

    MultiDecoder builds whichever MLP keys the launcher selects. MultiEncoder is
    routed independently, so goal_vec remains excluded while proprio can be added.
    """
    text = path.read_text()

    # Remove the older v13.14 override block if this checkout already has it.
    old_decoder_block = '            # CAMERA_POSE_DECODER_KEY_SCALES_V13_14\n            # Upstream gives every decoder key the same recon scale. Allow\n            # explicit loss_scales.<key> entries (especially the 4-D goal vector)\n            # to override that shared default.\n            for key in self.decoder.all_keys:\n                if key in config.loss_scales:\n                    self._loss_scales[key] = float(config.loss_scales[key])\n            print(\n                "[V13.14 DECODER SCALES] "\n                + ", ".join(f"{k}={self._loss_scales[k]:g}" for k in self.decoder.all_keys)\n            )\n'
    old_weighted_block = '        # CAMERA_POSE_WEIGHTED_LOSS_METRICS_V13_14\n        metrics.update({\n            f"loss_weighted/{name}": loss * self._loss_scales[name]\n            for name, loss in losses.items()\n        })\n'
    text = text.replace(old_decoder_block, "").replace(old_weighted_block, "")

    marker = "# GOALVEC_DECODER_KEY_SCALES_V13_20"
    if marker not in text:
        backup = path.with_suffix(".py.before_goalvec_decoder_v13_20")
        if not backup.exists():
            shutil.copy2(path, backup)
        old = '            recon = self._loss_scales.pop("recon")\n            self._loss_scales.update({k: recon for k in self.decoder.all_keys})\n'
        new = old + (
            '            # GOALVEC_DECODER_KEY_SCALES_V13_20\n'
            '            # Explicit loss_scales.<decoder_key> entries override the shared\n'
            '            # reconstruction scale. Configured auxiliary keys include\n'
            '            # proprio and the decoder-only goal_vec probe.\n'
            '            for key in self.decoder.all_keys:\n'
            '                if key in config.loss_scales:\n'
            '                    self._loss_scales[key] = float(config.loss_scales[key])\n'
            '            print(\n'
            '                "[V13.20 DECODER SCALES] "\n'
            '                + ", ".join(f"{k}={self._loss_scales[k]:g}" for k in self.decoder.all_keys)\n'
            '            )\n'
        )
        if old not in text:
            raise RuntimeError(
                "Could not find Dreamer decoder recon-scale block; incompatible r2dreamer revision."
            )
        text = text.replace(old, new, 1)

    weighted_marker = "# GOALVEC_WEIGHTED_LOSS_METRICS_V13_20"
    if weighted_marker not in text:
        old = '        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})\n        metrics.update({"opt/loss": total_loss})\n'
        new = '        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})\n' + '        # GOALVEC_WEIGHTED_LOSS_METRICS_V13_20\n        metrics.update({\n            f"loss_weighted/{name}": loss * self._loss_scales[name]\n            for name, loss in losses.items()\n        })\n' + '        metrics.update({"opt/loss": total_loss})\n'
        if old not in text:
            raise RuntimeError("Could not find Dreamer loss metric block for weighted metrics.")
        text = text.replace(old, new, 1)
    path.write_text(text)

def patch_trainer_episode_metrics(path: pathlib.Path) -> None:
    """Logging only; does not change the training objective."""
    text = path.read_text()
    marker = "# CAMERA_POSE_TRAIN_EPISODE_METRICS_V13_7"
    if marker in text:
        return
    backup = path.with_suffix(".py.before_camera_pose_v13_7_metrics")
    if not backup.exists():
        shutil.copy2(path, backup)
    old = "        train_metrics = {}\n        agent_state = agent.get_initial_state(envs.env_num)"
    new = (
        "        train_metrics = {}\n"
        f"        {marker}\n"
        "        episode_log_metrics = {}\n"
        "        agent_state = agent.get_initial_state(envs.env_num)"
    )
    if old not in text:
        raise RuntimeError("Could not find OnlineTrainer train_metrics initialization")
    text = text.replace(old, new, 1)
    old = '''                        self.logger.scalar("episode/score", returns[i])
                        self.logger.scalar("episode/length", lengths[i])
                        self.logger.write(step + i)  # to show all values on tensorboard
                        returns[i] = lengths[i] = 0
'''
    new = '''                        self.logger.scalar("episode/score", returns[i])
                        self.logger.scalar("episode/length", lengths[i])
                        for key, values in episode_log_metrics.items():
                            value = values[i]
                            if key == "log_success":
                                value = torch.clamp(value, max=1.0)
                            self.logger.scalar(f"episode/{key}", value)
                        self.logger.write(step + i)  # to show all values on tensorboard
                        returns[i] = lengths[i] = 0
                        for values in episode_log_metrics.values():
                            values[i] = 0.0
'''
    if old not in text:
        raise RuntimeError("Could not find OnlineTrainer episode logging block")
    text = text.replace(old, new, 1)
    old = '            returns += trans["reward"][:, 0]\n            # Update models after enough data has accumulated'
    new = '''            returns += trans["reward"][:, 0]
            for key, value in trans.items():
                if key.startswith("log_"):
                    if key not in episode_log_metrics:
                        episode_log_metrics[key] = torch.zeros_like(returns)
                    episode_log_metrics[key] += value[:, 0]
            # Update models after enough data has accumulated'''
    if old not in text:
        raise RuntimeError("Could not find OnlineTrainer reward accumulation block")
    path.write_text(text.replace(old, new, 1))


def patch_trainer_run_checkpoints(path: pathlib.Path) -> None:
    """Install lightweight latest/best-success/best-reward checkpointing."""
    text = path.read_text()
    marker = "# CAMERA_POSE_RUN_CHECKPOINTS_V13_45"
    old_marker = "# CAMERA_POSE_RUN_CHECKPOINTS_V13_44"
    if marker in text:
        return
    if old_marker in text:
        # The trainer hooks are unchanged. Replacing the marker is sufficient;
        # the copied v13.45 checkpoint manager supplies the simpler policy.
        path.write_text(text.replace(old_marker, marker, 1))
        return

    backup = path.with_suffix(".py.before_camera_pose_v13_45_checkpoints")
    if not backup.exists():
        shutil.copy2(path, backup)

    old_init = """        # CAMERA_POSE_PERIODIC_CHECKPOINT_V13_11
        self._camera_pose_logdir = pathlib.Path(logdir)
        self._camera_pose_checkpoint_every = tools.Every(10000)
"""
    text = text.replace(old_init, "")
    old_save = """            if self._camera_pose_checkpoint_every(step):
                torch.save(
                    {
                        "agent_state_dict": agent.state_dict(),
                        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                    },
                    self._camera_pose_logdir / "latest.pt",
                )
            # Update models after enough data has accumulated
"""
    text = text.replace(
        old_save, "            # Update models after enough data has accumulated\n"
    )

    if "import camera_pose_checkpoints\n" not in text:
        if "import torch\n" in text:
            text = text.replace(
                "import torch\n", "import torch\n\nimport camera_pose_checkpoints\n", 1
            )
        else:
            text = "import camera_pose_checkpoints\n" + text

    init_anchor = "        self._action_repeat = config.action_repeat\n"
    init_block = init_anchor + f"""        {marker}
        self._camera_pose_checkpoints = camera_pose_checkpoints.CheckpointManager(logdir, config)
        self._camera_pose_last_step = 0
"""
    if init_anchor not in text:
        raise RuntimeError("Could not find trainer action_repeat line")
    text = text.replace(init_anchor, init_block, 1)

    episode_anchor = (
        "                        self.logger.write(step + i)  # to show all values on tensorboard\n"
    )
    episode_block = """                        self._camera_pose_checkpoints.record_episode(
                            score=returns[i],
                            length=lengths[i],
                            metrics={key: values[i] for key, values in episode_log_metrics.items()},
                        )
""" + episode_anchor
    if episode_anchor not in text:
        raise RuntimeError("Could not find patched episode logger for checkpoint selection")
    text = text.replace(episode_anchor, episode_block, 1)

    update_anchor = "            # Update models after enough data has accumulated\n"
    update_block = """            self._camera_pose_last_step = int(step)
            if self._camera_pose_checkpoints.should_save(step):
                self._camera_pose_checkpoints.save(agent, step, reason="periodic")
""" + update_anchor
    if update_anchor not in text:
        raise RuntimeError("Could not find trainer update anchor")
    text = text.replace(update_anchor, update_block, 1)

    method = """    def save_final_checkpoint(self, agent):
        self._camera_pose_checkpoints.save(
            agent, self._camera_pose_last_step, reason="final"
        )

"""
    method_anchor = "\n    def eval(self, agent, train_step):"
    if method_anchor not in text:
        method_anchor = "\n    def train(self, agent, envs):"
    if method_anchor not in text:
        method_anchor = "\n    def begin(self, agent):"
    if method_anchor not in text:
        raise RuntimeError("Could not find trainer method insertion anchor")
    text = text.replace(method_anchor, "\n" + method + method_anchor.lstrip("\n"), 1)
    path.write_text(text)


def patch_train_run_artifacts(path: pathlib.Path) -> None:
    """Save only resolved code parameters and delegate the final checkpoint."""
    text = path.read_text()
    marker = "# CAMERA_POSE_RUN_ARTIFACTS_V13_45"
    if marker in text:
        return
    backup = path.with_suffix(".py.before_camera_pose_v13_45_run_artifacts")
    if not backup.exists():
        shutil.copy2(path, backup)

    config_block = f"""    {marker}
    from omegaconf import OmegaConf
    config_dir = logdir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True)
    )
    try:
        from hydra.core.hydra_config import HydraConfig
        overrides = HydraConfig.get().overrides.task
    except Exception:
        overrides = []
    (config_dir / "hydra_overrides.txt").write_text(
        "\\n".join(overrides) + ("\\n" if overrides else "")
    )
"""

    old_marker = "    # CAMERA_POSE_RUN_ARTIFACTS_V13_44\n"
    old_end = (
        '    (config_dir / "hydra_overrides.txt").write_text('
        '"\\n".join(overrides) + ("\\n" if overrides else ""))\n'
    )
    if old_marker in text:
        start = text.index(old_marker)
        end_start = text.find(old_end, start)
        if end_start < 0:
            raise RuntimeError("Could not identify the end of the v13.44 config block")
        end = end_start + len(old_end)
        text = text[:start] + config_block + text[end:]
    else:
        config_anchor = "    logger.log_hydra_config(config)\n"
        if config_anchor not in text:
            raise RuntimeError("Could not find logger.log_hydra_config in train.py")
        text = text.replace(config_anchor, config_anchor + config_block, 1)

    upstream_final = """    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")
"""
    old_v1344_final = """    if hasattr(policy_trainer, "save_final_checkpoint"):
        policy_trainer.save_final_checkpoint(agent)
    else:
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    (logdir / "RUNNING").unlink(missing_ok=True)
    (logdir / "COMPLETED").write_text("completed\n")
"""
    final = """    if hasattr(policy_trainer, "save_final_checkpoint"):
        policy_trainer.save_final_checkpoint(agent)
    else:
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
"""
    if old_v1344_final in text:
        text = text.replace(old_v1344_final, final, 1)
    elif upstream_final in text:
        text = text.replace(upstream_final, final, 1)
    elif final not in text:
        raise RuntimeError("Could not find final latest.pt save block in train.py")

    # v13.44 wrote completion marker files from upstream train.py. Run status is
    # now owned exclusively by run_manager.py, so remove those lines even when
    # whitespace or escaped-newline spelling differs from the exact old block.
    import re
    text = re.sub(
        r'^\s*\(logdir / ["\']RUNNING["\']\)\.unlink\(missing_ok=True\)\s*\n',
        '',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^\s*\(logdir / ["\']COMPLETED["\']\)\.write_text\([^\n]*\)\s*\n',
        '',
        text,
        flags=re.MULTILINE,
    )
    path.write_text(text)


def patch_tools_gif_logger(path: pathlib.Path) -> None:
    """Visualization only; does not change Dreamer training."""
    text = path.read_text()
    backup = path.with_suffix(".py.before_camera_pose_v13_1_spot")
    if not backup.exists():
        shutil.copy2(path, backup)
    marker_begin = "# BEGIN CAMERA_POSE_GIF_PATCH"
    marker_end = "# END CAMERA_POSE_GIF_PATCH"
    if marker_begin in text:
        start = text.index(marker_begin)
        end = text.index(marker_end, start) + len(marker_end)
        text = text[:start].rstrip() + "\n\n" + text[end:].lstrip()
    patch = (PAYLOAD / "tools_gif_patch.py.txt").read_text().strip()
    path.write_text(text.rstrip() + "\n\n" + patch + "\n")


def install(repo: pathlib.Path) -> None:
    copies = [
        (PAYLOAD / "envs" / "isaaclab_anymal_room.py", repo / "envs" / "isaaclab_anymal_room.py"),
        (PAYLOAD / "envs" / "target_pose_debug.py", repo / "envs" / "target_pose_debug.py"),
        (PAYLOAD / "configs" / "env" / "isaaclab_anymal_room.yaml", repo / "configs" / "env" / "isaaclab_anymal_room.yaml"),
        (PAYLOAD / "isaaclab_runtime.py", repo / "isaaclab_runtime.py"),
        (HERE / "camera_pose_checkpoints.py", repo / "camera_pose_checkpoints.py"),
    ]
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    patch_env_init(repo / "envs" / "__init__.py")
    patch_dreamer_decoder_key_scales(repo / "dreamer.py")
    patch_discrete_action_compat(repo / "dreamer.py")
    patch_trainer_episode_metrics(repo / "trainer.py")
    patch_trainer_run_checkpoints(repo / "trainer.py")
    patch_train_run_artifacts(repo / "train.py")
    patch_tools_gif_logger(repo / "tools.py")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo")
    args = parser.parse_args()
    root = find_root(HERE)
    repo = find_repo(root, args.repo)
    install(repo)
    print(f"[V13.45 SIMPLE RUNS] IsaacLab root: {root}")
    print(f"[V13.45 SIMPLE RUNS] r2dreamer:     {repo}")
    print("[V13.45 SIMPLE RUNS] Installed the unchanged visible-arrow ANYmal task plus isolated run folders, minimal parameter snapshots, latest/best-success/best-reward checkpoints, and optional 33-D proprioception.")
    print("[V13.45 SIMPLE RUNS] Reward is exactly the v13.20 bounded dense state reward: 0.5*exp(-E/2) plus 0.5 precise-tolerance occupancy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
