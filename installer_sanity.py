#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bundle_install", HERE / "install.py")
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


def make_repo(root: pathlib.Path) -> pathlib.Path:
    repo = root / "r2dreamer"
    (repo / "envs").mkdir(parents=True)
    (repo / "configs/env").mkdir(parents=True)
    (repo / "train.py").write_text('''import pathlib
import torch
import tools

def main(config):
    logdir = pathlib.Path(config.logdir)
    logger = tools.Logger(logdir)
    logger.log_hydra_config(config)
    agent = object()
    policy_trainer = object()
    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")
''')
    (repo / "envs/__init__.py").write_text('''from __future__ import annotations

def _make_isaaclab_envs(config):
    return None

def make_envs(config):
    suite = config.task.split("_", 1)[0]
    if suite == "isaaclab":
        return _make_isaaclab_envs(config)

def make_env(config, id):
    return None
''')
    (repo / "dreamer.py").write_text('''class MockDreamer:
    def __init__(self, act_space, config):
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        if True:
            recon = self._loss_scales.pop("recon")
            self._loss_scales.update({k: recon for k in self.decoder.all_keys})

    def metrics(self, losses, total_loss):
        metrics = {}
        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return metrics
''')
    (repo / "trainer.py").write_text('''import torch

class MockTrainer:
    def __init__(self, config, logdir):
        self._action_repeat = config.action_repeat

    def train(self, agent, envs):
        train_metrics = {}
        agent_state = agent.get_initial_state(envs.env_num)
        returns = torch.zeros(envs.env_num)
        lengths = torch.zeros(envs.env_num)
        trans = {"reward": torch.zeros(envs.env_num, 1)}
        step = 0
        for i in range(envs.env_num):
            if False:
                if False:
                    if False:
                        self.logger.scalar("episode/score", returns[i])
                        self.logger.scalar("episode/length", lengths[i])
                        self.logger.write(step + i)  # to show all values on tensorboard
                        returns[i] = lengths[i] = 0
        if True:
            returns += trans["reward"][:, 0]
            # Update models after enough data has accumulated
        return train_metrics, agent_state
''')
    (repo / "tools.py").write_text("# mock tools\n")
    return repo


with tempfile.TemporaryDirectory() as td:
    repo = make_repo(pathlib.Path(td))
    mod.install(repo)
    mod.install(repo)

    env = (repo / "envs/isaaclab_anymal_room.py").read_text()
    cfg = (repo / "configs/env/isaaclab_anymal_room.yaml").read_text()
    dreamer = (repo / "dreamer.py").read_text()
    trainer = (repo / "trainer.py").read_text()
    init = (repo / "envs/__init__.py").read_text()
    train = (repo / "train.py").read_text()

    assert '"goal_vec": self._goal_conditioned_observation()' in env
    assert 'r_proximity + r_in_tolerance' in env and 'r_joint_pose' not in env
    assert 'def _mixed_task_seed' in env and 'self._task_rng' not in env
    assert 'spaces["proprio"]' in env
    assert 'balanced_randomization: true' in cfg
    assert 'proprioception_enabled: false' in cfg
    assert '_v13_45_balanced_randomization_proprio' in cfg
    assert dreamer.count('GOALVEC_DECODER_KEY_SCALES_V13_20') == 1
    assert dreamer.count('GOALVEC_WEIGHTED_LOSS_METRICS_V13_20') == 1
    assert trainer.count('CAMERA_POSE_TRAIN_EPISODE_METRICS_V13_7') == 1
    assert trainer.count('CAMERA_POSE_RUN_CHECKPOINTS_V13_45') == 1
    assert 'CAMERA_POSE_RUN_CHECKPOINTS_V13_44' not in trainer
    assert 'record_episode(' in trainer and 'save_final_checkpoint' in trainer
    assert 'suite == "isaaclabanymalroom"' in init
    assert train.count('CAMERA_POSE_RUN_ARTIFACTS_V13_45') == 1
    assert 'resolved_config.yaml' in train
    assert 'hydra_overrides.txt' in train
    assert 'unresolved_config.yaml' not in train
    assert 'RUNNING' not in train and 'COMPLETED' not in train
    assert 'policy_trainer.save_final_checkpoint(agent)' in train
    assert (repo / 'camera_pose_checkpoints.py').is_file()

# Explicitly exercise the v13.44-to-v13.45 cleanup path.
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    trainer = root / "trainer.py"
    trainer.write_text("# CAMERA_POSE_RUN_CHECKPOINTS_V13_44\n")
    mod.patch_trainer_run_checkpoints(trainer)
    assert trainer.read_text() == "# CAMERA_POSE_RUN_CHECKPOINTS_V13_45\n"

    train = root / "train.py"
    train.write_text('''def main(config):
    logger.log_hydra_config(config)
    # CAMERA_POSE_RUN_ARTIFACTS_V13_44
    from omegaconf import OmegaConf
    config_dir = logdir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml = OmegaConf.to_yaml(config, resolve=True)
    (config_dir / "resolved_config.yaml").write_text(resolved_yaml)
    unresolved_yaml = OmegaConf.to_yaml(config, resolve=False)
    (config_dir / "unresolved_config.yaml").write_text(unresolved_yaml)
    try:
        from hydra.core.hydra_config import HydraConfig
        overrides = HydraConfig.get().overrides.task
    except Exception:
        overrides = []
    (config_dir / "hydra_overrides.txt").write_text("\\n".join(overrides) + ("\\n" if overrides else ""))
    if hasattr(policy_trainer, "save_final_checkpoint"):
        policy_trainer.save_final_checkpoint(agent)
    else:
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    (logdir / "RUNNING").unlink(missing_ok=True)
    (logdir / "COMPLETED").write_text("completed\\n")
''')
    mod.patch_train_run_artifacts(train)
    upgraded = train.read_text()
    assert 'CAMERA_POSE_RUN_ARTIFACTS_V13_45' in upgraded
    assert 'CAMERA_POSE_RUN_ARTIFACTS_V13_44' not in upgraded
    assert 'unresolved_config.yaml' not in upgraded
    assert 'RUNNING' not in upgraded and 'COMPLETED' not in upgraded

print("[INSTALLER SANITY] fresh, idempotent, and v13.44 upgrade paths passed")
