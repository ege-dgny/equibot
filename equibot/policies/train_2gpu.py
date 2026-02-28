"""
EquiBot training with 2 GPUs via DataParallel.

Wraps the encoder and noise_pred_net sub-networks in DataParallel so
forward/backward passes are split across GPUs. The agent, actor, optimizer,
EMA, and all other logic remain unchanged on the primary GPU.

Produces checkpoints fully compatible with the standard single-GPU eval.

Run:  python -m equibot.policies.train_2gpu --config-name <config> ...
"""

import os
import torch
import torch.nn as nn
import hydra
import wandb
import omegaconf
import numpy as np
from tqdm import tqdm
from glob import glob
from omegaconf import OmegaConf

from equibot.policies.utils.media import save_video
from equibot.policies.utils.misc import get_env_class, get_dataset, get_agent
from equibot.policies.vec_eval import run_eval
from equibot.envs.subproc_vec_env import SubprocVecEnv

DP_DEVICE_IDS = [0, 1]


def _make_dp_encoder_handle(encoder, device_ids):
    """Return a callable that wraps the encoder in DataParallel.
    Scalar-tensor kwargs (like target_norm) are converted to Python floats
    so DataParallel's scatter doesn't choke on 0-d tensors."""
    dp = nn.DataParallel(encoder, device_ids=device_ids)

    def _handle(*args, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                kwargs[k] = v.item()
        return dp(*args, **kwargs)

    return _handle


def _wrap_subnets_dp(agent):
    """Replace encoder_handle and noise_pred_net_handle with DataParallel
    wrappers.  Uses object.__setattr__ so the wrappers are NOT registered
    as nn.Module sub-modules, keeping state_dict / save_snapshot clean."""
    actor = agent.actor
    if actor.encoder is not None:
        dp_enc = _make_dp_encoder_handle(actor.encoder, DP_DEVICE_IDS)
        object.__setattr__(actor, "encoder_handle", dp_enc)
    dp_npn = nn.DataParallel(actor.noise_pred_net, device_ids=DP_DEVICE_IDS)
    object.__setattr__(actor, "noise_pred_net_handle", dp_npn)


def _unwrap_subnets_dp(agent):
    """Restore original (non-DP) handles for save / eval."""
    actor = agent.actor
    if actor.encoder is not None:
        object.__setattr__(actor, "encoder_handle", actor.encoder)
    object.__setattr__(actor, "noise_pred_net_handle", actor.noise_pred_net)


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):
    assert cfg.mode == "train"
    np.random.seed(cfg.seed)

    cfg.device = f"cuda:{DP_DEVICE_IDS[0]}"

    batch_size = cfg.training.batch_size

    if cfg.use_wandb:
        wandb_config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=False
        )
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            tags=["train", "2gpu"],
            name=cfg.prefix,
            settings=wandb.Settings(code_dir="."),
            config=wandb_config,
        )
    log_dir = os.getcwd()

    train_dataset = get_dataset(cfg, "train")
    num_workers = cfg.data.dataset.num_workers
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    cfg.data.dataset.num_training_steps = (
        cfg.training.num_epochs * len(train_dataset) // batch_size
    )

    env_class = get_env_class(cfg.env.env_class)
    env_args = dict(OmegaConf.to_container(cfg.env.args, resolve=True))

    def create_env(env_args, i):
        env_args_copy = env_args.copy()
        env_args_copy["seed"] = cfg.seed * 100 + i
        return env_class(OmegaConf.create(env_args_copy))

    if cfg.training.eval_interval <= cfg.training.num_epochs:
        env = SubprocVecEnv(
            [
                lambda seed=i: create_env(env_args, seed)
                for i in range(cfg.training.num_eval_episodes)
            ]
        )
    else:
        env = None

    agent = get_agent(cfg.agent.agent_name)(cfg)
    if cfg.training.ckpt is not None:
        agent.load_snapshot(cfg.training.ckpt)
        start_epoch_ix = int(cfg.training.ckpt.split("/")[-1].split(".")[0][4:])
    else:
        start_epoch_ix = 0

    _wrap_subnets_dp(agent)

    global_step = 0
    for epoch_ix in tqdm(range(start_epoch_ix, cfg.training.num_epochs)):
        batch_ix = 0
        for batch in tqdm(train_loader, leave=False, desc="Batches"):
            train_metrics = agent.update(
                batch,
                vis=epoch_ix % cfg.training.vis_interval == 0 and batch_ix == 0,
            )
            if cfg.use_wandb:
                wandb.log(
                    {"train/" + k: v for k, v in train_metrics.items()},
                    step=global_step,
                )
                wandb.log({"epoch": epoch_ix}, step=global_step)
            del train_metrics
            global_step += 1
            batch_ix += 1

        if (
            (
                epoch_ix % cfg.training.eval_interval == 0
                or epoch_ix == cfg.training.num_epochs - 1
            )
            and epoch_ix > 0
            and env is not None
        ):
            _unwrap_subnets_dp(agent)
            eval_metrics = run_eval(
                env,
                agent,
                vis=True,
                num_episodes=cfg.training.num_eval_episodes,
                reduce_horizon_dim=cfg.data.dataset.reduce_horizon_dim,
                use_wandb=cfg.use_wandb,
            )
            _wrap_subnets_dp(agent)
            if cfg.use_wandb:
                if epoch_ix > cfg.training.eval_interval and "vis_pc" in eval_metrics:
                    del eval_metrics["vis_pc"]
                wandb.log(
                    {
                        "eval/" + k: v
                        for k, v in eval_metrics.items()
                        if k not in ["vis_rollout", "rew_values"]
                    },
                    step=global_step,
                )
                if "vis_rollout" in eval_metrics:
                    for eval_idx, eval_video in enumerate(eval_metrics["vis_rollout"]):
                        video_path = os.path.join(
                            log_dir,
                            f"eval{epoch_ix:05d}_ep{eval_idx}_rew{eval_metrics['rew_values'][eval_idx]}.mp4",
                        )
                        save_video(eval_video, video_path)
                        print(f"Saved eval video to {video_path}")
            del eval_metrics

        if (
            epoch_ix % cfg.training.save_interval == 0
            or epoch_ix == cfg.training.num_epochs - 1
        ):
            save_path = os.path.join(log_dir, f"ckpt{epoch_ix:05d}.pth")
            num_ckpt_to_keep = 10
            ckpts = sorted(glob(os.path.join(log_dir, "ckpt*.pth")))
            if len(ckpts) > num_ckpt_to_keep:
                for fn in ckpts[:-num_ckpt_to_keep]:
                    os.remove(fn)
            _unwrap_subnets_dp(agent)
            agent.save_snapshot(save_path)
            _wrap_subnets_dp(agent)


if __name__ == "__main__":
    main()
