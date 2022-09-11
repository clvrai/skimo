"""Train skill-based RL methods."""

import numpy as np
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import torch

from rolf.trainer import Trainer
from rolf.utils import Logger, LOG_TYPES, Every, StopWatch, Info
from rolf.utils.pytorch import to_tensor


class SkillTrainer(Trainer):
    """Trainer class for skill-based methods."""

    def __init__(self, cfg):
        super().__init__(cfg)

        if cfg.env.id == "maze":
            self._overlay = imageio.imread(f"envs/assets/maze_40.png")

    def _get_agent_by_name(self, algo):
        if algo == "spirl_dreamer":
            from spirl_dreamer_agent import SPiRLDreamerAgent

            return SPiRLDreamerAgent
        if algo == "spirl_tdmpc":
            from spirl_tdmpc_agent import SPiRLTDMPCAgent

            return SPiRLTDMPCAgent
        if algo == "skimo":
            from skimo_agent import SkiMoAgent

            return SkiMoAgent
        return super()._get_agent_by_name(algo)

    def train(self):
        if self._cfg.rolf.phase == "pretrain":
            self._pretrain()
        elif self._cfg.rolf.phase == "rl":
            super()._train()
        else:
            raise ValueError(f"rolf.phase={self._cfg.rolf.phase} is not supported")

    def evaluate(self):
        rollouts = super().evaluate()
        self._log_test(step=0, ep_info={}, rollouts=rollouts)

    def _pretrain(self):
        """Pretrain the agent with offline data."""
        cfg = self._cfg

        # Load checkpoint
        ckpt_info = self._load_ckpt(cfg.init_ckpt_path, cfg.ckpt_num)
        step = ckpt_info.get("step", 0)

        # Sync the networks across the cpus
        self._agent.sync_networks()

        Logger.info(f"Start pretraining at step={step}")
        if self._is_chef:
            pbar = tqdm(
                initial=step, total=cfg.rolf.pretrain.max_global_step, desc=cfg.run_name
            )
            train_info = Info()
            should_log = Every(cfg.rolf.pretrain.log_every, step)
            should_evaluate = Every(cfg.rolf.pretrain.evaluate_every, step)
            should_ckpt = Every(cfg.rolf.pretrain.ckpt_every, step)
            timer = StopWatch(step)
            ep_rollouts = []

        while step < cfg.rolf.pretrain.max_global_step:
            # Train an agent
            _train_info = self._agent.pretrain()
            train_info.add(_train_info)
            step += 1

            if not self._is_chef:
                continue

            pbar.update(1)
            if should_log(step):
                train_info.add({"steps_per_sec": timer(step)})
                self._log_pretrain(step, train_info.get_dict())

            if should_evaluate(step):
                Logger.warning(f"Evaluate at step={step}")
                eval_info = self._agent.pretrain_eval()
                self._log_pretrain(step, eval_info, "_eval")

                # Add environment rollout for evaluation
                Logger.info("Creating environment rollout")
                ep_rollouts_, ep_info = self._evaluate(step, cfg.record_video)
                if cfg.env.id == "maze":
                    ep_rollouts.extend(ep_rollouts_)
                    self._log_test(step=step, ep_info={}, rollouts=ep_rollouts)
                self._log_pretrain(step, ep_info.get_dict(), "_eval")

            if should_ckpt(step):
                self._save_ckpt(step)

        # Store the final model
        if self._is_chef:
            self._save_ckpt(step)

        Logger.info(f"Reached {step} steps. Worker {cfg.rank} stopped.")

    def _log_pretrain(self, step, train_info, name=""):
        """Logs training and episode information to wandb.

        Args:
            step: the number of environment steps.
            train_info: training information to log, such as loss, gradient.
            name: postfix for the log section.
        """
        for k, v in train_info.items():
            if isinstance(v, wandb.Video) or isinstance(v, wandb.Image):
                wandb.log({f"pretrain{name}/{k}": v}, step=step)
            elif isinstance(v, list) and isinstance(v[0], wandb.Video):
                for i, video in enumerate(v):
                    wandb.log({f"pretrain{name}/{k}_{i}": video}, step=step)
            elif isinstance(v, list) and isinstance(v[0], wandb.Image):
                # Only log the first image
                wandb.log({f"pretrain{name}/{k}": v[0]}, step=step)
            elif isinstance(v, list) and isinstance(v[0], LOG_TYPES):
                wandb.log({f"pretrain{name}/{k}": np.mean(v)}, step=step)
            elif isinstance(v, LOG_TYPES):
                wandb.log({f"pretrain{name}/{k}": v}, step=step)

    def _log_test(self, step, ep_info, name="", rollouts=None):
        """Logs visualization of maze experiments."""
        if self._cfg.env.id == "maze":
            key = "state" if self._cfg.env.pixel_ob else "ob"
            if rollouts is None:
                buffer = self._agent.skill_agent.buffer.buffer
                ob = np.concatenate([v["ob"][key][:, :2] for v in buffer], 0)
            else:
                ob = np.concatenate(
                    [np.stack([_v[key] for _v in v["ob"]])[:, :2] for v in rollouts], 0
                )

            fig = plt.figure()

            plt.imshow(
                self._overlay, alpha=0.3, extent=(0, 40, 0, 40),
            )
            plt.xlim(0, 40)
            plt.ylim(0, 40)

            plt.scatter(
                40 - ob[:, 1], ob[:, 0], s=5, c=np.arange(len(ob)), cmap="Blues"
            )
            plt.scatter(
                40 - self._env.START_POS[1],
                self._env.START_POS[0],
                s=150,
                color="g",
                edgecolors="honeydew",
                linewidths=2,
            )
            plt.scatter(
                40 - self._env.TARGET_POS[1],
                self._env.TARGET_POS[0],
                s=150,
                color="r",
                edgecolors="mistyrose",
                linewidths=2,
            )
            plt.axis("equal")

            # Save evaluation image, since wandb is not initialized
            if not self._cfg.is_train:
                plt.savefig(f"{self._cfg.run_name}_{step}_eval.png")
                return

            wandb.log(
                {f"test_ep{name}/replay_vis": wandb.Image(fig)}, step=step,
            )
            plt.close(fig)

            # Value heatmap visualization
            try:
                self._visualize_value(step)
            except:
                Logger.warning("Failed to visualize value heatmap")

        super()._log_test(step, ep_info, name)

    def _visualize_value(self, step):
        """Visualize the learned value function as a heatmap."""
        buffer = self._agent.meta_agent.buffer.buffer
        num_samples = self._cfg.rolf.warm_up_step // 10
        if len(buffer) < num_samples:
            return
        ob = np.concatenate([v["ob"]["ob"] for v in buffer], 0)
        ob = np.concatenate([ob[:num_samples], ob[-num_samples:]], 0)
        ob_tensor = dict(ob=to_tensor(ob, device="cuda"))
        ob_tensor = self._agent.meta_agent.preprocess(ob_tensor)

        feat = self._agent.meta_agent.model.encoder(ob_tensor)
        ac = self._agent.meta_agent.actor(feat).mode()
        value = (
            torch.min(*self._agent.meta_agent.model.critic(feat, ac))
            .detach()
            .cpu()
            .numpy()
        )

        fig, ax = plt.subplots()
        ax.axis([0, 40, 0, 40])

        sc = plt.scatter(40 - ob[:, 1], ob[:, 0], s=5, c=value, cmap="Blues")
        plt.colorbar(sc)

        plt.scatter(
            40 - self._env.START_POS[1],
            self._env.START_POS[0],
            s=150,
            color="g",
            edgecolors="honeydew",
            linewidths=2,
        )
        plt.scatter(
            40 - self._env.TARGET_POS[1],
            self._env.TARGET_POS[0],
            s=150,
            color="r",
            edgecolors="mistyrose",
            linewidths=2,
        )

        # plt.savefig(f"{self._cfg.run_name}_{step}_heatmap_eval.png")

        wandb.log(
            {f"test_ep/value_heatmap": wandb.Image(fig)}, step=step,
        )
        plt.close(fig)
