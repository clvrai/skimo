"""Train spirl."""

import numpy as np
import torch
import wandb
from tqdm import tqdm
from mpi4py import MPI

from rolf.trainer import Trainer
from rolf.rolf.utils import Logger, Every, StopWatch, Info

from spirl.rl.components.sampler import HierarchicalSampler
from spirl.rl.utils.mpi import mpi_sum, mpi_gather_experience
from spirl.rl.components.replay_buffer import RolloutStorage

from spirl_agent import SPiRLAgent


class SPiRLTrainer(Trainer):
    """Trainer class for spirl."""

    def __init__(self, cfg):
        """Initializes class with the configuration."""
        cfg.rolf.env = cfg.env.id
        self._cfg = cfg
        super().__init__(cfg)

        # set up device
        self._agent.to(cfg.device)
        self._agent.device = cfg.device
        self._env.device = cfg.device

        # build sampler
        self.sampler = HierarchicalSampler(
            cfg.rolf.sampler_config,
            self._env,
            self._agent,
            Logger,
            cfg.env.max_episode_steps,
        )

        self._num_workers = MPI.COMM_WORLD.Get_size()

    def _get_agent_by_name(self, algo):
        return SPiRLAgent

    def train(self):
        """Trains an agent."""
        cfg = self._cfg

        # load checkpoint
        ckpt_info = self._load_ckpt(cfg.init_ckpt_path, cfg.ckpt_num)
        step = ckpt_info.get("step", 0)

        # sync the networks across the cpus
        self._agent.sync_networks()

        Logger.info(f"Start training at step={step}")
        if self._is_chef:
            pbar = tqdm(initial=step, total=cfg.rolf.max_global_step, desc=cfg.run_name)
            ep_info = Info()
            train_info = Info()
            should_log = Every(cfg.rolf.log_every, step)
            should_evaluate = Every(cfg.rolf.evaluate_every, step)
            should_ckpt = Every(cfg.rolf.ckpt_every, step)
            should_sampler_init = Every(cfg.rolf.sampler_init_every, step)
            timer = StopWatch(step)

        # collect warm-up rollout with random policy if starting from scratch
        if self._is_chef:
            Logger.info(f"Warmup data collection for {cfg.rolf.warm_up_step} steps...")
        with self._agent.rand_act_mode():
            self.sampler.init(is_train=True)
            warmup_experience_batch, _ = self.sampler.sample_batch(
                batch_size=int(cfg.rolf.warm_up_step / self._num_workers)
            )
            if self._num_workers > 1:
                warmup_experience_batch = mpi_gather_experience(warmup_experience_batch)
        if self._is_chef:
            self._agent.add_experience(warmup_experience_batch)
            Logger.info("...Warmup done!")

        while step < cfg.rolf.max_global_step:
            if should_sampler_init(step):
                self.sampler.init(is_train=True)

            # collect experience
            experience_batch, env_steps = self.sampler.sample_batch(
                batch_size=cfg.rolf.n_steps_per_update, global_step=step
            )
            ep_info = Info(self.sampler.ep_info)
            if self._num_workers > 1:
                experience_batch = mpi_gather_experience(experience_batch)
            rollout_steps = mpi_sum(env_steps)
            step += rollout_steps

            # update policy
            if self._is_chef:
                agent_outputs = self._agent.update(experience_batch)
                train_info = Info(agent_outputs)
            if self._num_workers > 1:
                self._agent.sync_networks()

            # log training and episode information
            if not self._is_chef:
                continue

            pbar.update(rollout_steps)

            if should_log(step):
                train_info.add({"steps_per_sec": timer(step)})
                self._log_train(step, train_info.get_dict(), ep_info.get_dict())
                self._agent.log_outputs(
                    agent_outputs,
                    None,
                    Logger,
                    log_images=False,
                    step=step,
                    log_scalar=False,
                )

            if should_evaluate(step):
                Logger.info(f"Evaluate at step={step}")
                ep_info = self._evaluate(step, cfg.record_video)
                self._log_test(step, ep_info[1].get_dict())

            if should_ckpt(step):
                self._save_ckpt(step)

        # store the final model
        if self._is_chef:
            self._save_ckpt(step)

        Logger.info(f"Reached {step} steps. Worker {cfg.rank} stopped.")

    def _evaluate(self, step=None, record_video=False):
        """Runs `self._cfg.num_eval` rollouts to evaluate agent.

        Args:
            step: the number of environment steps.
            record_video: whether to record video or not.
        """
        cfg = self._cfg
        Logger.info(f"Run {cfg.num_eval} evaluations at step={step}")
        info_history = Info()
        rollouts = []

        val_rollout_storage = RolloutStorage()
        with self._agent.val_mode():
            with torch.no_grad():
                for i in range(cfg.num_eval):
                    rollout = self.sampler.sample_episode(is_train=False, render=True)
                    rollouts.append(rollout)
                    val_rollout_storage.append(rollout)
                    rollout_stats = val_rollout_storage.rollout_stats()
                    info = Info(rollout_stats)

                    if record_video:
                        frames = np.stack(rollout.image)
                        if frames.max() <= 1.0:
                            frames = frames * 255.0
                            frames = frames.astype(np.uint8)
                        fname = f"{cfg.env.id}_step_{step:011d}_{i}.mp4"
                        video_path = self._save_video(fname, frames)
                        if cfg.is_train:
                            caption = f"{cfg.run_name}-{step}-{i}"
                            info["video"] = wandb.Video(
                                video_path, caption=caption, fps=15, format="mp4"
                            )
                    info_history.add(info)

        if self._is_chef:
            Logger.warning(f"Evaluation Avg_Reward: {rollout_stats.rew}")

        del val_rollout_storage
        return rollouts, info_history

    def _log_test(self, step, ep_info, name=""):
        """Logs episode information during testing to wandb.

        Args:
            step: the number of environment steps.
            ep_info: episode information to log, such as reward, episode time.
            name: postfix for the log section.
        """
        import matplotlib.pyplot as plt
        import imageio

        if self._cfg.env.id == "maze":
            buffer = self._agent.hl_agent.replay_buffer._replay_buffer
            ob = np.stack([v[:2] for v in buffer["observation"] if max(v) > 0], 0)

            fig = plt.figure()
            plt.imshow(
                imageio.imread("envs/assets/maze_40.png"),
                alpha=0.3,
                extent=(0, 40, 0, 40),
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

            # save evaluation image, since wandb is not initialized
            if not self._cfg.is_train:
                plt.savefig(
                    f"{self._cfg.run_name}_{step}_{self._cfg.num_eval}_eval.png"
                )
                return

            wandb.log(
                {f"test_ep{name}/replay_vis": wandb.Image(fig)}, step=step,
            )
            plt.close(fig)

        super()._log_test(step, ep_info, name)
