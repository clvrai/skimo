"""
Runs rollouts (RolloutRunner class) and collects transitions using Rollout class.
"""

import numpy as np
import gym.spaces

from rolf.utils import Logger, Info, Every
from rolf.algorithms.rollout import Rollout, RolloutRunner


class SkiMoRolloutRunner(RolloutRunner):
    """Rollout hierarchical policy."""

    def __init__(self, cfg, env, env_eval, agent):
        """
        Args:
            cfg: configurations for the environment.
            env: training environment.
            env_eval: testing environment.
            agent: policy.
        """
        self._cfg = cfg
        self._env = env
        self._env_eval = env_eval
        self._agent = agent
        self._meta_agent = agent.meta_agent
        self._skill_agent = agent.skill_agent
        self._exclude_rollout_log = ["episode_success_state"]

    def run(
        self, every_steps=None, every_episodes=None, log_prefix="", step=0,
    ):
        """
        Collects trajectories and yield every `every_steps`/`every_episodes`.

        Args:
            every_steps: if not None, returns rollouts `every_steps`.
            every_episodes: if not None, returns rollouts `every_epiosdes`.
            log_prefix: log as `log_prefix` rollout: %s.
        """
        if every_steps is None and every_episodes is None:
            raise ValueError("Both every_steps and every_episodes cannot be None")

        cfg = self._cfg
        env = self._env
        meta_agent = self._meta_agent
        skill_agent = self._skill_agent

        done_rollout = (
            Every(every_steps, step) if every_steps else Every(every_episodes, 0)
        )

        # Initialize rollout buffer.
        meta_rollout = Rollout(["ob", "ac", "rew", "done"], cfg.rolf.precision)
        rollout = Rollout(["ob", "meta_ac", "ac", "rew", "done"], cfg.rolf.precision)
        reward_info = Info()
        ep_info = Info()
        episode = 0
        rollout_len = 0

        while True:
            done = False
            ep_len = 0
            ep_rew = 0
            ob_next = env.reset()
            ob = ob_next
            state_next_hl = None
            state_next_ll = None

            rollout.add(dict(ob=ob_next))
            meta_rollout.add(dict(ob=ob_next))

            # Rollout one episode.
            while not done:
                state_hl = state_next_hl

                # Sample meta action from meta policy.
                if step < cfg.rolf.warm_up_step:
                    meta_ac, state_next_hl = meta_agent.act(
                        ob_next, state_hl, is_train=True, warmup=True
                    )
                else:
                    meta_ac, state_next_hl = meta_agent.act(
                        ob_next, state_hl, is_train=True
                    )

                # Execute a skill.
                skill_len = 0
                meta_rew = 0
                while not done and skill_len < cfg.rolf.skill_horizon:
                    ob = ob_next
                    state_ll = state_next_ll

                    # Sample low-level action from skill policy.
                    ac, state_next_ll = skill_agent.act(
                        ob, state_ll, cond=meta_ac, is_train=False
                    )

                    # Take a step.
                    ob_next, reward, done, info = env.step(ac)
                    info.update(env.get_episode_info())
                    info.update(dict(meta_ac=meta_ac))

                    step += 1
                    ep_len += 1
                    ep_rew += reward
                    skill_len += 1
                    meta_rew += reward

                    flat_ac = gym.spaces.flatten(env.action_space, ac)
                    rollout.add(dict(ob=ob_next, ac=flat_ac, done=done))
                    rollout.add(dict(meta_ac=meta_ac))
                    rollout.add(dict(rew=reward * cfg.rolf.reward_scale))
                    reward_info.add(info)
                    rollout_len += 1

                meta_rollout.add(dict(ob=ob_next, ac=meta_ac, done=done))
                meta_rollout.add(dict(rew=meta_rew * cfg.rolf.reward_scale))
                if every_steps and done_rollout(step):
                    yield (
                        meta_rollout.get(),
                        rollout.get(),
                    ), rollout_len, ep_info.get_dict(reduction="max", only_scalar=True)
                    rollout_len = 0

            # Compute average/sum/max of information.
            reward_info_dict = reward_info.get_dict(reduction="max", only_scalar=True)
            reward_info_dict.update(
                dict(len=ep_len, rew=ep_rew * cfg.rolf.reward_scale)
            )
            ep_info.add(reward_info_dict)

            Logger.info(
                "rollout: %s",
                {
                    k: v
                    for k, v in reward_info_dict.items()
                    if k not in self._exclude_rollout_log and np.isscalar(v)
                },
            )

            episode += 1
            if every_episodes and done_rollout(episode):
                yield (
                    meta_rollout.get(),
                    rollout.get(),
                ), rollout_len, ep_info.get_dict(only_scalar=True)
                rollout_len = 0

    def run_episode(self, record_video=False):
        """
        Runs one episode and returns the rollout (mainly for evaluation).

        Args:
            record_video: record video of rollout if True.
        """
        cfg = self._cfg
        env = self._env_eval
        meta_agent = self._meta_agent
        skill_agent = self._skill_agent

        # Initialize rollout buffer.
        rollout = Rollout(["ob", "meta_ac", "ac", "rew", "done"], cfg.rolf.precision)
        reward_info = Info()

        done = False
        ep_len = 0
        ep_rew = 0
        ob_next = env.reset()
        ob = ob_next
        state_next_hl = None
        state_next_ll = None

        record_frames = []
        if record_video:
            record_frames.append(self._render_frame(ep_len, ep_rew))

        # Rollout one episode.
        while not done:
            state_hl = state_next_hl

            # Sample meta action from meta policy.
            meta_ac, state_next_hl = meta_agent.act(ob_next, state_hl, is_train=False)

            skill_len = 0
            while not done and skill_len < cfg.rolf.skill_horizon:
                ob = ob_next
                state_ll = state_next_ll

                # Sample low-level action from skill policy.
                ac, state_next_ll = skill_agent.act(
                    ob, state_ll, cond=meta_ac, is_train=False
                )

                # Take a step.
                ob_next, reward, done, info = env.step(ac)
                info.update(env.get_episode_info())
                info.update(dict(meta_ac=meta_ac))

                ep_len += 1
                ep_rew += reward
                skill_len += 1

                flat_ac = gym.spaces.flatten(env.action_space, ac)
                rollout.add(dict(ob=ob_next, ac=flat_ac, done=done))
                rollout.add(dict(meta_ac=meta_ac, rew=reward * cfg.rolf.reward_scale))
                reward_info.add(info)

                if record_video:
                    frame_info = info.copy()
                    record_frames.append(self._render_frame(ep_len, ep_rew, frame_info))

        # Compute average/sum/max of information.
        ep_info = {"len": ep_len, "rew": ep_rew}
        if "episode_success_state" in reward_info.keys():
            ep_info["episode_success_state"] = reward_info["episode_success_state"]
        ep_info.update(reward_info.get_dict(reduction="max", only_scalar=True))

        Logger.info(
            "rollout: %s",
            {
                k: v
                for k, v in ep_info.items()
                if k not in self._exclude_rollout_log and np.isscalar(v)
            },
        )
        return rollout.get(), ep_info, record_frames
