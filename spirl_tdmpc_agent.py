from collections import OrderedDict

import gym.spaces
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from rolf.algorithms import BaseAgent, TDMPCAgent
from rolf.algorithms.dataset import ReplayBufferEpisode, SeqSampler
from rolf.utils import Logger, Info
from rolf.utils.pytorch import soft_copy_network, to_tensor, optimizer_cuda
from rolf.networks.distributions import mc_kl

from spirl_tdmpc_rollout import SPiRLTDMPCRolloutRunner
from spirl_agent import SPiRLAgent


class SPiRLTDMPCAgent(BaseAgent):
    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)
        self._ob_space = ob_space
        self._ac_space = ac_space

        # Build networks
        meta_ob_space = ob_space
        meta_ac_space = gym.spaces.Box(-2, 2, [cfg.skill_dim])
        ob_space = gym.spaces.Dict(OrderedDict(ob_space.spaces))
        self.skill_agent = SPiRLAgent(cfg, ob_space, ac_space)
        self.meta_agent = TDMPCPriorAgent(
            cfg,
            meta_ob_space,
            meta_ac_space,
            self.skill_agent.hl_agent.policy.prior_net,
        )

        # Per-episode replay buffer
        sampler = SeqSampler(cfg.meta_batch_length)
        meta_buffer_keys = ["ob", "ac", "rew", "skill_len", "done"]
        self._meta_buffer = ReplayBufferEpisode(
            meta_buffer_keys, cfg.buffer_size, sampler.sample_func, cfg.precision
        )
        self.meta_agent.set_buffer(self._meta_buffer)
        buffer_keys = ["ob", "ac", "done"]
        self._buffer = ReplayBufferEpisode(
            buffer_keys, cfg.buffer_size, sampler.sample_func, cfg.precision
        )
        self.skill_agent.set_buffer(self._buffer)

        if cfg.phase == "rl" and cfg.pretrain_ckpt_path is not None:
            Logger.warning(f"Load pretrained checkpoint {cfg.pretrain_ckpt_path}")
            ckpt = torch.load(cfg.pretrain_ckpt_path, map_location=self._device)
            ckpt = ckpt["agent"]
            ckpt["meta_agent"]["skill_prior"] = ckpt["meta_agent"].copy()
            self.load_state_dict(ckpt)
        else:
            Logger.warning("No pretrained checkpoint found")

    def get_runner(self, cfg, env, env_eval):
        return SPiRLTDMPCRolloutRunner(cfg, env, env_eval, self)

    def is_off_policy(self):
        return True

    def store_episode(self, rollouts):
        self._meta_buffer.store_episode(rollouts[0], include_last_ob=False)
        self._buffer.store_episode(rollouts[1])

    def state_dict(self):
        return {
            "meta_agent": self.meta_agent.state_dict(),
            "skill_agent": self.skill_agent.state_dict(),
            "ob_norm": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self.meta_agent.load_state_dict(ckpt["meta_agent"])
        self.skill_agent.load_state_dict(ckpt["skill_agent"])
        self.to(self._device)

    def update(self):
        train_info = Info()
        for _ in range(self._cfg.train_iter):
            meta_train_info = self.meta_agent.update()
            train_info.add(meta_train_info)
        return train_info.get_dict()


class TDMPCPriorAgent(TDMPCAgent):
    def __init__(self, cfg, ob_space, ac_space, prior_net):
        super().__init__(cfg, ob_space, ac_space)

        self._prior_net = prior_net.to(self._device)
        self._o_prev = None
        self.mse = torch.nn.MSELoss()

        self._log_alpha = torch.tensor(
            np.log(cfg.alpha_init_temperature), requires_grad=True, device=self._device,
        )
        self._alpha_optim = optim.Adam(
            [self._log_alpha], lr=cfg.alpha_lr, betas=(0.5, 0.999)
        )
        optimizer_cuda(self._alpha_optim, self._device)

    def _compute_prior_divergence(self, o):
        # compute the predicted skill distribution from the actor
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        state = self.model.encoder(o).detach()
        _, actor_dist = self.actor.act(
            flatten(state), std=self._cfg.min_std, return_dist=True
        )

        # compute the predicted skill distribution from the prior
        if self._cfg.pixel_ob:
            o_prev = self._o_prev if self._o_prev is not None else o.copy()
            obs = flatten(torch.cat([o_prev["image"], o["image"]], dim=-1)).permute(
                0, 3, 1, 2
            )
        else:
            obs = flatten(o["ob"])
        prior_dist = self._prior_net.compute_learned_prior(
            obs, first_only=True
        ).detach()

        # compute the KL divergence and clip it
        kl_div = mc_kl(actor_dist, prior_dist, scale=2.0)
        skill_prior_loss = torch.clamp(
            kl_div, -self._cfg.max_divergence, self._cfg.max_divergence
        )

        # prepare for the next call
        if self._cfg.pixel_ob:
            self._o_prev = o.copy()

        return skill_prior_loss

    def prior_act(self, ob_prev, ob):
        if self._cfg.pixel_ob:
            obs = np.concatenate([ob_prev["image"], ob["image"]], 2)
            obs = obs.transpose(2, 0, 1) / 127.5 - 1
        else:
            obs = ob["ob"]
        obs = to_tensor(obs, self._device, self._dtype)[None]

        self._prior_net.eval()
        prior_dist = self._prior_net.compute_learned_prior(
            obs, first_only=True
        ).detach()

        z = prior_dist.sample().cpu().numpy()
        return z.squeeze(0)

    def preprocess(self, ob):
        if isinstance(ob, torch.Tensor):
            if self._cfg.env == "maze":
                shape = ob.shape
                ob = ob.view(-1, shape[-1])
                ob = torch.cat([ob[k][:, :2] / 40 - 0.5, ob[k][:, 2:] / 10], -1)
                ob = ob.view(shape)
            return ob
        ob = ob.copy()
        for k, v in ob.items():
            if len(v.shape) >= 4:
                ob[k] = ob[k] / 255.0 - 0.5
            elif self._cfg.env == "maze":
                shape = ob[k].shape
                ob[k] = ob[k].view(-1, shape[-1])
                ob[k] = torch.cat([ob[k][:, :2] / 40 - 0.5, ob[k][:, 2:] / 10], -1)
                ob[k] = ob[k].view(shape)
        return ob

    def preprocess1(self, ob):
        ob = ob.copy()
        for k, v in ob.items():
            if len(v.shape) >= 4:
                ob[k] = ob[k] / 127.5 - 1
        return ob

    def state_dict(self):
        return {
            "log_alpha": self._log_alpha.cpu().detach().numpy(),
            "model": self.model.state_dict(),
            "model_target": self.model_target.state_dict(),
            "actor": self.actor.state_dict(),
            "alpha_optim": self._alpha_optim.state_dict(),
            "model_optim": self._model_optim.state_dict(),
            "actor_optim": self._actor_optim.state_dict(),
            "ob_norm": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        # load alpha and optimizer state
        if "log_alpha" not in ckpt:
            missing = self.actor.load_state_dict(ckpt["actor_state_dict"], strict=False)
            for missing_key in missing.missing_keys:
                if "stds" not in missing_key:
                    Logger.warning("Missing key", missing_key)
            if len(missing.unexpected_keys) > 0:
                Logger.warning("Unexpected keys", missing.unexpected_keys)
            self.to(self._device)
            return

        self._log_alpha.data = torch.tensor(
            ckpt["log_alpha"], requires_grad=True, device=self._device
        )
        self._alpha_optim.load_state_dict(ckpt["alpha_optim"])
        optimizer_cuda(self._alpha_optim, self._device)

        super().load_state_dict(ckpt)

    def _update_alpha(self, prior_div, info):
        if self._cfg.fixed_alpha is not None:
            info["alpha"] = self._cfg.fixed_alpha
            return self._cfg.fixed_alpha

        alpha = self._log_alpha.exp()

        # update alpha
        alpha_loss = alpha * (self._cfg.target_divergence - prior_div).detach().mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

        info["alpha"] = alpha.cpu().item()
        info["alpha_loss"] = alpha_loss.cpu().item()

        return alpha.detach()

    def _update_network(self, batch):
        cfg = self._cfg
        info = Info()
        mse = nn.MSELoss(reduction="none")

        o = to_tensor(batch["ob"], self._device, self._dtype)
        ac = to_tensor(batch["ac"], self._device, self._dtype)
        rew = to_tensor(batch["rew"], self._device, self._dtype)
        o = self.preprocess(o)

        with torch.autocast(self._cfg.device, enabled=self._use_amp):
            # compute the divergence of predicted skill distribution between actor and skill prior
            prior_div = self._compute_prior_divergence(o)
            alpha = self._update_alpha(prior_div, info)

        # Flip dimensions, BxT -> TxB
        def flip(x, l=None):
            if isinstance(x, dict):
                return [{k: v[:, t] for k, v in x.items()} for t in range(l)]
            else:
                return x.transpose(0, 1)

        o = flip(o, cfg.horizon)
        ac = flip(ac)
        rew = flip(rew)

        with torch.autocast(self._cfg.device, enabled=self._use_amp):
            z = z_next_pred = self.model.encoder(o[0])
            zs = [z.detach()]

            consistency_loss = 0
            reward_loss = 0
            value_loss = 0
            for t in range(cfg.horizon - 1):
                z = z_next_pred
                q_pred = self.model.critic(z, ac[t])
                z_next_pred, reward_pred = self.model.imagine_step(z, ac[t])
                with torch.no_grad():
                    # `z` for contrastive learning
                    z_next = self.model_target.encoder(o[t + 1])

                    # `z` for `q_target`
                    z_next_q = self.model.encoder(o[t + 1])
                    ac_next = self.actor(z_next_q, cfg.min_std)
                    q_next = torch.min(*self.model_target.critic(z_next_q, ac_next))
                    # q_target = rew[t] + (1 - done[t]) * cfg.rl_discount * q_next
                    q_target = rew[t] + cfg.rl_discount * q_next
                zs.append(z_next_pred.detach())

                rho = cfg.rho ** t
                consistency_loss += rho * mse(z_next_pred, z_next).mean(dim=1)
                reward_loss += rho * mse(reward_pred, rew[t])
                value_loss += rho * (
                    mse(q_pred[0], q_target) + mse(q_pred[1], q_target)
                )
            model_loss = (
                cfg.consistency_coef * consistency_loss.clamp(max=1e4)
                + cfg.reward_coef * reward_loss.clamp(max=1e4)
                + cfg.value_coef * value_loss.clamp(max=1e4)
            ).mean()
            model_loss.register_hook(lambda grad: grad * (1 / cfg.horizon))  # CHECK
        model_grad_norm = self._model_optim.step(model_loss)

        with torch.autocast(self._cfg.device, enabled=self._use_amp):
            # self.model.critic.requires_grad_(False)  # CHECK
            actor_loss = 0
            for t, z in enumerate(zs):
                a = self.actor(z, cfg.min_std)
                rho = cfg.rho ** t
                actor_loss += (
                    -rho * torch.min(*self.model.critic(z, a)).mean()
                    + alpha * prior_div.mean()
                )
        actor_grad_norm = self._actor_optim.step(actor_loss)
        # self.model.critic.requires_grad_(True)  # CHECK

        self._update_iter += 1
        if self._update_iter % cfg.target_update_freq == 0:
            soft_copy_network(self.model_target, self.model, cfg.target_update_tau)

        info["min_q_target"] = q_target.min().item()
        info["q_target"] = q_target.mean().item()
        info["min_q_pred1"] = q_pred[0].min().item()
        info["min_q_pred2"] = q_pred[1].min().item()
        info["q_pred1"] = q_pred[0].mean().item()
        info["q_pred2"] = q_pred[1].mean().item()
        info["model_grad_norm"] = model_grad_norm.item()
        info["actor_grad_norm"] = actor_grad_norm.item()
        info["actor_loss"] = actor_loss.mean().item()
        info["model_loss"] = model_loss.mean().item()
        info["consistency_loss"] = consistency_loss.mean().item()
        info["reward_loss"] = reward_loss.mean().item()
        info["value_loss"] = value_loss.mean().item()

        return info.get_dict()
