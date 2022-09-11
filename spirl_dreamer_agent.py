from collections import OrderedDict

import gym.spaces
import numpy as np
import torch
import torch.optim as optim

from rolf.algorithms import BaseAgent, DreamerAgent
from rolf.algorithms.dataset import ReplayBufferEpisode, SeqSampler
from rolf.utils import Logger, Info
from rolf.utils.pytorch import to_tensor, optimizer_cuda, RequiresGrad
from rolf.utils.dreamer import lambda_return
from rolf.networks.distributions import mc_kl

from spirl_dreamer_rollout import SPiRLDreamerRolloutRunner
from spirl_agent import SPiRLAgent


class SPiRLDreamerAgent(BaseAgent):
    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)
        self._ob_space = ob_space
        self._ac_space = ac_space

        # Build networks
        meta_ob_space = ob_space
        meta_ac_space = gym.spaces.Box(-2, 2, [cfg.skill_dim])
        ob_space = gym.spaces.Dict(OrderedDict(ob_space.spaces))
        self.skill_agent = SPiRLAgent(cfg, ob_space, ac_space)

        # let a dreamer agent act in the skill space, while regularized by the learned skill prior from spirl
        self.meta_agent = DreamerPriorAgent(
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
        return SPiRLDreamerRolloutRunner(cfg, env, env_eval, self)

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


class DreamerPriorAgent(DreamerAgent):
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

    def _compute_prior_divergence(self, post, o):
        # compute the predicted skill distribution from the actor
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        state = {k: flatten(v) for k, v in post.items()}
        pred_z, actor_dist = self.actor.act(
            self.model.dynamics.get_feat(state).detach(), return_dist=True
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
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "alpha_optim": self._alpha_optim.state_dict(),
            "model_optim": self.model_optim.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
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

    def _update_network(self, batch, log_image=False):
        info = Info()

        o_orig = to_tensor(batch["ob"], self._device, self._dtype)
        ac = to_tensor(batch["ac"], self._device, self._dtype)
        rew = to_tensor(batch["rew"], self._device, self._dtype)
        o = self.preprocess(o_orig)
        o_prior = self.preprocess1(o_orig)

        # Compute model loss
        with RequiresGrad(self.model):
            with torch.autocast(self._cfg.device, enabled=self._use_amp):
                embed = self.model.encoder(o)
                post, prior = self.model.dynamics.observe(embed, ac)
                feat = self.model.dynamics.get_feat(post)

                ob_pred = self.model.decoder(feat)
                recon_losses = {k: -ob_pred[k].log_prob(v).mean() for k, v in o.items()}
                recon_loss = sum(recon_losses.values())

                reward_pred = self.model.reward(feat)
                reward_loss = -reward_pred.log_prob(rew.unsqueeze(-1)).mean()

                prior_dist = self.model.dynamics.get_dist(prior)
                post_dist = self.model.dynamics.get_dist(post)

                # Clipping KL divergence after taking mean (from official code)
                div = torch.distributions.kl.kl_divergence(post_dist, prior_dist).mean()
                div_clipped = torch.clamp(div, min=self._cfg.free_nats)
                model_loss = self._cfg.kl_scale * div_clipped + recon_loss + reward_loss
            model_grad_norm = self.model_optim.step(model_loss)

        # Compute actor loss with imaginary rollout
        with RequiresGrad(self.actor):
            with torch.autocast(self._cfg.device, enabled=self._use_amp):
                post = {k: v.detach() for k, v in post.items()}

                # compute the divergence of predicted skill distribution between actor and skill prior
                prior_div = self._compute_prior_divergence(post, o_prior)
                alpha = self._update_alpha(prior_div, info)

                imagine_feat = self._imagine_ahead(post)
                imagine_reward = (
                    self.model.reward(imagine_feat).mode().squeeze(-1).float()
                )
                imagine_value = self.critic(imagine_feat).mode().squeeze(-1).float()
                pcont = self._cfg.rl_discount * torch.ones_like(imagine_reward)
                imagine_return = lambda_return(
                    imagine_reward[:-1],
                    imagine_value[:-1],
                    pcont[:-1],
                    bootstrap=imagine_value[-1],
                    lambda_=self._cfg.gae_lambda,
                )
                with torch.no_grad():
                    discount = torch.cumprod(
                        torch.cat([torch.ones_like(pcont[:1]), pcont[:-2]], 0), 0
                    )

                actor_loss = (
                    -(discount * imagine_return).mean() + alpha * prior_div.mean()
                )
            actor_grad_norm = self.actor_optim.step(actor_loss)

        # Compute critic loss
        with RequiresGrad(self.critic):
            with torch.autocast(self._cfg.device, enabled=self._use_amp):
                value_pred = self.critic(imagine_feat.detach()[:-1])
                target = imagine_return.detach().unsqueeze(-1)
                critic_loss = -(discount * value_pred.log_prob(target)).mean()
            critic_grad_norm = self.critic_optim.step(critic_loss)

        # Log scalar
        for k, v in recon_losses.items():
            info[f"recon_loss_{k}"] = v.item()
        info["reward_loss"] = reward_loss.item()
        info["prior_entropy"] = prior_dist.entropy().mean().item()
        info["posterior_entropy"] = post_dist.entropy().mean().item()
        info["kl_loss"] = div_clipped.item()
        info["model_loss"] = model_loss.item()
        info["actor_loss"] = actor_loss.item()
        info["critic_loss"] = critic_loss.item()
        info["value_target"] = imagine_return.mean().item()
        info["value_predicted"] = value_pred.mode().mean().item()
        info["model_grad_norm"] = model_grad_norm.item()
        info["actor_grad_norm"] = actor_grad_norm.item()
        info["critic_grad_norm"] = critic_grad_norm.item()
        info["actor_entropy"] = self.actor(feat).entropy().mean().item()
        info["prior_div"] = prior_div.mean().item()

        if log_image and self._cfg.pixel_ob and self._cfg.log_image:
            with torch.no_grad(), torch.autocast(
                self._cfg.device, enabled=self._use_amp
            ):

                # 5 timesteps for each of 4 samples
                init, _ = self.model.dynamics.observe(embed[:4, :5], ac[:4, :5])
                init = {k: v[:, -1] for k, v in init.items()}
                prior = self.model.dynamics.imagine(ac[:4, 5:], init)
                openloop = self.model.decoder(
                    self.model.dynamics.get_feat(prior)
                ).mode()
                for k, v in o.items():
                    if len(v.shape) != 5:
                        continue
                    truth = o[k][:4] + 0.5
                    recon = ob_pred[k].mode()[:4]
                    model = torch.cat([recon[:, :5] + 0.5, openloop[k] + 0.5], 1)
                    error = (model - truth + 1) / 2
                    openloop = torch.cat([truth, model, error], 2)
                    img = openloop.detach().cpu().numpy() * 255
                    info[f"recon_{k}"] = img.transpose(0, 1, 4, 2, 3).astype(np.uint8)

        return info.get_dict()
