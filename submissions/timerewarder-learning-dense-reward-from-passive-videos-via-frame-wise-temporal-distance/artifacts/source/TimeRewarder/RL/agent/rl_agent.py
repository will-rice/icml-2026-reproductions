import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd

import utils
from agent.encoder import Encoder
from models.simple_tokenizer import tokenize
from rewarder import optimal_transport_plan, cosine_distance, euclidean_distance, mask_optimal_transport_plan
from torchvision.transforms import Normalize

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))
        self.action_shape = action_shape[0]

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class RLAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, augment,
                 rewards, sinkhorn_rew_scale,
                 auto_rew_scale, auto_rew_scale_factor, suite_name, name, obs_type, update_policy_freq, expl_noise, use_clip=False, suc_signal_scale=10, suc_scale_calib_episodes=100):
        self.device = device
        self.lr = lr
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.update_policy_freq = update_policy_freq
        self.expl_noise = expl_noise
        self.augment = augment
        self.rewards = rewards
        self.sinkhorn_rew_scale = sinkhorn_rew_scale
        self.auto_rew_scale = auto_rew_scale
        self.auto_rew_scale_factor = auto_rew_scale_factor
        self.use_encoder = True if obs_type=='pixels' else False
        self.obs_type = obs_type
        self.computed_suc_scale = 0
        self.suc_scale = 0
        self.suc_signal_scale = suc_signal_scale
        self.suc_scale_calib_episodes = suc_scale_calib_episodes

        # models
        if self.use_encoder:
            self.encoder = Encoder(obs_shape).to(device)
            repr_dim = self.encoder.repr_dim
        else:
            repr_dim = obs_shape[0]

        self.actor = Actor(repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        if self.use_encoder:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = utils.RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()
        if use_clip:
            import clip  # openai CLIP: only needed for the use_clip side branch
            self.clipmodel = clip.load("ViT-L/14")[0].eval().to(device)
        else:
            self.clipmodel = None
        self.img_norm = Normalize(mean=torch.tensor([123.675, 116.28, 103.53]),
                                    std=torch.tensor([58.395, 57.12, 57.375]))
    def __repr__(self):
        return "rl_agent"

    def train(self, training=True):
        self.training = training
        if self.use_encoder:
            self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0)) if self.use_encoder else obs.unsqueeze(0)

        stddev = utils.schedule(self.expl_noise, step)
        dist = self.actor(obs, stddev)

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)

        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder and critic
        if self.use_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.use_encoder:
            self.encoder_opt.step()

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            
        return metrics

    def update_actor(self, obs, step, bc_ratio, obs_bc=None, action_bc=None):
        metrics = dict()

        if step > 1000 and bc_ratio > 0:
            bc_ratio = max(0, 1 - (step - 1000) / 2000)

        stddev = utils.schedule(self.stddev_schedule, step)

        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        actor_loss = - Q.mean() * (1 - bc_ratio)

        if bc_ratio > 0:
            stddev = 0.1
            dist_bc = self.actor(obs_bc, stddev)
            log_prob_bc = dist_bc.log_prob(action_bc).sum(-1, keepdim=True)
            actor_loss += -log_prob_bc.mean() * bc_ratio

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            metrics['actor_q'] = Q.mean().item()
            metrics['rl_loss'] = -Q.mean().item()
            if bc_ratio > 0:
                metrics['bc_loss'] = -log_prob_bc.mean().item()
                metrics['regularized_bc_loss'] = - log_prob_bc.mean().item() * bc_ratio
        return metrics

    def update(self, replay_iter, expert_replay_iter, step, bc_ratio):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        if self.use_encoder and self.augment:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
        else:
            obs = obs.float()
            next_obs = next_obs.float()

        if self.use_encoder:
            # encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

        if bc_ratio > 0:
            batch_bc = next(expert_replay_iter)
            obs_bc, action_bc = utils.to_torch(batch_bc, self.device)
            # augment
            if self.use_encoder and self.augment:
                obs_bc = self.aug(obs_bc.float())
            else:
                obs_bc = obs_bc.float()
            obs_bc = self.encoder(obs_bc) if self.use_encoder else obs_bc 
            # Detach grads
            obs_bc = obs_bc.detach()
        else:
            obs_bc = None 
            action_bc = None

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        if step % (self.update_every_steps * self.update_policy_freq) == 0:
            metrics.update(self.update_actor(obs.detach(), step, bc_ratio, obs_bc, action_bc))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def init_demos(self, demos, cost_encoder):
        self.cost_encoder = cost_encoder
        self.demos = demos

    def init_text(self, clip_text, cost_encoder):
        self.cost_encoder = cost_encoder
        if clip_text is None:
            self.text_feature = None
            return None
        clip_text = tokenize(clip_text).to(self.device)
        self.text_feature = cost_encoder.encode_text(clip_text)
        return self.text_feature

    def goal_rewarder(self, observations, goal_achieved, step, suc_signal=True):
        len_obs = observations.shape[0]
        reward = np.zeros(len_obs)
        if suc_signal:
            reward += goal_achieved * 1
        return reward

    def get_q_values(self, observations, actions):
        obs = torch.tensor(observations).to(self.device).float()
        obs = obs.detach()
        if self.use_encoder:
            with torch.no_grad():
                obs = self.encoder(obs)
        action = torch.tensor(actions).to(self.device).float()
        with torch.no_grad():
            q1, q2 = self.critic(obs, action)
        return q1.cpu().numpy(), q2.cpu().numpy()

    def clip_rewarder(self, observations, goal_achieved, step, suc_signal=True):
        # obs: (63, 9, 224, 224); text_feature: (1, 512)
        obs = torch.tensor(observations).to(self.device).float()
        obs = obs.detach()
        obs = obs[:, -3:]
        n, tc, h, w = obs.shape
        t = tc // 3
        obs = obs.view(n, t, 3, h, w)
        obs = obs.view(-1, 3, h, w)
        obs = self.img_norm(obs)
        obs = obs.view(n, t, 3, h, w)
        with torch.no_grad():
            # Bidirectional adjacent-frame reward (forward-pair minus reverse-pair).
            reward = self.cost_encoder.predict_progress(obs, self.text_feature)
        value = reward.cumsum(dim=0)

        raw_reward = reward.cpu().numpy()
        reward = raw_reward.copy()
        value = value.cpu().numpy()

        if suc_signal:
            if self.computed_suc_scale < self.suc_scale_calib_episodes:
                recent_scale = np.max(reward)
                self.suc_scale = max(self.suc_scale, recent_scale)
                self.computed_suc_scale += 1
            reward += goal_achieved * self.suc_scale * self.suc_signal_scale

        return reward, value, raw_reward, self.suc_signal_scale * self.suc_scale

    def ot_rewarder(self, observations, goal_achieved, step, return_infos=False, suc_signal=True, temporal_ot=False):
        scores_list = list()
        ot_rewards_list = list()
        if return_infos:
            cost_matrix_list = list()
            transport_plan_list = list()

        obs = torch.tensor(observations).to(self.device).float()
        obs = obs.detach()
        if self.use_encoder:
            with torch.no_grad():
                obs = self.cost_encoder(obs)
        
        for demo in self.demos:
            if not isinstance(demo, torch.Tensor):
                demo = torch.tensor(demo)
            exp = demo.to(self.device).float()
            exp = exp.detach()
            
            if self.rewards == 'sinkhorn_cosine':
                cost_matrix = cosine_distance(obs, exp)
            elif self.rewards == 'sinkhorn_euclidean':
                cost_matrix = euclidean_distance(obs, exp)
            else:
                raise NotImplementedError()
            if return_infos:
                cost_matrix_list.append(cost_matrix.detach().cpu().numpy())
            if temporal_ot:
                tot_mask = np.triu(np.tril(np.ones(cost_matrix.shape), k=10), k=-10)
                transport_plan = mask_optimal_transport_plan(obs,
                                                        exp,
                                                        cost_matrix,
                                                        tot_mask,
                                                        niter=100,
                                                        epsilon=0.01)
            else:
                transport_plan = optimal_transport_plan(
                    obs, exp, cost_matrix, method='sinkhorn',
                    niter=100).float()  # Getting optimal coupling
            ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
                torch.mm(transport_plan, cost_matrix.T)).detach().cpu().numpy()

            if return_infos:
                transport_plan_list.append(transport_plan.detach().cpu().numpy())
            scores_list.append(np.sum(ot_rewards) / self.sinkhorn_rew_scale)
            ot_rewards_list.append(ot_rewards)

        closest_demo_index = np.argmax(scores_list)
        rewards = ot_rewards_list[closest_demo_index]
        
        if suc_signal:
            if self.computed_suc_scale < self.suc_scale_calib_episodes:
                recent_scale = np.max(rewards)
                self.suc_scale = max(self.suc_scale, recent_scale)
                self.computed_suc_scale += 1
            rewards += goal_achieved * self.suc_scale * self.suc_signal_scale

        if return_infos:
            return cost_matrix_list[closest_demo_index], transport_plan_list[closest_demo_index], rewards, np.max(scores_list)
        return rewards
    def save_snapshot(self):
        keys_to_save = ['actor', 'critic', 'actor_opt', 'critic_opt', 'sinkhorn_rew_scale']
        if self.use_encoder:
            keys_to_save += ['encoder', 'encoder_opt']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        return payload

    def load_snapshot(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Update optimizers
        if self.use_encoder:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
            self.encoder_opt.load_state_dict(payload['encoder_opt'].state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.actor_opt.load_state_dict(payload['actor_opt'].state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.critic_opt.load_state_dict(payload['critic_opt'].state_dict())

