#!/usr/bin/env python3
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_clip_repo = os.environ.get("CLIP_REPO")
if _clip_repo:
    sys.path.insert(0, _clip_repo)

import warnings

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
import torch
from dm_env import specs
import math

import utils
from logger import Logger, log_formats_for_cost_encoder
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_expert_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import pickle
import matplotlib.pyplot as plt
from modules.cost_encoder import get_cost_encoder, resolve_timerewarder_use_text

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def _to_float32_env_rewards(rewards_list):
    """Normalize per-step rewards to a 1D float32 vector for np.stack."""
    out = []
    for r in rewards_list:
        if r is None:
            out.append(0.0)
        elif hasattr(r, "detach"):
            out.append(float(r.detach().float().reshape(-1).cpu()[0]))
        else:
            out.append(float(np.asarray(r, dtype=np.float64).reshape(-1)[0]))
    return np.asarray(out, dtype=np.float32)


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec[cfg.obs_type].shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)

class WorkspaceIL:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), cfg.agent)

        if repr(self.agent) == 'bc':
            self.cfg.suite.num_train_frames = self.cfg.num_train_frames_bc
            print('num_train_frames_bc:', self.cfg.suite.num_train_frames)
            self.cfg.suite.num_seed_frames = 0

        if self.cfg.adaptive_discount:
            self.cfg.ads.horizon = self.env_horizon
            self.ads = hydra.utils.instantiate(self.cfg.ads)

        if self.cfg.bc_ratio > 0:
            self.expert_replay_loader = make_expert_replay_loader(
                self.cfg.expert_dataset_bc, self.cfg.expert_batch_size, self.cfg.num_demos_bc, self.cfg.obs_type)
            self.expert_replay_iter = iter(self.expert_replay_loader)
        
        if repr(self.agent) == 'gaifo' or repr(self.agent) == 'bc':
            self.expert_replay_loader = make_expert_replay_loader(
                self.cfg.expert_dataset, self.cfg.expert_batch_size, self.cfg.num_demos, self.cfg.obs_type)
            self.expert_replay_iter = iter(self.expert_replay_loader)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        # expert_pixel is 224*224, expert_demo is 84*84
        if self.cfg.bc_ratio > 0 or self.cfg.cost_encoder == 'resnet':
            with open(self.cfg.expert_dataset, 'rb') as f:
                data = pickle.load(f)
                if self.cfg.obs_type == 'pixels':
                    self.expert_demo, _, self.expert_action, self.expert_reward, self.expert_pixel = data
                    self.expert_pixel = self.expert_pixel[:self.cfg.num_demos]
                elif self.cfg.obs_type == 'features':
                    _, self.expert_demo, self.expert_action, self.expert_reward, self.expert_pixel = data
            self.expert_demo = self.expert_demo[:self.cfg.num_demos]
            self.expert_action = self.expert_action[:self.cfg.num_demos]
            try:
                self.expert_reward = np.mean(self.expert_reward[:self.cfg.num_demos])
            except:
                self.expert_reward = np.mean(self.expert_reward[0])

            if self.cfg.obs_type == 'pixels':
                for i in range(len(self.expert_pixel)):
                    self.expert_pixel[i] = self.expert_pixel[i][::self.cfg.suite.action_repeat]
        if repr(self.agent) == 'rl_agent':
            cost_encoder = get_cost_encoder(
                self.cfg.cost_encoder,
                self.cfg.device,
                self.cfg.cost_encoder_ckpt,
                self.cfg.task_name,
                timerewarder_use_text=getattr(self.cfg, "timerewarder_use_text", False),
                timerewarder_bin_num=getattr(self.cfg, "timerewarder_bin_num", 20),
                timerewarder_diff_representation=getattr(
                    self.cfg, "timerewarder_diff_representation", False),
                timerewarder_mit_layers=getattr(self.cfg, "timerewarder_mit_layers", 1),
                timerewarder_parse_ckpt_hints=getattr(
                    self.cfg, "timerewarder_parse_ckpt_hints", False),
                timerewarder_t_override=getattr(self.cfg, "timerewarder_t_override", 0),
            )
        if self.cfg.cost_encoder == 'resnet':
            with torch.no_grad():
                demos = [cost_encoder(torch.tensor(demo).to(self.device)) for demo in self.expert_pixel]
            if self.cfg.obs_type == 'pixels':
                if repr(self.agent) == 'rl_agent':
                    self.agent.init_demos(demos, cost_encoder)
                if self.cfg.adaptive_discount:
                    self.ads.init_demos(demos, 'pixels', cost_encoder)
                for i in range(len(self.expert_demo)):
                    self.expert_demo[i] = self.expert_demo[i][::self.cfg.suite.action_repeat]
            else:
                if repr(self.agent) == 'rl_agent':
                    self.agent.init_demos(self.expert_demo, None)
                if self.cfg.adaptive_discount:
                    self.ads.init_demos(demos, 'features', None)
        elif self.cfg.cost_encoder == 'timerewarder':
            use_tr_text = resolve_timerewarder_use_text(
                self.cfg.cost_encoder_ckpt,
                getattr(self.cfg, "timerewarder_use_text", False),
                getattr(self.cfg, "timerewarder_parse_ckpt_hints", False),
            )
            clip_text_dict = {
                'drawer-open-v2': 'Opening drawer',
                'door-open-v2': 'Opening door',
                'window-open-v2': 'Opening window',
                'window-close-v2': 'Closing window',
                'button-press-topdown-v2': 'Pressing button from top',
                'lever-pull-v2': 'Pulling lever',
                'plate-slide-v2': 'Sliding plate into gate',
                'stick-push-v2': 'Pushing kettle with stick',
                'basketball-v3': 'Moving ball to above basket',
                'disassemble-v2': 'Disassembling ring from rod',
            }
            if use_tr_text:
                clip_text = clip_text_dict[self.cfg.task_name]
                text_feature = self.agent.init_text(clip_text, cost_encoder)
            else:
                text_feature = self.agent.init_text(None, cost_encoder)
            if self.cfg.adaptive_discount:
                self.ads.init_encoder(
                    cost_encoder,
                    use_clip=True,
                    text_feature=text_feature if use_tr_text else None,
                )

    def setup(self):
        # create logger
        train_fmt, eval_fmt = log_formats_for_cost_encoder(self.cfg.cost_encoder)
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb,
                             train_format=train_fmt, eval_format=eval_fmt)
        # create envs
        self.train_env, self.env_horizon = hydra.utils.call(self.cfg.suite.task_make_fn)
        self.eval_env, self.env_horizon = hydra.utils.call(self.cfg.suite.task_make_fn)
        self.env_horizon = math.ceil(self.env_horizon / self.cfg.suite.action_repeat)

        # create replay buffer
        data_specs = [
            self.train_env.observation_spec()[self.cfg.obs_type],
            self.train_env.action_spec(),
            specs.Array((1, ), np.float32, 'reward'),
            specs.Array((1, ), np.float32, 'discount')
        ]

        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_experiences, self.cfg.nstep, self.cfg.suite.discount)

        self._replay_iter = None
        self.expert_replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    TRAIN_VIDEO_EVERY_N_EPISODES = 100

    def _save_train_episode_video(self, proxy_reward, env_rewards):
        if not self.cfg.save_train_video:
            return
        if self._global_episode % self.TRAIN_VIDEO_EVERY_N_EPISODES != 0:
            return
        self.train_video_recorder.save_episode(str(self.global_frame), proxy_reward, env_rewards)

    def _log_timerewarder_reward_curves(self, observations, actions, env_rewards, env_value,
                                        new_rewards_clip, value_clip, ori_value, reward_scale):
        q1_value, q2_value = self.agent.get_q_values(observations, actions[1:])
        plt.clf()
        plt.figure(figsize=(10, 5))
        plt.subplot(231)
        plt.plot(q1_value, label='Q1')
        plt.plot(q2_value, label='Q2')
        plt.title('Q values')
        plt.legend()
        plt.subplot(234)
        plt.plot(ori_value)
        plt.title('reward (pre-success)')
        plt.subplot(232)
        plt.plot(new_rewards_clip)
        plt.title('reward (adjacent-frame score)')
        plt.subplot(235)
        plt.plot(value_clip)
        plt.title('value=cumsum(reward)')
        plt.subplot(233)
        plt.plot(env_rewards)
        plt.title('env reward curve')
        plt.subplot(236)
        plt.plot(env_value)
        plt.title('env value curve')
        plt.suptitle(f'reward success scale: {reward_scale:.3f}')
        plt.tight_layout()
        train_video_dir = self.work_dir / 'train_video'
        train_video_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(train_video_dir / f'{self.global_frame}_visualize.jpg')
        plt.close()

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)

        if self.cfg.suite.name == 'metaworld':
            paths = []
        costs = []
        while eval_until_episode(episode):
            if self.cfg.suite.name == 'metaworld':
                path = []
            time_step = self.eval_env.reset()
            observations = []
            pixels = []
            self.video_recorder.init(enabled=(episode == 0))
            episode_step = 0
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation[self.cfg.obs_type], self.global_step, eval_mode=True)
                observations.append(time_step.observation[self.cfg.obs_type])
                pixels.append(time_step.observation['pixels_large'])
                time_step = self.eval_env.step(action)
                if self.cfg.suite.name == 'metaworld':
                    path.append(time_step.observation['goal_achieved'])
                self.video_recorder.record(time_step.observation)
                total_reward += time_step.reward
                step += 1
                episode_step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')
            paths.append(1 if np.sum(path)>3 else 0)

            observations = np.stack(observations, 0)
            pixels = np.stack(pixels, 0)
            if self.cfg.adaptive_discount:
                if self.cfg.obs_type == 'features':
                    reward_obs = observations
                else:
                    reward_obs = pixels
                cost = self.ads.compute_cost(reward_obs)
                costs.append(cost)
                                
        if self.cfg.adaptive_discount:
            discount, ads_metrics = self.ads.update(costs)
            self.replay_storage.update_parameters({'_discount': discount})

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.suite.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            if self.cfg.cost_encoder == 'resnet':
                log('expert_reward', self.expert_reward)
            if self.cfg.suite.name == 'metaworld':
                log("success_percentage", np.mean(paths))
            if self.cfg.adaptive_discount:
                for k, v in ads_metrics.items():
                    log(k, v)
        
        if self.cfg.save_every_model:
            save_dir = self.work_dir / 'models'
            save_dir.mkdir(exist_ok=True)
            self.save_snapshot(save_dir / f'snapshot{self.global_frame}.pt')

    def train_il(self):
        # predicates
        train_until_step = utils.Until(self.cfg.suite.num_train_frames,
                                       self.cfg.suite.action_repeat)
        seed_until_step = utils.Until(self.cfg.suite.num_seed_frames,
                                      self.cfg.suite.action_repeat)
        eval_every_step = utils.Every(self.cfg.suite.eval_every_frames,
                                      self.cfg.suite.action_repeat)

        episode_step, episode_reward = 0, 0
        time_steps = list()
        observations = list()
        pixels = list()
        actions = list()
        goal_achieved = list()
        env_rewards = list()

        time_step = self.train_env.reset()
        time_steps.append(time_step)
        actions.append(time_step.action)

        if repr(self.agent) == 'rl_agent':
            if self.agent.auto_rew_scale:
                self.agent.sinkhorn_rew_scale = 1.  # Set after first episode

        self.train_video_recorder.init(time_step.observation['pixels_large'])
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # wait until all the metrics schema is populated
                observations = np.stack(observations, 0)
                pixels = np.stack(pixels, 0)
                goal_achieved = np.stack(goal_achieved, 0)
                env_rewards = _to_float32_env_rewards(env_rewards)

                env_value = np.cumsum(env_rewards)
                if self.cfg.cost_encoder == 'timerewarder':
                    reward_obs = pixels
                else:
                    if self.cfg.obs_type == 'features':
                        reward_obs = observations
                    else:
                        reward_obs = pixels
                actions = np.stack(actions, 0)
                if repr(self.agent) == 'rl_agent':
                    if self.cfg.cost_encoder == 'gt':
                        new_rewards_gt = self.agent.goal_rewarder(reward_obs, goal_achieved, self.global_step, suc_signal=self.cfg.suc_signal)
                        if env_rewards.shape != new_rewards_gt.shape:
                            ncol = env_rewards.shape[1] if env_rewards.ndim > 1 else 1
                            new_rewards_gt = np.tile(new_rewards_gt, (1, ncol)).T
                        new_rewards = env_rewards + new_rewards_gt
                    elif self.cfg.cost_encoder == 'resnet':
                        new_rewards = self.agent.ot_rewarder(reward_obs, goal_achieved, self.global_step, suc_signal=self.cfg.suc_signal)
                    elif self.cfg.cost_encoder == 'timerewarder':
                        new_rewards_clip, value_clip, ori_value, reward_scale = self.agent.clip_rewarder(
                            reward_obs, goal_achieved, self.global_step, suc_signal=self.cfg.suc_signal)
                        new_rewards = new_rewards_clip
                        if self.cfg.save_train_video and self._global_episode % self.TRAIN_VIDEO_EVERY_N_EPISODES == 0:
                            self._log_timerewarder_reward_curves(
                                observations, actions, env_rewards, env_value,
                                new_rewards_clip, value_clip, ori_value, reward_scale)
                    else:
                        raise NotImplementedError
                    self._save_train_episode_video(new_rewards, env_rewards)
                    new_rewards_sum = np.sum(new_rewards)

                elif repr(self.agent) == 'gaifo':
                    new_rewards = self.agent.gaifo_rewarder(observations, actions, goal_achieved, suc_signal=self.cfg.suc_signal)
                    self._save_train_episode_video(new_rewards, env_rewards)
                    new_rewards_sum = np.sum(new_rewards)
                
                if repr(self.agent) == 'rl_agent':
                    if self.agent.auto_rew_scale:
                        if self._global_episode == 1:
                            self.agent.sinkhorn_rew_scale = self.agent.sinkhorn_rew_scale * self.agent.auto_rew_scale_factor / float(np.abs(new_rewards_sum))
                            if self.cfg.cost_encoder == 'resnet':
                                new_rewards = self.agent.ot_rewarder(reward_obs, goal_achieved, self.global_step, suc_signal=self.cfg.suc_signal)
                            elif self.cfg.cost_encoder == 'timerewarder':
                                new_rewards_clip, value_clip, ori_value, reward_scale = self.agent.clip_rewarder(
                                    reward_obs, goal_achieved, self.global_step, suc_signal=self.cfg.suc_signal)
                                new_rewards = new_rewards_clip
                            new_rewards_sum = np.sum(new_rewards)

                for i, elt in enumerate(time_steps):
                    elt = elt._replace(
                        observation=time_steps[i].observation[self.cfg.obs_type])
                    if repr(self.agent) == 'rl_agent' or repr(self.agent) == 'gaifo':
                        if i == 0:
                            elt = elt._replace(reward=float('nan'))
                        else:
                            elt = elt._replace(reward=float(new_rewards[i - 1]))
                    self.replay_storage.add(elt)

                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.suite.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                        if (repr(self.agent) == 'rl_agent' or repr(self.agent) == 'gaifo') and self.cfg.cost_encoder == 'resnet':
                            log('expert_reward', self.expert_reward)
                            log('imitation_reward', new_rewards_sum)

                # reset env
                time_steps = list()
                observations = list()
                pixels = list()
                actions = list()
                goal_achieved = list()
                env_rewards = list()

                time_step = self.train_env.reset()
                time_steps.append(time_step)
                actions.append(time_step.action)
                self.train_video_recorder.init(time_step.observation['pixels_large'])
                # try to save snapshot
                if self.cfg.save_model:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval()
                
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation[self.cfg.obs_type], self.global_step, eval_mode=False)
                    

            # try to update the agent
            if not seed_until_step(self.global_step):
                # Update
                if repr(self.agent) == 'rl_agent':
                    metrics = self.agent.update(self.replay_iter, self.expert_replay_iter, self.global_step, self.cfg.bc_ratio)
                else:
                    metrics = self.agent.update(self.replay_iter, self.expert_replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
                    

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            env_rewards.append(time_step.reward)
            time_steps.append(time_step)
            observations.append(time_step.observation[self.cfg.obs_type])
            pixels.append(time_step.observation['pixels_large'])
            actions.append(time_step.action)
            goal_achieved.append(time_step.observation['goal_achieved'])

            self.train_video_recorder.record(time_step.observation['pixels_large'])
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self, save_dir=None):
        snapshot = self.work_dir / 'snapshot.pt'
        if save_dir is not None:
            snapshot = save_dir
        keys_to_save = ['timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, snapshot):
        # Warning: The replay buffer is not loaded.
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            agent_payload[k] = v
        self.agent.load_snapshot(agent_payload)

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import WorkspaceIL as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    
    # Load weights
    if cfg.load_checkpoint:
        snapshot = Path(cfg.checkpoint_path)
        if snapshot.exists():
            print(f'resuming checkpoint: {snapshot}')
            workspace.load_snapshot(snapshot)
    workspace.train_il()

    # remove *.npz files
    if not cfg.save_experiences:
        remove_dir = workspace.work_dir / 'buffer'
        for fn in remove_dir.glob('*.npz'):
            os.remove(fn)


if __name__ == '__main__':
    main()
