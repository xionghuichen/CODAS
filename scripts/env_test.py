# Created by xionghuichen at 2022/6/24
# Email: chenxh@lamda.nju.edu.cn
import argparse
import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
import pickle
import os.path as osp
import tqdm
from typing import Union
import gym
import ast

from codas.utils import tf_util as U
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.vec_env import sync_envs_normalization, VecEnv

from RLA.easy_log.tester import tester

from codas.utils.functions import *
from codas.utils.config import *
from codas.wrapper.env_wrapper import GeneratorWrapper, make_vec_env
from codas.train.rollout import Runner
from codas.rl.ppo2 import PPO2
from private import *
from env_config_map import env_config_map


def get_param():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Variational Sequence")
    parser.add_argument('--seed', help='RNG seed', type=int, default=88)
    parser.add_argument('--info', type=str, default='')
    parser.add_argument('--expert_root', type=str, default=DATA_ROOT)
    parser.add_argument('--task', type=str, default='data_collect')
    # new params
    parser.add_argument('--env_id', help='environment ID', default='InvertedDouble-v5')
    parser.add_argument('--num_env', default=1, type=int)

    parser.add_argument('--policy_timestep', type=int, default=1000000)
    parser.add_argument('--collect_trajs', type=int, default=COLLECT_TRAJ)
    parser.add_argument('--max_sequence', type=int, default=1002)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--dynamic_param', type=float, default=1.0)

    parser.add_argument('--deter', type=ast.literal_eval, default=False)
    parser.add_argument('--auto_env_map', type=ast.literal_eval, default=True)
    parser.add_argument('--log_tb', type=ast.literal_eval, default=False)
    parser.add_argument('--log_dir', type=str, default="../log/")
    parser.add_argument('--start_fraction', type=float, default=0.05)
    parser.add_argument('--end_fraction', type=float, default=1.0)
    parser.add_argument('--trajs_per_callback', type=int, default=2)
    parser.add_argument('--tot_training_trajs', type=int, default=COLLECT_SIM_TRAJ)
    parser.add_argument('--stoc_init_range', type=float, default=0.005)
    args = parser.parse_args()
    kwargs = vars(args)
    if kwargs['auto_env_map'] and kwargs['env_id'] in env_config_map:
        kwargs.update(env_config_map[kwargs['env_id']])
    args = argparse.Namespace(**kwargs)
    return args


class CollectStateCallback(BaseCallback):
    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 trajs_per_callback: int = 5,
                 tot_steps: int = 1000000, start_fraction: float = 0.5,
                 end_fraction: float = 0.5, tot_trajs_to_collect=600,
                 deterministic: bool = True,
                 max_horizon: int = 1002,
                 output_path: str = "/tmp/codas_callback_cache.npz", verbose: int = 1):
        super(CollectStateCallback, self).__init__(verbose=verbose)
        self.tot_steps = tot_steps
        self.start_fraction = start_fraction
        self.output_path = output_path
        self.traj_obs = []
        self.traj_action = []
        self.traj_length = []
        self.traj_return = []
        self.collected_traj_lengths = []
        self.eval_env = eval_env
        self.deterministic = deterministic
        self.tot_trajs_to_collect = tot_trajs_to_collect
        self.max_horizon = max_horizon
        # calculate collect interval and trajs per collection
        num_validate_callbacks = tot_trajs_to_collect // trajs_per_callback
        validate_fraction = max(0, end_fraction - start_fraction)
        self.callback_interval = int((tot_steps * validate_fraction) // num_validate_callbacks)
        self.start_callback_step = int(tot_steps * start_fraction)
        self.start_callback_id = 0
        self.final_callback_id = num_validate_callbacks

        self.trajs_per_callback = trajs_per_callback
        self.curr_callback_id = 0
        print("\033[32msetting data collection callback\033[0m:\n"
              "\ttotal callbacks:{}\n"
              "\tstarting timestep:{}\n"
              "\teval freq:{}\n"
              "\ttrajs_per_callback:{}".format(
            self.final_callback_id, self.start_callback_step, self.callback_interval, trajs_per_callback))

    def _on_step(self):
        curr_timestep = self.n_calls + 1
        # if curr_timestep > 8192:
        tester.time_step_holder.set_time(curr_timestep)
        if curr_timestep <= self.start_callback_step or \
                ((curr_timestep - self.start_callback_step) % self.callback_interval) != 0 \
                or self.curr_callback_id >= self.final_callback_id:
            return True
        # collect trajectories
        self.curr_callback_id += 1
        if self.curr_callback_id < self.final_callback_id:
            num_trajs_to_collect = self.trajs_per_callback
        else:
            num_trajs_to_collect = self.tot_trajs_to_collect - len(self.traj_obs)
        obs, actions, lengths, returns = self.collect_trajs(num_trajs_to_collect)
        self.traj_obs += obs
        self.traj_action += actions
        self.traj_length += lengths
        self.traj_return += returns
        if self.curr_callback_id >= self.final_callback_id:
            # write collected trajs to temp file
            if len(self.traj_obs) > self.tot_trajs_to_collect:
                self.traj_obs = self.traj_obs[:self.tot_trajs_to_collect]
                self.traj_length = self.traj_length[:self.tot_tra]
            np.savez(self.output_path, obs=self.traj_obs, acs=self.traj_action, traj_len=self.traj_length,
                     ep_rets=self.traj_return)
            print("saving trajs:", len(self.traj_obs), self.traj_obs[0].shape, "to", self.output_path)
        return True

    def collect_trajs(self, num_trajs):
        if num_trajs <= 0:
            return []
        tot_obs = []
        ep_rewards = []
        ep_lengths = []
        tot_actions = []
        sync_envs_normalization(self.training_env, self.eval_env)
        # obs_mean = self.eval_env.obs_rms.mean
        # obs_var = self.eval_env.obs_rms.var
        ret_mean = self.eval_env.ret_rms.mean
        ret_var = self.eval_env.ret_rms.var
        epsilon = self.eval_env.epsilon
        # obs_std = np.sqrt(obs_var + epsilon)
        ret_std = np.sqrt(ret_var + epsilon)
        for i in range(num_trajs):
            # if not isinstance(self.eval_env, VecEnv) or i == 0:
            obs = self.eval_env.reset()
            done, state = False, None
            episode_reward = 0.0
            episode_length = 0
            obs_traj = np.zeros((self.max_horizon, self.eval_env.observation_space.shape[0]), dtype=np.float64)
            ac_traj = np.zeros((self.max_horizon, self.eval_env.action_space.shape[0]), dtype=np.float64)
            for j in range(self.max_horizon):
                action, state = self.model.predict(obs, state=state, deterministic=self.deterministic)
                # denormalize obs
                raw_obs = obs  # (obs * obs_std) + obs_mean
                obs_traj[j] = raw_obs
                ac_traj[j] = action
                obs, reward, done, _info = self.eval_env.step(action)
                raw_reward = reward * ret_std
                episode_reward += raw_reward
                episode_length += 1
                if done:
                    break
            tot_obs.append(obs_traj)
            tot_actions.append(ac_traj)
            ep_rewards.append(episode_reward)
            ep_lengths.append(episode_length)
        logger.record_tabular("perf/ret", np.mean(ep_rewards))
        logger.dump_tabular()
        if self.verbose > 0:
            print("Callback {}/{}:\t average return: {:.01f}\t average length:{:.01f}".format(self.curr_callback_id,
                                                                                              self.final_callback_id,
                                                                                              np.mean(ep_rewards),
                                                                                              np.mean(ep_lengths)))
        return tot_obs, tot_actions, ep_lengths, ep_rewards


def main():
    args = get_param()
    kwargs = vars(args)
    tester.set_hyper_param(**kwargs)
    tester.add_record_param(['info',
                             "seed",
                             "env_id", "policy_timestep"])

    def get_package_path():
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    tester.configure(task_name='data_collect',
                     private_config_path=os.path.join(get_package_path(), 'rla_config.yaml'),
                     log_root=get_package_path())
    tester.log_files_gen()
    tester.print_args()

    load_path = osp.join(DATA_ROOT, "saved_model")
    set_global_seeds(args.seed)
    sess = U.make_session(adaptive=True).__enter__()
    for env_id in ["InvertedPendulum-v2", "Walker2d-v4", "Hopper-v4", "HalfCheetah-v2", "Swimmer-v4", "InvertedDouble-v5"]:
        model_path = osp.join(load_path, "ppo_{}_{}_full.zip".format(args.env_id, args.policy_timestep))
        env_path = osp.join(load_path, "{}_full".format(env_id))
        real_env = make_vec_env(env_id, num_env=args.num_env, dynamic_param=args.dynamic_param, stoc_init_range=0.005)
        real_env = VecNormalize.load(env_path, real_env)
        print("env_id", env_id, "obs_rms", real_env.obs_rms.mean, "norm_obs", real_env.norm_obs, "train", real_env.training)


if __name__ == '__main__':
    main()
