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
    for env_id in ["InvertedPendulum-v2", "Walker2d-v4", "Hopper-v4", "HalfCheetah-v2", "Swimmer-v4", "InvertedDouble-v5"]:
        kwargs['policy_timestep'] = 1000000
        if kwargs['auto_env_map'] and env_id in env_config_map:
            kwargs.update(env_config_map[env_id])
        args = argparse.Namespace(**kwargs)
        model_path = osp.join(load_path, "ppo_{}_{}_full.zip".format(env_id, args.policy_timestep))
        print("env", env_id)
        try:
            model = PPO2.load(model_path)
            env_path = osp.join(load_path, "{}_full".format(env_id))
            real_env = make_vec_env(env_id, num_env=args.num_env, dynamic_param=args.dynamic_param, stoc_init_range=0.005)
            real_env = VecNormalize.load(env_path, real_env)
            real_env.training = False
            real_env.norm_reward = False
            print("env_id", env_id, "obs_rms", real_env.obs_rms.mean, "norm_obs", real_env.norm_obs, "train", real_env.training)
            real_env = GeneratorWrapper(real_env)
            img_shape = {ImgShape.WIDTH: args.image_size, ImgShape.HEIGHT: args.image_size, ImgShape.CHANNEL: 3}
            runner = Runner(simulator_env=None, real_world_env=real_env, max_horizon=args.max_sequence, img_shape=img_shape,
                            clip_acs=False, real_policy=model, sim_policy=None, exact_consist=False)
            avg_rew = []
            for _ in tqdm.tqdm(range(20)):
                ret_dict = runner.run_traj(deter=False, mapping_holder=None, render_img=False, run_in_realworld=True)
                total_rew = ret_dict[runner.TOTAL_REW]
                avg_rew.append(total_rew)
            print("env rew", np.mean(avg_rew), "+-", np.std(avg_rew))
        except Exception as e:
            print("load failed", env_id, e)
if __name__ == '__main__':
    main()
