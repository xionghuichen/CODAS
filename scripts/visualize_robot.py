import argparse
import os.path as osp
import tqdm
import mj_envs
import pickle
from codas.utils.functions import *
from private import *
from codas.utils.config import *
from codas.train.rollout import Runner
from stable_baselines.common import set_global_seeds
from codas.wrapper.env_wrapper import GeneratorWrapper
from codas.wrapper.policy_wrapper import WrappedPolicy
from env_config_map import env_config_map
import gym
import cv2
import os

def get_param():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Variational Sequence")
    parser.add_argument('--seed', help='RNG seed', type=int, default=88)
    parser.add_argument('--info',  type=str, default='')
    parser.add_argument('--expert_root', type=str, default=DATA_ROOT)
    parser.add_argument('--task', type=str, default='data_collect')
    # new params
    parser.add_argument('--env_id', help='environment ID', default='relocate-v1')
    parser.add_argument('--num_env', default=1, type=int)

    parser.add_argument('--policy_timestep', type=int, default=1000000)
    parser.add_argument('--collect_trajs', type=int, default=2)
    parser.add_argument('--max_sequence', type=int, default=200)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--dynamic_param', type=float, default=1.0)
    parser.add_argument('--sim_noise', type=float, default=0.5)

    import ast
    parser.add_argument('--deter', type=ast.literal_eval, default=False)
    parser.add_argument('--output_mean', type=ast.literal_eval, default=False)
    parser.add_argument('--clip_action', type=ast.literal_eval, default=False)
    parser.add_argument('--log_tb', type=ast.literal_eval, default=False)
    parser.add_argument('--auto_env_map', type=ast.literal_eval, default=True)
    parser.add_argument('--eval', type=ast.literal_eval, default=False)
    parser.add_argument('--exact_consist', type=ast.literal_eval, default=False)
    parser.add_argument('--log_dir', type=str, default="../log/")
    parser.add_argument('--tot_training_trajs', type=int, default=COLLECT_SIM_TRAJ)
    args = parser.parse_args()
    kwargs = vars(args)
    if kwargs['auto_env_map'] and kwargs['env_id'] in env_config_map:
        kwargs.update(env_config_map[kwargs['env_id']])
    args = argparse.Namespace(**kwargs)
    return args

def main():
    args = get_param()
    load_path = osp.join(DATA_ROOT, "saved_model")

    set_global_seeds(args.seed)
    model_path = osp.join(load_path, '{}.pickle'.format(args.env_id))
    assert osp.exists(model_path)

    original_policy = pickle.load(open(model_path, 'rb'))
    env = gym.make(args.env_id, use_full_state=False)

    img_shape = {ImgShape.WIDTH: args.image_size, ImgShape.HEIGHT: args.image_size, ImgShape.CHANNEL: 3}
    env = GeneratorWrapper(env, normalize=False)
    model = WrappedPolicy(original_policy, env)
    runner = Runner(simulator_env=env, real_world_env=env, max_horizon=args.max_sequence, img_shape=img_shape,
                    clip_acs=False, sim_policy=model, real_policy=model, exact_consist=args.exact_consist)

    obs_acs = {"obs": [], "acs": [], "ep_rets": [], "imgs": [], 'ac_means': [], 'traj_len': []}
    model_name = str(model_path).split('/')[-1].split('.')[0]
    tot_rews = []
    if not os.path.exists(args.env_id):
        os.mkdir(args.env_id)
    for i in tqdm.tqdm(range(args.collect_trajs)):

        video_save_path = os.path.join(args.env_id, str(i) + ".mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_save_path, fourcc, 30, (args.image_size, args.image_size))
        ret_dict = runner.run_traj(deter=False,
                                   mapping_holder=None, render_img=True, run_in_realworld=True)
        total_rew = ret_dict[runner.TOTAL_REW]
        img_traj = ret_dict[runner.IMG_TRAJ]
        tot_rews.append(total_rew)

        img_traj = (img_traj * 255.0).astype(np.uint8)
        for img in img_traj:
            video_writer.write(img)
        print(i,":\t", total_rew)

        video_writer.release()

   


if __name__ =='__main__':
    main()
    #WHAT
