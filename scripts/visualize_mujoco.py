import argparse
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
import os.path as osp
import tqdm

from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common import set_global_seeds

from RLA.easy_log.tester import tester
from RLA.easy_log import logger

from codas.utils.functions import *
from codas.utils.config import *
from codas.wrapper.env_wrapper import GeneratorWrapper, make_vec_env
from codas.train.rollout import Runner
from codas.rl.ppo2 import PPO2
from private import *
from env_config_map import env_config_map
import cv2

def get_param():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Variational Sequence")
    parser.add_argument('--seed', help='RNG seed', type=int, default=88)
    parser.add_argument('--info',  type=str, default='')
    parser.add_argument('--expert_root', type=str, default="../data")
    parser.add_argument('--task', type=str, default='data_collect')
    # new params
    parser.add_argument('--env_id', help='environment ID', default='InvertedDouble-v4')
    parser.add_argument('--num_env', default=1, type=int)

    parser.add_argument('--policy_timestep', type=int, default=1000000)
    parser.add_argument('--collect_trajs', type=int, default=2)
    parser.add_argument('--max_sequence', type=int, default=1002)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--dynamic_param', type=float, default=1.0)

    import ast
    parser.add_argument('--deter', type=ast.literal_eval, default=False)
    parser.add_argument('--auto_env_map', type=ast.literal_eval, default=True)
    parser.add_argument('--output_mean', type=ast.literal_eval, default=False)
    parser.add_argument('--log_tb', type=ast.literal_eval, default=False)
    parser.add_argument('--eval', type=ast.literal_eval, default=False)
    parser.add_argument('--log_dir', type=str, default="../log/")
    parser.add_argument('--collect_train_data', type=ast.literal_eval, default=True)
    parser.add_argument('--action_noise_std', type=float, default=0.0)
    args = parser.parse_args()
    kwargs = vars(args)
    if kwargs['auto_env_map'] and kwargs['env_id'] in env_config_map:
        kwargs.update(env_config_map[kwargs['env_id']])
    args = argparse.Namespace(**kwargs)
    return args



def main():
    args = get_param()
    if args.action_noise_std > 0:
        args.collect_train_data = False
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

    load_path = osp.join(args.expert_root, "saved_model")
    set_global_seeds(args.seed)

    model_path = osp.join(load_path, "ppo_{}_{}_full.zip".format(args.env_id, args.policy_timestep))
    env_path = osp.join(load_path, "{}_full".format(args.env_id))
    env = make_vec_env(args.env_id, num_env=args.num_env, dynamic_param=args.dynamic_param)
    env = VecNormalize.load(env_path, env)
    env._shape = env
    model = PPO2.load(model_path, env=env)
    print("loaded pre-trained policy from {}".format(load_path))
   
    env.training = False
    env.norm_reward = False
    
    env = GeneratorWrapper(env)

    img_shape = {ImgShape.WIDTH: args.image_size, ImgShape.HEIGHT: args.image_size,
                 ImgShape.CHANNEL: 3}
    runner = Runner(simulator_env=None, real_world_env=env, max_horizon=args.max_sequence, img_shape=img_shape,
                    clip_acs=False, real_policy=model, sim_policy=None, exact_consist=False,
                    action_noise_std=args.action_noise_std)

    # obs_mean = env.mean
    # obs_std = env.std
    # #normalize training data
    # if args.collect_train_data:
    #     train_obs = [(traj-obs_mean)/obs_std for traj in train_obs]
    # print("obs mean:", obs_mean, "\nobs_std:", obs_std)

    if not os.path.exists("videos", args.env_id):
        os.makedirs(os.path.exists("videos", args.env_id))
    tot_rews = []
    for i in tqdm.tqdm(range(args.collect_trajs)):
    
        video_save_path = os.path.join("videos", args.env_id, str(i) + ".mp4")
        
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

if __name__ == '__main__':
    main()
