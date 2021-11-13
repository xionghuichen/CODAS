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
    parser.add_argument('--collect_trajs', type=int, default=COLLECT_TRAJ)
    parser.add_argument('--max_sequence', type=int, default=200)
    parser.add_argument('--image_size', type=int, default=1024)
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
    for _ in tqdm.tqdm(range(args.collect_trajs)):
        ret_dict = runner.run_traj(deter=False,
                                   mapping_holder=None, render_img=True, run_in_realworld=True)
        total_rew = ret_dict[runner.TOTAL_REW]
        ob_traj = ret_dict[runner.OB_TRAJ]
        ac_traj = ret_dict[runner.AC_TRAJ]
        img_traj = ret_dict[runner.IMG_TRAJ]
        traj_len = ret_dict[runner.TRAJ_LEN]
        tot_rews.append(total_rew)
        print(total_rew, traj_len)

        img_traj = (img_traj * 255.0).astype(np.uint8)
        # import matplotlib.pyplot as plt
        # for i in range(200):
        #     plt.imshow(img_traj[i])
        #     plt.title(i)
        #     plt.show()
        obs_acs['obs'].append(ob_traj)
        obs_acs['acs'].append(ac_traj)
        obs_acs['ep_rets'].append(total_rew)
        obs_acs['imgs'].append(img_traj)
        obs_acs['traj_len'].append(traj_len)

    # sim dataset gen

    tot_obs = []
    ep_rewards = []
    ep_lengths = []
    tot_actions = []
    # obs_mean = self.eval_env.obs_rms.mean
    # obs_var = self.eval_env.obs_rms.var
    for i in range(args.tot_training_trajs):

        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        obs_traj = np.zeros((args.max_sequence, env.state_space.shape[0]), dtype=np.float64)
        ac_traj = np.zeros((args.max_sequence, env.action_space.shape[0]), dtype=np.float64)
        epsilon = args.sim_noise * i / args.tot_training_trajs
        for j in range(args.max_sequence):
            # obs = env.full_state_to_obs(state[None])
            action = model.step(obs,  deterministic=False)[0]
            action = action + np.random.uniform(-1, 1, size=action.shape) * epsilon
            # denormalize obs
            raw_obs = obs # env.full_state_to_obs(obs[None])  # (obs * obs_std) + obs_mean
            obs_traj[j] = raw_obs
            ac_traj[j] = action
            if args.exact_consist:
                obs, reward, done, _info = env.set_ob_and_step(obs, action.squeeze())
            else:
                obs, reward, done, _info = env.step(action.squeeze())
            raw_reward = reward
            episode_reward += raw_reward
            episode_length += 1
            if done:
                break
        tot_obs.append(obs_traj)
        tot_actions.append(ac_traj)
        ep_rewards.append(episode_reward)
        ep_lengths.append(episode_length)
        print("sim data collect: ", i, "noise: ", epsilon, "rew", episode_reward)

    np.savez(osp.join(args.expert_root,
                      model_name + '_' + str(args.collect_trajs) + '_deter_' + str(args.deter) +
                      '_exact_' + str(args.exact_consist) + "_img_" + str(args.image_size) + '_uint8.npz'),
             obs=obs_acs['obs'], acs=obs_acs['acs'], ep_rets=obs_acs['ep_rets'], imgs=obs_acs['imgs'],
             traj_len=obs_acs['traj_len'], train_obs=tot_obs, train_acs=tot_actions,
             train_ep_rets=ep_rewards, train_traj_len=ep_lengths)

    print(args.env_id, np.mean(np.array(tot_rews)), np.std(np.array(tot_rews)))


if __name__ =='__main__':
    main()
    #WHAT
