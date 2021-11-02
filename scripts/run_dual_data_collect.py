import argparse
import os.path as osp
import tqdm
from SRG.mj_env.dyn_envs import get_new_density_env, get_new_gravity_env, get_new_friction_env, get_source_env
from SRG.utils.functions import *
from SRG.utils.private import *
from SRG.utils.structure import *
from SRG.codas.src.env_wrapper import GeneratorWrapper
from SRG.codas.src.env_wrapper import make_vec_env
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import EvalCallback

def get_param():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Variational Sequence")
    parser.add_argument('--seed', help='RNG seed', type=int, default=88)
    parser.add_argument('--info',  type=str, default='')
    parser.add_argument('--expert_root', type=str, default=DATA_ROOT)
    parser.add_argument('--task', type=str, default='data_collect')
    # new params
    parser.add_argument('--env_id', help='environment ID', default='InvertedDouble-v4')
    parser.add_argument('--num_env', default=1, type=int)

    parser.add_argument('--policy_timestep', type=int, default=1000000)
    parser.add_argument('--collect_trajs', type=int, default=COLLECT_TRAJ)
    parser.add_argument('--max_sequence', type=int, default=1002)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--dynamic_param', type=float, default=1.0)

    import ast
    parser.add_argument('--deter', type=ast.literal_eval, default=False)
    parser.add_argument('--output_mean', type=ast.literal_eval, default=False)
    parser.add_argument('--clip_action', type=ast.literal_eval, default=False)
    parser.add_argument('--log_tb', type=ast.literal_eval, default=False)
    parser.add_argument('--eval', type=ast.literal_eval, default=False)
    parser.add_argument('--log_dir', type=str, default="../log/")
    parser.add_argument('--action_noise_std', type=float, default=0.0)
    args = parser.parse_args()
    return args

EPSILON = 0.
class Runner(object):
    TOTAL_REW = 'total_rew'
    TRAJ_LEN = 'traj_len'
    OB_TRAJ = 'ob_traj'
    AC_TRAJ = 'ac_traj'
    R2S_OB_TRAJ = 'r2s_ob_traj'
    IMG_TRAJ = 'img_traj'

    def __init__(self, simulator_env, real_world_env,
                 max_horizon, img_shape, clip_acs, action_noise_std):
        self.simulator_env = simulator_env
        self.real_world_env = real_world_env
        self.max_horizon = max_horizon
        self.img_shape = img_shape
        self.clip_acs = clip_acs
        self.action_noise_std = action_noise_std
        pass

    def map_to_target(self):
        pass

    def run_traj(self, policy, deter, mapping_holder, render_img, run_in_realworld):
        if run_in_realworld:
            select_env = self.real_world_env
        else:
            select_env = self.simulator_env
        de_normalize_ob_traj = np.zeros((self.max_horizon, select_env.observation_space.shape[0]), dtype=np.float32)
        de_normalize_r2s_ob_traj = np.zeros((self.max_horizon, select_env.observation_space.shape[0]), dtype=np.float32)
        ac_traj = np.zeros((self.max_horizon, select_env.action_space.shape[0]), dtype=np.float32)
        img_traj = np.zeros((self.max_horizon, self.img_shape[ImgShape.WIDTH], self.img_shape[ImgShape.HEIGHT],
                             self.img_shape[ImgShape.CHANNEL]), dtype=np.float32)

        if mapping_holder is not None:
            do_mapping = True
            assert isinstance(mapping_holder, MappingHolder)
        else:
            do_mapping = False
        ob = select_env.reset()
        total_rew = 0
        i = 0
        for i in range(self.max_horizon):
            if render_img:
                img = select_env.render(mode='rgb_array', width=self.img_shape[ImgShape.WIDTH],
                                        height=self.img_shape[ImgShape.HEIGHT], camera_name='track')
                img = (img / 255.0).astype(np.float16)
                img_traj[i] = img
            de_normalize_ob_traj[i] = ob
            if do_mapping:
                # tmp set to oracle obs for step == 0
                # if i == 0:
                #     r2s_ob = ob
                # else:
                assert render_img
                r2s_ob = mapping_holder.do_map(index=i, img_traj=img_traj, ac_traj=ac_traj, stoc_infer=not run_in_realworld)
                de_normalize_r2s_ob_traj[i] = r2s_ob
            else:
                r2s_ob = ob
            ac, *_ = policy.step(r2s_ob[None, :], deterministic=deter)
            if self.clip_acs:
                ac = np.clip(ac, env.action_space.low, env.action_space.high)
            ac = ac + np.random.normal(size=ac.shape) * self.action_noise_std
            ac_traj[i] = ac + EPSILON  # to avoid zero output.
            ob, rew, done, _ = select_env.step(ac)
            total_rew += rew
            if done:
                break
        assert np.where(np.all(de_normalize_ob_traj[np.all(ac_traj == 0, axis=-1)] != 0, axis=-1))[0].shape[0] == 0
        assert np.where(np.all(ac_traj[np.all(de_normalize_ob_traj == 0, axis=-1)] != 0, axis=-1))[0].shape[0] == 0
        return {self.TOTAL_REW: total_rew, self.TRAJ_LEN: i,
                self.OB_TRAJ: de_normalize_ob_traj, self.AC_TRAJ: ac_traj,
                self.R2S_OB_TRAJ: de_normalize_r2s_ob_traj, self.IMG_TRAJ: img_traj}

def main():
    args = get_param()
    load_path = osp.join(DATA_ROOT, "saved_model")
    sched_lr = LinearSchedule(args.policy_timestep, 0., 3e-4)

    set_global_seeds(args.seed)
    # tester.print_args()
    model_path = osp.join(load_path, "ppo_{}_{}.zip".format(args.env_id, args.policy_timestep))
    env_path = osp.join(load_path, "{}".format(args.env_id))
    if osp.exists(model_path) and osp.exists(env_path):
        model = PPO2.load(model_path)
        print("loaded pre-trained policy from {}".format(load_path))
        env = make_vec_env(args.env_id, num_env=args.num_env, dynamic_param=args.dynamic_param)
        env = VecNormalize.load(env_path, env)
        env.training = False
        env.norm_reward = False
    else:
        env = make_vec_env(args.env_id, num_env=args.num_env, dynamic_param=args.dynamic_param)
        env = VecNormalize(env)
        if args.log_tb:
            tb_log_dir = args.log_dir
        else:
            tb_log_dir = None
        model = PPO2(policy=MlpPolicy, env=env, verbose=1, n_steps=2048, nminibatches=32, lam=0.95, gamma=0.99,
                     noptepochs=10, ent_coef=0.0, learning_rate=sched_lr.value, cliprange=0.2, tensorboard_log=tb_log_dir)
        if args.eval:
            eval_env = make_vec_env(args.env_id, num_env=args.num_env, dynamic_param=args.dynamic_param)
            eval_env = VecNormalize(eval_env)
            callback = EvalCallback(eval_env, log_path=args.log_dir, eval_freq=2000, deterministic=True)
            model.learn(total_timesteps=args.policy_timestep, callback=callback)
        else:
            model.learn(total_timesteps=args.policy_timestep)
        model.save(model_path)
        env.save(env_path)
        env.training = False
        env.norm_reward = False
    env = GeneratorWrapper(env)
    img_shape = {ImgShape.WIDTH: args.image_size, ImgShape.HEIGHT: args.image_size,
                 ImgShape.CHANNEL: 3}
    runner = Runner(simulator_env=env, real_world_env=env, max_horizon=args.max_sequence, img_shape=img_shape,
                    clip_acs=False, action_noise_std=args.action_noise_std)

    obs_acs = {"obs": [], "acs": [], "ep_rets": [], "imgs": [], 'ac_means': [], 'traj_len': []}
    model_name = str(model_path).split('/')[-1].split('.')[0]
    tot_rews = []
    for _ in tqdm.tqdm(range(args.collect_trajs)):
        # total_rew = 0
        # while(total_rew < 3000):
        #     total_rew, _, ob_traj, ac_traj, img_traj, ac_mean_traj, traj_len = runner()
        ret_dict = runner.run_traj(policy=model, deter=False,
                                   mapping_holder=None, render_img=True, run_in_realworld=True)
        total_rew = ret_dict[runner.TOTAL_REW]
        ob_traj = ret_dict[runner.OB_TRAJ]
        ac_traj = ret_dict[runner.AC_TRAJ]
        img_traj = ret_dict[runner.IMG_TRAJ]
        traj_len = ret_dict[runner.TRAJ_LEN]
        tot_rews.append(total_rew)
        print(total_rew, traj_len)
        img_traj = (img_traj * 255.0).astype(np.uint8)
        obs_acs['obs'].append(ob_traj)
        obs_acs['acs'].append(ac_traj)
        obs_acs['ep_rets'].append(total_rew)
        obs_acs['imgs'].append(img_traj)
        obs_acs['traj_len'].append(traj_len)
    #test
    np.savez(osp.join(args.expert_root,
                      "dual_{:.02f}_".format(args.action_noise_std)+model_name + '_' + str(args.collect_trajs) + '_deter_' + str(args.deter) + '_uint8.npz'),
             obs=obs_acs['obs'], acs=obs_acs['acs'], ep_rets=obs_acs['ep_rets'], imgs=obs_acs['imgs'],
             traj_len=obs_acs['traj_len'])
    print(args.env_id, np.mean(np.array(tot_rews)), np.std(np.array(tot_rews)))


if __name__ =='__main__':
    main()
    #WHAT
