import argparse
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
import os.path as osp
import pickle
from collections import deque
import gym
import cv2
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from tqdm import tqdm
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

from codas.wrapper.env_wrapper import make_vec_env, GeneratorWrapper, is_dapg_env
from codas.wrapper.policy_wrapper import WrappedPolicy
from env_config_map import env_config_map
# from SRG.codas.src.env_config_map import env_config_map
from codas.train.discriminator import StateDistributionDiscriminator, ImgDiscriminator, TrajDiscriminator
from codas.train.img_codec import Encoder, Decoder, LargeEncoder, LargeDecoder
from codas.train.mapping_func import Embedding, Sim2Real, Real2Sim
from codas.train.policy import Policy
from codas.train.variational_seq import VarSeq
from codas.train.rollout import MappingHolder, Runner
from codas.data_process.mujoco_dset import Mujoco_Dset
from codas.utils import tf_util as U
from codas.utils.config import *
from codas.utils.functions import *
from private import *
from codas.utils.replay_buffer import TrajectoryBuffer
from RLA.easy_log.tester import tester
from RLA.easy_log import logger
from RLA.easy_log.tools import time_used_wrap, time_record, time_record_end
import pdb

def get_param():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Variational Sequence")
    import ast
    # task info
    parser.add_argument('--seed', help='RNG seed', type=int, default=88)
    parser.add_argument('--task', type=str, default='')
    parser.add_argument('--env_id', help='environment ID', default='InvertedDouble-v5')
    parser.add_argument('--auto_env_map', type=ast.literal_eval, default=True)
    parser.add_argument('--pretrain_path', type=str, default=osp.join(DATA_ROOT, 'saved_model/transition_weights.npy'))
    parser.add_argument('--pretrain_mean_std', type=str, default=osp.join(DATA_ROOT, 'saved_model/state_mean_std.npy'))
    parser.add_argument('--policy_timestep', type=int, default=1000000)
    parser.add_argument('--collect_trajs', type=int, default=COLLECT_TRAJ)
    parser.add_argument('--alg_type', type=str, default=AlgType.CODAS)
    parser.add_argument('--cycle_loss', type=ast.literal_eval, default=False)
    parser.add_argument('--info', type=str, default='')
    parser.add_argument('--traj_limitation', type=int, default=-1)
    parser.add_argument('--expert_perf_eval_times', type=int, default=100)
    parser.add_argument('--action_noise_level', type=float, default=0.0)
    parser.add_argument('--dynamic_param', type=float, default=1.0)
    parser.add_argument('--load_date', type=str, default='')
    parser.add_argument('--load_task', type=str, default='')
    parser.add_argument('--max_sequence', type=int, default=500)
    parser.add_argument("--max_tf_util", help="per process gpu memory fraction fot tf", type=float, default=1.0)
    # reduce the rollout step of traj-discriminator will make the training process more stable
    # if the size of dataset is not enough.
    parser.add_argument('--rollout_step', type=int, default=100)
    parser.add_argument('--minibatch_size', type=int, default=-1)
    parser.add_argument('--dis_test', type=ast.literal_eval, default=False)
    parser.add_argument('--deter_policy', type=ast.literal_eval, default=False)
    parser.add_argument('--label_image_test', type=ast.literal_eval, default=False)
    parser.add_argument('--dynamic_test', type=ast.literal_eval, default=False)
    parser.add_argument('--use_dataset_mean_std', type=ast.literal_eval, default=False)
    parser.add_argument('--exact_consist', type=ast.literal_eval, default=False)
    # transformation setting
    parser.add_argument('--ob_transformation', type=str, default=Transformation.IMG)
    # params added for image_input
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--image_channel', type=int, default=3)
    parser.add_argument('--output_image', type=ast.literal_eval, default=True)
    # learn expert policy
    # network params
    parser.add_argument('--act_fn', type=str, default=ActivateFn.LEAKLYRELU)
    parser.add_argument('--dyn_act_fn', type=str, default=ActivateFn.TANH)
    parser.add_argument('--layer_norm', type=ast.literal_eval, default=True)
    parser.add_argument('--safe_log', type=ast.literal_eval, default=False)
    parser.add_argument('--mapping_direction', type=str, default=MappingDirecition.RSR)
    parser.add_argument('--stack_imgs', type=int, default=1)
    # 1. discriminator params
    parser.add_argument('--dis_struc', type=str, default=DiscriminatorStructure.OB_AC_CONCATE)
    parser.add_argument('--rnn_cell', type=str, default=RNNCell.GRU)
    parser.add_argument('--disc_hid_dims', type=ast.literal_eval, default=[256, 256, 256])
    parser.add_argument('--disc_img_hid_dims', type=ast.literal_eval, default=[256])
    parser.add_argument('--dyn_hid_dims', type=ast.literal_eval, default=[512, 512, 512])
    parser.add_argument('--disc_emb_hid_dim', type=int, default=256)
    parser.add_argument('--num_env', type=int, default=1)
    # 2. embedding params
    parser.add_argument('--emb_hid_dims', type=ast.literal_eval, default=[256, 256, 256, 256])
    parser.add_argument('--emb_output_size', type=int, default=256)
    parser.add_argument('--mlp', type=ast.literal_eval, default=False)
    parser.add_argument('--clip_acs', type=ast.literal_eval, default=True)
    # 3. encode param.
    parser.add_argument('--gan_loss', type=str, default=GanLoss.MINIMAX)
    parser.add_argument('--r2s_rnn_hid_dims', type=ast.literal_eval, default=[128, 128])
    parser.add_argument('--r2s_output_hid_dims', nargs='+', type=int, default=[])
    parser.add_argument('--adjust_allowed', type=float, default=1.0)
    parser.add_argument('--emb_dynamic', type=ast.literal_eval, default=True)
    parser.add_argument('--policy_infer_transition', type=ast.literal_eval, default=True)
    # 3. reconstruction params
    parser.add_argument('--s2r_hid_dims', type=ast.literal_eval, default=[256, 256, 256, 256])
    parser.add_argument('--s2r_rnn_hid_dims', type=ast.literal_eval, default=[])
    parser.add_argument('--s2r_emb_dim', type=int, default=256)
    parser.add_argument('--reconstruct_clip', type=float, default=-1)
    parser.add_argument('--res_struc', type=str, default=ResnetStructure.EMBEDDING_RAS)
    parser.add_argument('--resc_act_fn', type=str, default=ActivateFn.ID)
    parser.add_argument('--real_ob_input', type=ast.literal_eval, default=False)
    parser.add_argument('--hard_training', type=ast.literal_eval, default=False)
    parser.add_argument('--retrain_dynamics', type=ast.literal_eval, default=False)
    parser.add_argument('--filter_traj', type=ast.literal_eval, default=False)
    # learning params
    parser.add_argument('--lr_gen', type=float, default=0.0001)
    parser.add_argument('--lr_dyn', type=float, default=0.0001)
    parser.add_argument('--dyn_lr_pretrain', type=float, default=0.0001)
    parser.add_argument('--dyn_l2_loss', type=float, default=0.0000002)
    parser.add_argument('--mapping_l2_loss', type=float, default=0.0)
    parser.add_argument('--dis_l2_loss', type=float, default=0.0)
    parser.add_argument('--lr_dis', type=float, default=0.00005)
    parser.add_argument('--lr_rescale', type=float, default=1.0)
    parser.add_argument('--dyn_batch_size', type=int, default=1024)
    parser.add_argument('--mapping_train_epoch', type=int, default=5)
    parser.add_argument('--dis_train_epoch', type=int, default=1)
    parser.add_argument('--trajectory_batch', type=int, default=10)
    parser.add_argument('--decay_ratio', type=float, default=0.5)
    parser.add_argument('--total_timesteps', type=int, default=40000)
    parser.add_argument('--lambda_a', type=float, default=2)
    parser.add_argument('--lambda_b', type=float, default=0.1)
    parser.add_argument('--norm_std_bound', type=float, default=0.05)
    parser.add_argument('--stoc_init_range', type=float, default=0.005)
    parser.add_argument('--grad_clip_norm', type=float, default=10)
    parser.add_argument('--random_set_to_zero', type=ast.literal_eval, default=False)
    parser.add_argument('--data_normalize', type=ast.literal_eval, default=True)
    parser.add_argument('--minmax_normalize', type=ast.literal_eval, default=False)
    parser.add_argument('--npmap_replace', type=ast.literal_eval, default=False) # namap will reduce the occupancy of mermory.
    parser.add_argument('--merge_d_train', type=ast.literal_eval, default=True)
    parser.add_argument('--traj_dis', type=ast.literal_eval, default=False)
    parser.add_argument('--clip_policy_bound', type=ast.literal_eval, default=True)
    # TODO： the correctness of related code should be check.
    parser.add_argument('--init_first_state', type=ast.literal_eval, default=False)
    # trajectory-buffer hyperparameter
    parser.add_argument('--use_env_sample', type=ast.literal_eval, default=True)
    parser.add_argument('--do_save_checkpoint', type=ast.literal_eval, default=True)
    parser.add_argument('--pool_size', type=int, default=6000)
    parser.add_argument('--data_reused_times', type=int, default=10)
    # ablation study
    parser.add_argument('--data_used_fraction', type=float, default=1)
    parser.add_argument("--use_noise_env", type=ast.literal_eval, default=False)
    parser.add_argument("--dual_policy_noise_std", help="use obs collected by noise action", type=float, default=0.0)

    args = parser.parse_args()
    kwargs = vars(args)
        # kwargs['exact_consist'] = True
    if kwargs['auto_env_map'] and kwargs['env_id'] in env_config_map:
        kwargs.update(env_config_map[kwargs['env_id']])
    assert kwargs['max_sequence'] % kwargs['rollout_step'] == 0
    # seq_length = int(np.ceil(args.max_sequence / args.rollout_step) * args.rollout_step)
    if kwargs['alg_type'] == AlgType.VAN_GAN:
        kwargs['traj_dis'] = False
        kwargs['emb_dynamic'] = False
        kwargs['lambda_b'] = 0
        kwargs['mlp'] = True
        kwargs['cycle_loss'] = False
    elif kwargs['alg_type'] == AlgType.CYCLE_GAN:
        kwargs['traj_dis'] = False
        kwargs['emb_dynamic'] = False
        kwargs['lambda_b'] = 0
        kwargs['mlp'] = True
        kwargs['cycle_loss'] = True
    elif kwargs["alg_type"] == AlgType.VAN_GAN_STACK:
        kwargs['traj_dis'] = False
        kwargs['emb_dynamic'] = False
        kwargs['lambda_b'] = 0
        kwargs['mlp'] = True
        kwargs["stack_imgs"] = 4
    elif kwargs['alg_type'] == AlgType.VAN_GAN_RNN:
        kwargs['traj_dis'] = False
        kwargs['emb_dynamic'] = False
        kwargs['lambda_b'] = 0
        kwargs['mlp'] = False
    elif kwargs['alg_type'] == AlgType.CODAS:
        kwargs['traj_dis'] = True
    elif kwargs['alg_type'] == AlgType.NO_DYN:
        kwargs['traj_dis'] = True
        kwargs['emb_dynamic'] = False
    elif kwargs['alg_type'] == AlgType.NO_DYN_NO_TRAJ_DIS:
        kwargs['traj_dis'] = False
        kwargs['emb_dynamic'] = False
    elif kwargs['alg_type'] == AlgType.MLP:
        kwargs['traj_dis'] = False
        kwargs['emb_dynamic'] = False
        kwargs['mlp'] = True
    elif kwargs['alg_type'] == AlgType.MLP_TRAJ_DIS:
        kwargs['traj_dis'] = True
        kwargs['emb_dynamic'] = False
        kwargs['mlp'] = True
        # kwargs["r2s_output_hid_dims"] = kwargs["dyn_hid_dims"]
    elif kwargs['alg_type'] == AlgType.NO_TRAJ_DIS:
        kwargs['traj_dis'] = False
    else:
        raise NotImplementedError

    kwargs["lr_dis"] *= kwargs["lr_rescale"]
    kwargs["lr_gen"] *= kwargs["lr_rescale"]
    args = argparse.Namespace(**kwargs)
    # kwargs = vars(args)
    tester.set_hyper_param(**kwargs)
    # add parameters to track:
    tester.add_record_param(['info',
                             "seed",
                             "alg_type",
                             "emb_dynamic",
                             "traj_dis",
                             'dual_policy_noise_std',
                             'dynamic_param',
                             'stoc_init_range',
                             'traj_limitation',
                             ])
    return args


def main():

    args = get_param()

    def task_name_gen():
        return '-'.join([args.ob_transformation, args.task, args.env_id]) + 'v2'

    def get_package_path():
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    tester.configure(task_name=task_name_gen(),
                     private_config_path=os.path.join(get_package_path(), 'rla_config.yaml'),
                     log_root=get_package_path())
    tester.log_files_gen()
    tester.print_args()

    assert args.action_noise_level == 0.0, "Non-zero action noise level is not supported yet"

    act_fn = ActivateFn.obj_convect(args.act_fn)
    resc_act_fn = ActivateFn.obj_convect(args.resc_act_fn)
    dyn_act_fn = ActivateFn.obj_convect(args.dyn_act_fn)
    rnn_cell = RNNCell.obj_convect(args.rnn_cell)
    sess = U.make_session(adaptive=True, percent=args.max_tf_util).__enter__()
    set_global_seeds(args.seed)
    env = None

    os.makedirs(os.path.join(tester.results_dir, "imgs/"), exist_ok=True)
    # step 0: path join
    load_path = osp.join(DATA_ROOT, "saved_model")
    if args.norm_std_bound == 1:
        norm_std_str = ''
    else:
        norm_std_str = 'std-{}'.format(args.norm_std_bound)
    if args.clip_policy_bound:
        cpb_str = ''
    else:
        cpb_str = 'ncpb'
    if is_dapg_env(args.env_id):
        OBS_BOUND = 150
    else:
        OBS_BOUND = 100
    img_shape = {ImgShape.WIDTH: args.image_size, ImgShape.HEIGHT: args.image_size, ImgShape.CHANNEL: 3}
    # step 1: environment construction
    is_robot_env = is_dapg_env(args.env_id)
    if is_robot_env:
        import mj_envs
        model_path = osp.join(load_path, '{}.pickle'.format(args.env_id))
        assert osp.exists(model_path)
        assert args.dual_policy_noise_std == 0
        real_expert_path = sim_expert_path = expert_path = osp.join(DATA_ROOT, '{}_{}_deter_False_exact_{}_img_{}_uint8.npz'.
                               format(args.env_id,  args.collect_trajs, args.exact_consist,  args.image_size))

        original_policy = pickle.load(open(model_path, 'rb'))
        env = gym.make(args.env_id, use_full_state=False)
        real_world_env = gym.make(args.env_id, use_full_state=False)
        env = GeneratorWrapper(env)
        real_world_env = GeneratorWrapper(real_world_env)
        model = WrappedPolicy(original_policy, env)
        real_model = WrappedPolicy(original_policy, real_world_env)
        dynamics_model_path = osp.join(load_path, '{}_{}_network_weights-full-{}{}{}.npy'.
                                       format(args.env_id, args.collect_trajs, args.dynamic_param,
                                              norm_std_str, cpb_str))
        dynamics_model_param_path = osp.join(load_path, '{}_{}_network_weights_param-full-{}{}{}.pkl'.
                                             format(args.env_id, args.collect_trajs, args.dynamic_param,
                                                    norm_std_str, cpb_str))

        runner = Runner(simulator_env=env, real_world_env=real_world_env, sim_policy=model, real_policy=real_model,
                        max_horizon=args.max_sequence, img_shape=img_shape, clip_acs=args.clip_acs,
                        exact_consist=args.exact_consist)

    else:

        model_path = osp.join(load_path, "ppo_{}_{}_full.zip".format(args.env_id, args.policy_timestep))
        env_path = osp.join(load_path, "{}_full".format(args.env_id))
        if np.abs(args.dual_policy_noise_std - 0.0) > 1e-5:
            real_expert_path = osp.join(DATA_ROOT, 'dual_{:.02f}_ppo_{}_{}_{}_deter_False_uint8.npz'.
                                   format(args.dual_policy_noise_std, args.env_id, args.policy_timestep,
                                          args.collect_trajs))
            sim_expert_path = osp.join(DATA_ROOT, 'ppo_{}_{}_full_{}_deter_False_uint8_full.npz'.
                                   format(args.env_id, args.policy_timestep, args.collect_trajs))
        else:
            real_expert_path = sim_expert_path = osp.join(DATA_ROOT, 'ppo_{}_{}_full_{}_deter_False_uint8_full.npz'.
                                   format(args.env_id, args.policy_timestep, args.collect_trajs))
        env = make_vec_env(args.env_id, num_env=args.num_env, dynamic_param=args.dynamic_param, stoc_init_range=args.stoc_init_range)
        real_world_env = make_vec_env(args.env_id, num_env=args.num_env, dynamic_param=1.0, stoc_init_range=0.005)
        env = VecNormalize.load(env_path, env)
        env.training = False
        env.norm_reward = False
        real_world_env = VecNormalize.load(env_path, real_world_env)
        real_world_env.training = False
        real_world_env.norm_reward = False
        model = PPO2.load(model_path)
        logger.info("loaded pre-trained policy from {}".format(model_path))
        logger.info("loaded normalized env from {}".format(env_path))
        real_world_env = GeneratorWrapper(real_world_env, use_image_noise=args.use_noise_env)
        env = GeneratorWrapper(env)

        dynamics_model_path = osp.join(DATA_ROOT, f'ppo_{args.env_id}_{args.policy_timestep}_{COLLECT_TRAJ}_network_weights-full'
                                                  f'-{args.dynamic_param}-ca-{args.clip_acs}-'
                                                  f'dn-{args.data_normalize}{args.minmax_normalize}{norm_std_str}{cpb_str}')
        dynamics_model_param_path = osp.join(DATA_ROOT, f'ppo_{args.env_id}_{args.policy_timestep}_{COLLECT_TRAJ}_'
                                                        f'network_weights_param-full-{ args.dynamic_param}-ca-{args.clip_acs}-'
                                                        f'dn-{args.data_normalize}{norm_std_str}{cpb_str}')

        if args.minmax_normalize:
            dynamics_model_path += '-mn'
            dynamics_model_param_path += '-mm'
        dynamics_model_path += '.npy'
        dynamics_model_param_path += '.pkl'
        runner = Runner(simulator_env=env, real_world_env=real_world_env, sim_policy=model, real_policy=model,
                        max_horizon=args.max_sequence, img_shape=img_shape, clip_acs=args.clip_acs, exact_consist=args.exact_consist)

    env.reset()
    real_world_env.reset()

    # step 2: dataset construction
    expert_dataset = Mujoco_Dset(sim_data=False, expert_path=real_expert_path, traj_limitation=args.traj_limitation,
                                 use_trajectory=True, max_sequence=args.max_sequence, env=env,
                                 data_used_fraction=args.data_used_fraction, clip_action=args.clip_acs,
                                 filter_traj=args.filter_traj, npmap_replace=args.npmap_replace)
    sim_training_dataset = Mujoco_Dset(sim_data=True, expert_path=sim_expert_path, traj_limitation=-1,
                                 use_trajectory=True, max_sequence=args.max_sequence, env=env,
                                 data_used_fraction=1.0, clip_action=args.clip_acs, filter_traj=False,
                                       npmap_replace=args.npmap_replace)
    expert_dataset.obs_std[expert_dataset.obs_std == 0] = 1
    sim_training_dataset.obs_std[sim_training_dataset.obs_std < args.norm_std_bound] = 1

    state_mean_std = [sim_training_dataset.obs_mean, sim_training_dataset.obs_std]
    if not args.data_normalize:
        state_mean_std[0] = np.zeros(state_mean_std[0].shape)
        state_mean_std[1] = np.ones(state_mean_std[1].shape)

    if args.minmax_normalize:
        state_mean_std[0] = sim_training_dataset.obs_min
        state_mean_std[1] = sim_training_dataset.obs_max - sim_training_dataset.obs_min
        state_mean_std[0][state_mean_std[1] == 0] = 0
        state_mean_std[1][state_mean_std[1] == 0] = 1

    logger.info("state_mean {} \n state std {} \n".format(state_mean_std[0], state_mean_std[1]))


    def data_normalization(input):
        norm_input = (input - state_mean_std[0]) / state_mean_std[1]
        if len(input.shape) == 3:
            norm_input[np.where(np.all(input == 0, axis=2))] = 0
        return norm_input

    def data_denormalization(input):
        norm_input = (input * state_mean_std[1]) + state_mean_std[0]
        if len(input.shape) == 3:
            norm_input[np.where(np.all(input == 0, axis=2))] = 0
        return norm_input


    env.reset()
    real_world_env.reset()
    # step 3: graph initialization
    if args.gan_loss == GanLoss.MINIMAX:
        if args.traj_dis:
            discriminator = TrajDiscriminator(hid_dims=args.disc_hid_dims, emb_hid_dim=args.disc_emb_hid_dim,
                                              output_size=1, input_type=args.dis_struc, layer_norm=False,
                                              rnn_cell=rnn_cell, rnn_hidden_dims=[128],
                                              scope='discriminator')

        else:
            discriminator = StateDistributionDiscriminator(hid_dims=args.disc_hid_dims, emb_hid_dim=args.disc_emb_hid_dim,
                                                           output_size=1, input_type=args.dis_struc, layer_norm=args.layer_norm,
                                                           scope='discriminator')
    elif args.gan_loss == GanLoss.WGAN:
        discriminator = StateDistributionDiscriminator(hid_dims=args.disc_hid_dims, emb_hid_dim=args.disc_emb_hid_dim,
                                                       output_size=1, input_type=args.dis_struc, layer_norm=False,
                                                       act_fn=tf.nn.leaky_relu,
                                                       scope='discriminator')
    else:
        raise NotImplementedError

    img_discriminator = ImgDiscriminator(hid_dims=args.disc_img_hid_dims, emb_hid_dim=args.disc_emb_hid_dim,
                                                   output_size=1,
                                                   input_type=args.dis_struc, layer_norm=args.layer_norm,
                                                   scope='img_discriminator')
    embedding = Embedding(output_size=args.emb_output_size, hidden_dims=args.emb_hid_dims,
                          scope='embedding', act_fn=act_fn, layer_norm=args.layer_norm)
    logger.info("min/max value of raw obs", sim_training_dataset.obs_min.min(),  sim_training_dataset.obs_max.max())
    if args.clip_policy_bound:
        norm_min = data_normalization(np.clip(sim_training_dataset.obs_min, -1 * OBS_BOUND, OBS_BOUND))
        norm_max = data_normalization(np.clip(sim_training_dataset.obs_max, -1 * OBS_BOUND, OBS_BOUND))
    else:
        norm_min = data_normalization(sim_training_dataset.obs_min)
        norm_max = data_normalization(sim_training_dataset.obs_max)
    logger.info("norm min", norm_min)
    logger.info("norm max", norm_max)
    obs_min, obs_max = norm_min.min(), norm_max.max()
    logger.info("obs range [{}, {}]. adjust allowed: {}".format(obs_min, obs_max, args.adjust_allowed))

    norm_range = norm_max - norm_min
    logger.info("norm obs scale ", norm_range)
    epsilon_expanded = 0.05
    update_dynamics_range_min = norm_min - epsilon_expanded * norm_range
    update_dynamics_range_max = norm_max + epsilon_expanded * norm_range
    update_dynamics_range_min_trans_learn = norm_min - (epsilon_expanded - 1e-3) * norm_range
    update_dynamics_range_max_trans_learn = norm_max + (epsilon_expanded - 1e-3) * norm_range
    logger.info("update_dynamics_range_min", update_dynamics_range_min)
    logger.info("update_dynamics_range_max", update_dynamics_range_max)
    if args.emb_dynamic:
        from codas.train.transition import Transition, TransitionDecoder, TransitionLearner
        transition = Transition(transition_hidden_dims=args.dyn_hid_dims, transition_trainable=False,
                                ob_shape=env.state_space.shape[0],
                                act_fn=dyn_act_fn, obs_min=update_dynamics_range_min,
                                obs_max=update_dynamics_range_max)
        transition_learned = Transition(transition_hidden_dims=args.dyn_hid_dims, transition_trainable=True,
                                        ob_shape=env.state_space.shape[0],
                                        scope='transition_learn',
                                        act_fn=dyn_act_fn,  obs_min=update_dynamics_range_min,
                                        obs_max=update_dynamics_range_max)
        transition_learner = TransitionLearner(transition=transition_learned, transition_target=transition,
                                               ob_shape=env.state_space.shape[0],
                                               ac_shape=env.action_space.shape[0],
                                               lr=args.lr_dyn, sess=sess, batch_size=args.dyn_batch_size,
                                               l2_loss=args.dyn_l2_loss)

        transition_decoder = TransitionDecoder(ob_shape=env.state_space.shape[0],
                                               hidden_dims=args.r2s_output_hid_dims,
                                               obs_min=update_dynamics_range_min,
                                               obs_max=update_dynamics_range_max,)
    else:
        transition = None
        transition_learner = None
        transition_decoder = None
    if args.mlp:
        from codas.train.mapping_func import MlpEncoder
        mlp = MlpEncoder(hidden_dims=args.dyn_hid_dims, act_fn=tf.nn.tanh, scope='mlp')
    else:
        mlp = None
    real2sim = Real2Sim(rnn_hidden_dims=args.r2s_rnn_hid_dims, target_mapping=False,
                        rnn_cell=rnn_cell, scope='real2sim_mapping', seq_length=args.max_sequence,
                        output_hidden_dims=args.r2s_output_hid_dims, act_fn=act_fn, layer_norm=args.layer_norm,
                        emb_dynamic=args.emb_dynamic, transition=transition, transition_decoder=transition_decoder,
                        ob_shape=env.state_space.shape[0], action_shape=env.action_space.shape[0],
                        mlp_layer=mlp)

    sim2real = Sim2Real(hidden_dims=args.s2r_hid_dims, emb_dim=args.s2r_emb_dim, scope='sim2real_mapping',
                        rnn_hidden_dims=args.s2r_rnn_hid_dims, rnn_cell=rnn_cell,
                        ob_shape=env.state_space.shape[0], ac_shape=env.action_space.shape[0],
                        layer_norm=args.layer_norm, act_fn=resc_act_fn, real_ob_input=args.real_ob_input)

    policy = Policy(False, env.action_space.shape[0], sess=sess)
    if args.image_size == 64:
        encoder = Encoder(stack_imgs=1)
        decoder = Decoder()
    else:
        assert args.image_size == 128
        encoder = LargeEncoder(stack_imgs=1)
        decoder = LargeDecoder()

    var_seq = VarSeq(sequence_length=args.max_sequence, scope='var_seq', img_shape=img_shape,
                     embedding=embedding, real2sim_mapping=real2sim, sim2real_mapping=sim2real,
                     discriminator=discriminator, obs_discriminator=img_discriminator,
                     encoder=encoder, decoder=decoder, policy=policy,
                     batch_size=args.trajectory_batch,
                     lambda_a=args.lambda_a, lambda_b=args.lambda_b,
                     ac_shape=env.action_space.shape[0], ob_shape=env.state_space.shape[0],
                     lr_dis=args.lr_dis, lr_gen=args.lr_gen,
                     total_timesteps=args.total_timesteps, decay_ratio=args.decay_ratio,
                     grad_clip_norm=args.grad_clip_norm, sess=sess,
                     dis_test=args.dis_test, label_image_test=args.label_image_test,
                     reconstruct_clip=args.reconstruct_clip,
                     emb_dynamic=args.emb_dynamic, rollout_step=args.rollout_step,
                     cycle_loss=args.cycle_loss, minibatch_size=args.minibatch_size, merge_d_train=args.merge_d_train,
                     stack_imgs=args.stack_imgs, random_set_to_zero=args.random_set_to_zero,
                     init_first_state=args.init_first_state, l2_coeff=args.mapping_l2_loss,
                     dis_l2_coeff=args.dis_l2_loss)
    var_seq.model_setup()
    if args.emb_dynamic:
        transition_learner.model_setup()

    # step 4: initialize variables and functions.
    U.initialize()
    tester.new_saver(var_prefix='', max_to_keep=1)
    pool_size = int(args.pool_size if args.pool_size > 0 else args.trajectory_batch * args.data_reused_times * 1000)
    trajecory_buffer = TrajectoryBuffer(pool_size, has_img=False)
    mapping_train_epoch = args.mapping_train_epoch
    test_episode = 20
    total_timesteps = args.total_timesteps
    adjust_allowed = args.adjust_allowed
    dis_train_epoch = args.dis_train_epoch
    default_zero_state = var_seq.real2sim_mapping.zero_state_run(1, sess=sess)

    # object -> obs stype:
    # codas: norm obs;
    # - dynamics. -> dynamics training;
    #   - update dynamics;
    #   - pretrain;
    #       - sample_next_batch_data
    # - r2s, s2r. -> D training, G training;
    #   - D training, G training;
    #       - sample_next_batch_data
    # - usage:
    #   - runnning: mapping function
    #   - evaluation
    # envirnoment: raw obs;
    # - env.set_ob_and_step;
    #   - one_step_transition;
    # - env.step;
    #   - runner.run_traj
    #
    # policy: raw obs;
    #   - model.step;
    #       - policy-infer transition generation;
    #       - runner.run_traj;

    # normalization 关系:
    # L1: GeneratorWrapper： 处理训练PPO自带的normalization，所有和GeneratorWrapper 交互的接口都是经过normalization 的状态，
    # 也就是可以直接和PPO policy 交互的state；
    # L2: 为了方便RNN和dynamics model 的训练，我们对 RNN 和dynamics model的输入输出进行归一化处理。
    # 由于PPO normalization的结果可能对于整个环境是由偏的（因为其局限于所最后收敛策略的mean state），
    # VarSeq和Dynamics Model使用额外的归一化，用dataset 的mean std进行归一化，见data_normalization和data_denormalization。
    # 因此，VarSeq和Dynamics Model输入的训练数据是归一化后的，输出的预测也是归一化后的。
    # L3: VarSeq和Dynamics Model 和外界的交互主要由以下几个部分：
    #    1. env.step_ob_and_step
    #    2. map_holder.do_map
    # 对于env.step_ob_and_step，如果输入的obs 是来源于VarSeq和Dynamics Model的，需要进行data_denormalization 在输入给step_ob_and_step。
    # 对于map_holder.do_map, 输入也不需要oracle的状态，所以不需要考虑归一化问题；
    # 我们内部已经处理好了反归一化逻辑，其输出是反归一化的结果，所以不用在外部做额外处理

    # OBS 的类型：
    # simulator state： 模拟器迭代用的state
    # simulator full state: 根据模拟器迭代用的state 推导出来的完整的state （给出了额外的环境信息），这个state 只用于 生成obs
    # obs state： 根据 full state 生成 obs，obs 是 策略训练用的

    def sample_sim_training_data(traj_batch_size=None, raw_state=False):
        if traj_batch_size is None:
            traj_batch_size = args.trajectory_batch
        obs, _, acs, _, lengths = sim_training_dataset.get_next_batch(traj_batch_size)
        if args.clip_policy_bound:
            obs = np.clip(obs, -1 * OBS_BOUND, OBS_BOUND)
        if not raw_state:
            obs = data_normalization(obs)
        return obs, None, acs, lengths

    def sample_next_batch_data(type, iter, traj_batch_size=None):
        if traj_batch_size is None:
            traj_batch_size = args.trajectory_batch
        MappingDirecition.config_check(args.mapping_direction)
        LossType.config_check(type)
        if args.mapping_direction == MappingDirecition.RSR and type == LossType.VAE \
                or args.mapping_direction == MappingDirecition.SRS and type == LossType.GAN:
            use_dataset = True
        else:
            use_dataset = False
        if not args.use_env_sample:
            use_dataset = True
        imgs = None
        if use_dataset:
                obs, imgs, acs, _, lengths = expert_dataset.get_next_batch(traj_batch_size)
        else:
            if (iter % args.data_reused_times == 0 or not trajecory_buffer.can_sample(
                    args.trajectory_batch * args.data_reused_times * 10)) and not trajecory_buffer.is_full():
                obs, acs, imgs, rews, lengths = [], [], [], [], []
                for _ in range(traj_batch_size):
                    ret_dict = runner.run_traj(deter=False, mapping_holder=None,
                                               render_img=False, run_in_realworld=False)
                    while ret_dict[runner.TRAJ_LEN] == 0:
                        ret_dict = runner.run_traj(deter=False, mapping_holder=None,
                                                   render_img=False, run_in_realworld=False)

                    obs.append(ret_dict[runner.OB_TRAJ])
                    acs.append(ret_dict[runner.AC_TRAJ])
                    imgs.append(ret_dict[runner.IMG_TRAJ])
                    lengths.append(ret_dict[runner.TRAJ_LEN])
                    rews = [ret_dict[runner.TOTAL_REW]]
                    trajecory_buffer.add([ret_dict[runner.OB_TRAJ],
                                          ret_dict[runner.AC_TRAJ], ret_dict[runner.TRAJ_LEN]])
                obs = np.array(obs)
                acs = np.array(acs)
                imgs = np.array(imgs)
                logger.info("sample rew :{}".format(np.mean(rews)))
            else:
                obs, acs, lengths = trajecory_buffer.sample(traj_batch_size)
        if args.clip_policy_bound:
            obs = np.clip(obs, -1 * OBS_BOUND, OBS_BOUND)
        obs = data_normalization(obs)
        return obs, imgs, acs, lengths

    def one_step_transition(de_norm_obs_input, acs_input):
        pass_one_step_transition_test = False
        de_normalize_next_obs = []
        obs = []
        full_states = []
        acs = []
        for idx_selected in range(de_norm_obs_input.shape[0]):
            obs_trans = de_norm_obs_input[idx_selected].reshape([-1, de_norm_obs_input.shape[-1]])
            acs_trans = acs_input[idx_selected].reshape([-1, acs_real.shape[-1]])
            for idx in range(de_norm_obs_input.shape[1] - 1):  # skip the zero time-step.
                de_normalize_ob = obs_trans[idx]
                # de_normalize_ob = data_denormalization(ob)
                # de_normalize_ob = de_normalize_obs_trans[idx]
                ac = acs_trans[idx]
                if np.all(de_normalize_ob == 0):
                    idx -= 1
                    break
                try:
                    de_normalize_next_ob, r, d, info = env.set_ob_and_step(de_normalize_ob, ac,
                                                                           ret_full_state=is_robot_env)
                    full_state = info['full_state']
                except Exception as e:
                    logger.warn(e)
                    logger.warn("[INVALID OB] idx {}".format(idx))
                    logger.info("transition test")
                    logger.info("this ob", de_normalize_ob)
                    next_obs_pred = transition_learner.pred(obs_real[0], acs_real[0])
                    logger.info(next_obs_pred.max(axis=0))
                    logger.info(next_obs_pred.min(axis=0))
                    logger.info("encoder test")
                    if idx > 0:
                        logger.info("last state predict")
                        next_obs_pred = transition_learner.pred(obs_trans[idx - 1:idx],
                                                                acs_trans[idx - 1: idx])
                        logger.info(next_obs_pred.max(axis=0))
                        logger.info(next_obs_pred.min(axis=0))
                    logger.ma_record_tabular("warn/invalid_ob", 1, 10000)
                    break
                    # raise RuntimeError
                if np.random.random() < 0.1 and not pass_one_step_transition_test:
                    # transition_consistency_test(data_normalization(de_normalize_ob), ac, data_normalization(de_normalize_next_ob))
                    pass_one_step_transition_test = True
                obs.append(de_normalize_ob)
                acs.append(ac)
                full_states.append(full_state)
                de_normalize_next_obs.append(de_normalize_next_ob)
            if len(obs) > args.dyn_batch_size:
                break
        logger.ma_record_tabular("warn/invalid_ob", 0, 10000)
        de_normalize_next_obs = np.array(de_normalize_next_obs)
        next_obs = de_normalize_next_obs
        obs = np.array(obs)
        acs = np.array(acs)
        full_states = np.array(full_states)
        assert np.where(np.isnan(obs))[0].shape[0] == 0
        assert np.where(np.isnan(next_obs))[0].shape[0] == 0
        return obs, acs, next_obs, full_states, idx

    def safe_one_step_transition(norm_obs_input, raw_acs_input):
        norm_obs_input_clip = np.clip(norm_obs_input, update_dynamics_range_min_trans_learn,
                                      update_dynamics_range_max_trans_learn)
        obs_input, acs_input, next_obs_input, full_states, end_idx = one_step_transition(data_denormalization(norm_obs_input_clip),
                                                                       raw_acs_input)
        obs_input = data_normalization(obs_input)
        next_obs_input = data_normalization(next_obs_input)
        next_obs_input = np.clip(next_obs_input, update_dynamics_range_min_trans_learn, update_dynamics_range_max_trans_learn)
        return obs_input, acs_input, next_obs_input, full_states, end_idx

    def obs_acs_reshape(obs_input, acs_input):
        # alignment
        obs_train = obs_input[:, :-1].reshape([-1, obs_input.shape[-1]])
        acs_train = acs_input[:, :-1].reshape([-1, acs_input.shape[-1]])
        obs_next_train = obs_input[:, 1:].reshape([-1, obs_input.shape[-1]])
        # remove mask states.
        not_done_idx = np.where(np.any(obs_train != 0, axis=1))
        obs_train = obs_train[not_done_idx]
        acs_train = acs_train[not_done_idx]
        obs_next_train = obs_next_train[not_done_idx]
        not_done_idx = np.where(np.any(obs_next_train != 0, axis=1))
        obs_train = obs_train[not_done_idx]
        acs_train = acs_train[not_done_idx]
        obs_next_train = obs_next_train[not_done_idx]
        return obs_train, acs_train, obs_next_train

    def transition_consistency_test(obs, acs, next_obs):
        de_normalize_next_ob2 = env.set_ob_and_step(data_denormalization(obs), acs)[0]
        norm_next_obs2 = data_normalization(de_normalize_next_ob2)
        norm_next_obs2 = np.clip(norm_next_obs2, update_dynamics_range_min_trans_learn, update_dynamics_range_max_trans_learn)
        next_obs = np.clip(next_obs, update_dynamics_range_min_trans_learn, update_dynamics_range_max_trans_learn)
        if 'Swimmer' in args.env_id:
            eb = 0.1
        else:
            eb = 0.01
        if not np.all(np.abs(next_obs - norm_next_obs2) < eb):
            logger.warn("ob: {}".format(obs))
            logger.warn("next_ob: {}".format(next_obs))
            logger.warn("next_obs2: {}".format(norm_next_obs2))
            raise RuntimeError

    def update_dynamic(obs_input, acs_input, next_obs_input, name=''):
        # init dataset
        # valid_idx = np.logical_and(np.all(raw_obs_input > -100, axis=-1), np.all(raw_obs_input < 100, axis=-1))
        # raw_obs_input_clip = raw_obs_input[valid_idx]
        # raw_acs_input = raw_acs_input[valid_idx]
        # obs_input, acs_input, next_obs_input, end_idx = time_used_wrap('one_step_transition',
        #                                                       safe_one_step_transition, norm_obs_input, raw_acs_input)

        # eval
        if obs_input.shape[0] == 0:
            logger.record_tabular('warn/no_valid_dynamics_{}'.format(name), 1)
            return

        obs_nex_pred = transition_learner.pred(obs_input, acs_input)
        r2 = compute_adjusted_r2(next_obs_input, obs_nex_pred)
        if r2 is not None:
            logger.record_tabular('performance/dynamic_adjusted_r2{}'.format(name), r2)
        if obs_input.shape[0] != 0:
            # first test
            batch_length = obs_input.shape[0]
            id_list = np.arange(0, batch_length)
            np.random.shuffle(id_list)
            mse_loss, max_error, l2_reg = time_used_wrap("update_transition",
                                                 transition_learner.update_transition,
                                                 obs_input,
                                                 acs_input,
                                                 next_obs_input)
            logger.record_tabular("performance_dynamics/sample-mse_loss{}".format(name),
                                  np.mean(mse_loss))
            logger.record_tabular("performance_dynamics/sample-l2_reg{}".format(name),
                                  np.mean(l2_reg))
            logger.record_tabular("performance_dynamics/sample-mse_max_error{}".format(name),
                                  max_error)
            if max_error > (adjust_allowed * 0.8) ** 2:
                logger.ma_record_tabular("performance_dynamics/sample-good_trans", 0, 100)
            else:
                logger.ma_record_tabular("performance_dynamics/sample-good_trans", 1, 100)
                # repeat training.
            if args.hard_training:
                mse_total_iter = 1000
            else:
                mse_total_iter = 0
            pass_one_step_transition_test = False

    logger.log('testing policy performance')
    rews = []
    for i in range(args.expert_perf_eval_times):
        if i % 10 == 0:
            logger.info("test expert perf {}".format(i))
        ret_dict = runner.run_traj(deter=False, mapping_holder=None, render_img=False,
                                    run_in_realworld=True)
        while ret_dict[runner.TRAJ_LEN] == 0:
            ret_dict = runner.run_traj(deter=False, mapping_holder=None, render_img=False,
                                        run_in_realworld=True)
        rews.append(ret_dict[runner.TOTAL_REW])
    expert_reward = np.mean(rews)
    if is_dapg_env(args.env_id):
        if args.env_id == 'pen-v0':
            expert_reward_lower_bound_assert = expert_dataset.avg_ret * ((args.max_sequence * 0.8) / 200)
        else:
            expert_reward_lower_bound_assert = expert_dataset.avg_ret * ((args.max_sequence * 0.6) / 200)
    else:
        expert_reward_lower_bound_assert = expert_dataset.avg_ret * ((args.max_sequence * 0.8) / 1000)

    assert expert_reward_lower_bound_assert < expert_reward, "dataset perf: {}, runner perf: {} < {}".\
        format(expert_dataset.avg_ret, expert_reward, expert_reward_lower_bound_assert)

    logger.info("expert_reward :{}".format(expert_reward))
    best_perf = 0.0
    update_ckp = False

    max_error_list = deque(maxlen=50)
    max_error_list.append(np.inf)
    inc_batch_counter = 0

    # step 5: unit test
    obs, acs = [], []

    logger.log('testing dynamics')
    for _ in tqdm(range(args.trajectory_batch)):
        ret_dict = runner.run_traj(deter=False, mapping_holder=None, render_img=False, run_in_realworld=False)
        while ret_dict[runner.TRAJ_LEN] == 0:
            ret_dict = runner.run_traj(deter=False, mapping_holder=None, render_img=False, run_in_realworld=False)
        obs.append(ret_dict[runner.OB_TRAJ])
        acs.append(ret_dict[runner.AC_TRAJ])
    obs_real = np.array(obs)
    acs_real = np.array(acs)

    # unit tset for a new dataset.
    # obs_real_dyn, _, acs_real_dyn, lengths = sample_sim_training_data(1000, raw_state=True)
    # obs_real = np.concatenate([obs_real_dyn, obs_real], axis=0)
    # acs_real = np.concatenate([acs_real_dyn, acs_real], axis=0)
    #
    # eb = 1e-2
    # for i in range(obs_real.shape[0]):
    #     obs_trans, acs_trans, next_obs, _, end_idx = one_step_transition(obs_real[i:i+1], acs_real[i:i+1])
    #
    #     if next_obs[:-1][np.abs(obs_trans[1:] - next_obs[:-1]) >= eb].shape[0] != 0:
    #         invalid_idx = np.where(np.any(np.abs(obs_trans[1:] - next_obs[:-1]) >= eb, axis=-1))[0]
    #         shift_obs = obs_trans[1:]
    #         for idx in invalid_idx:
    #             logger.info("traj {}, {}, obs : {}".format(i, idx, shift_obs[idx]))
    #             logger.info("traj {}, {}, next obs : {}".format(i, idx, next_obs[idx]))
    #         raise RuntimeError


    # step 6: pretrain dynamics model
    if args.emb_dynamic and args.load_date is '':
        if osp.exists(dynamics_model_path) and osp.exists(dynamics_model_param_path) and \
                not args.retrain_dynamics:
            weights = list(np.load(dynamics_model_path, allow_pickle=True))
            transition_learner.pretrained_value_assignment(*weights)
            hp_dict = pickle.load(open(dynamics_model_param_path, 'rb'))
            transition_learner.lr = hp_dict['lr']
            logger.info("pass pretrain")
        else:
            # fine-tune and test
            logger.info("train a new dynamics model")
            break_counter = 0
            inc_batch_counter = 0
            counter = 0
            while True:
                obs_real, _, acs_real, lengths = sample_sim_training_data(100)
                obs_train, acs_train, obs_next_train = obs_acs_reshape(obs_real, acs_real)
                assert np.where(np.isnan(obs_train))[0].shape[0] == 0
                assert np.where(np.isnan(acs_train))[0].shape[0] == 0
                loss, max_error, l2_reg = transition_learner.update_transition(obs_train, acs_train, obs_next_train,
                                                                       lr=args.dyn_lr_pretrain)
                if counter % 50 == 0:
                    logger.info("[{}] train loss: {}".format(counter, loss))
                    transition_learner.copy_params_to_target()
                    obs_nex_pred = transition_learner.pred(obs_train, acs_train)
                    logger.info("r^2 {}".format(compute_adjusted_r2(obs_next_train, obs_nex_pred)))
                    logger.info("[{}] max error: {}".format(counter, max_error))
                    logger.record_tabular("pretrain/max_error", np.mean(max_error))
                    logger.record_tabular("pretrain/r2", compute_adjusted_r2(obs_next_train, obs_nex_pred))
                    logger.record_tabular("pretrain/loss", np.mean(loss))
                    logger.record_tabular("pretrain/l2_reg", np.mean(l2_reg))
                    logger.record_tabular("pretrain/inc_batch_counter", inc_batch_counter)
                    logger.record_tabular("pretrain/break_counter", break_counter)
                    logger.dump_tabular()
                if np.min(max_error_list) > max_error:
                    inc_batch_counter = 0
                else:
                    inc_batch_counter += 1
                max_error_list.append(max_error)

                if inc_batch_counter >= 200 and counter > 100000:
                    # transition_learner.lr *= 0.95
                    inc_batch_counter = 0
                    new_weights = sess.run(transition.global_variables())
                    new_weights = np.array(new_weights, dtype=object)
                    np.save(dynamics_model_path, new_weights)
                    with open(dynamics_model_param_path, 'wb') as f:
                        pickle.dump({"lr": transition_learner.lr}, file=f)
                    break
                tester.time_step_holder.set_time(counter)
                if counter > 300000:
                    logger.warn('Embedding Dynamic may not be well trained', UserWarning)
                    break
                if counter % 1000 == 0:
                    new_weights = sess.run(transition.global_variables())
                    new_weights = np.array(new_weights, dtype=object)
                    np.save(dynamics_model_path, new_weights)
                    with open(dynamics_model_param_path, 'wb') as f:
                        pickle.dump({"lr": transition_learner.lr}, file=f)
                    tester.sync_log_file()
                if max_error < (adjust_allowed * 0.8) ** 2:
                    break_counter += 1
                else:
                    break_counter = 0
                if break_counter >= 20 and counter > 100000:
                    break
                counter += 1


    # load variables

    if args.load_date is not '':
        from RLA.easy_log.tester import load_from_record_date
        load_from_record_date(task_name=task_name_gen(), record_date=args.load_date)
        start_epoch = tester.time_step_holder.get_time()
    else:
        start_epoch = 0

    tester.time_step_holder.set_time(start_epoch)
    too_stronger_dis = False
    min_max_error = np.inf
    for iter in range(start_epoch, total_timesteps):
        time_record('epoch')
        tester.time_step_holder.set_time(iter)
        logger.info("iter {}".format(iter))
        acs_real, obs_real, imgs_real, obs_real = None, None, None, None
        ob_real2sim, ob_real2sim2real = None, None
        # 这里和img是否对应？
        obs_real, imgs_real, acs_real, lengths = time_used_wrap('VAE-sample', sample_next_batch_data, LossType.VAE, iter)
        obs_sim, _, acs_sim, lengths = time_used_wrap('GAN-sample', sample_next_batch_data, LossType.GAN, iter)
        res_dict = var_seq.infer_data(S_r=obs_real, O_r=imgs_real, A_r=acs_real,
                                      S_sim=obs_sim, A_sim=acs_sim,
                                      adjust_allowed=adjust_allowed)
        var_length, var_length_sim = res_dict["var_length"], res_dict["var_length_sim"]
        all_hidden_state, all_cycle_hidden_state = res_dict["all_hidden_state"], res_dict["all_cycle_hidden_state"]
        ob_real2sim = res_dict["hat_S_r"]
        ob_real2sim2real = res_dict["hat_O_r"]

        # no need to perform full_state_to_state here because no obs are fed into networks
        O_r_rollout, O_r_first, A_r_rollout, A_r_first, A_sim_rollout, A_sim_first, S_sim_rollout, S_r_rollout, \
        prev_hidden_state, prev_cycle_hidden_state = \
            var_seq.data_preprocess(S_r=obs_real, O_r=imgs_real, A_r=acs_real, S_sim=obs_sim, A_sim=acs_sim,
                                    var_length=var_length,  var_length_sim=var_length_sim,
                                    all_hidden_state=all_hidden_state,
                                    all_cycle_hidden_state=all_cycle_hidden_state)

        # train D
        if iter == 0:
            dte = 10
        else:
            dte = dis_train_epoch
        if (not too_stronger_dis) or iter % 20 == 0:
            for _ in range(dte):
                # obs_sim, acs_sim = obs_real, acs_real
                res_dict_d = time_used_wrap('d_train', var_seq.dis_train,
                                          O_r_rollout, O_r_first, A_r_rollout,
                                          A_r_first, A_sim_rollout, A_sim_first,
                                          S_sim_rollout, S_r_rollout,
                                          prev_hidden_state, prev_cycle_hidden_state, global_steps=iter,
                                          adjust_allowed=adjust_allowed, add_summary=(iter % 100 == 0))
                idx = np.random.randint(0, len(res_dict_d["summary"]))
                dis_summary = res_dict_d["summary"][idx]
                if dis_summary is not None:
                    tester.add_summary_to_logger(dis_summary, freq=100, name='dis')
                max_len = dis_train_epoch * 10
                logger.ma_record_tabular("loss/dis_loss", np.mean(res_dict_d["dis_loss"]), max_len)
                logger.ma_record_tabular("loss/minimax_loss", np.mean(res_dict_d["minimax_loss"]), max_len)
                logger.ma_record_tabular("loss/l2_reg_dis_loss", np.mean(res_dict_d["l2_reg_dis_loss"]), max_len)
                logger.ma_record_tabular("lr/dis_lr", np.mean(res_dict_d["dis_lr"]), max_len)
                logger.ma_record_tabular("acc/dis_accuracy_real", np.mean(res_dict_d["dis_accuracy_real"]), max_len)
                logger.ma_record_tabular("acc/dis_accuracy_fake", np.mean(res_dict_d["dis_accuracy_fake"]), max_len)
                logger.ma_record_tabular("acc/dis_real_prob", np.mean(res_dict_d["dis_real_prob"]), max_len)
                logger.ma_record_tabular("acc/dis_fake_prob", np.mean(res_dict_d["dis_fake_prob"]), max_len)
                logger.ma_record_tabular("learning/d_grad_norm", np.mean(res_dict_d["d_grad_norm"]), max_len)
                if args.cycle_loss:
                    logger.ma_record_tabular("learning/cycle_d_grad_norm", np.mean(res_dict_d["cycle_d_grad_norm"]), max_len)
                    logger.ma_record_tabular("acc/obs_dis_accuracy_real", np.mean(res_dict_d["obs_dis_accuracy_real"]), max_len)
                    logger.ma_record_tabular("acc/obs_dis_accuracy_fake", np.mean(res_dict_d["obs_dis_accuracy_fake"]), max_len)
                    logger.ma_record_tabular("loss/img_dis_loss", np.mean(res_dict_d["img_dis_loss"]), max_len)
                if args.gan_loss == GanLoss.MINIMAX:
                    if np.mean(res_dict_d["dis_fake_prob"]) < 0.4:
                        too_stronger_dis = True
                    else:
                        too_stronger_dis = False
                else:
                    too_stronger_dis = False
        # train G
        if not args.dis_test:
            for _ in range(mapping_train_epoch):
                res_dict = time_used_wrap('mapping_train', var_seq.mapping_train,
                                          O_r_rollout, O_r_first, A_r_rollout, A_r_first, A_sim_rollout, A_sim_first,
                                          S_sim_rollout, S_r_rollout,
                                          prev_hidden_state, prev_cycle_hidden_state,
                                          global_steps=iter, adjust_allowed=adjust_allowed, add_summary=(iter % 100 == 0))
                idx = np.random.randint(0, len(res_dict["summary"]))
                summary = res_dict["summary"][idx]
                if summary is not None: tester.add_summary_to_logger(summary, freq=100, name='gen')
                max_len = mapping_train_epoch * 10
                logger.ma_record_tabular("loss/mapping_loss", np.mean(res_dict["all_loss"]), max_len)
                logger.ma_record_tabular("loss/gen_loss", np.mean(res_dict["gen_loss"]), max_len)
                logger.ma_record_tabular("loss/mapping_likelihood", np.mean(res_dict["mapping_likelihood"]), max_len)
                logger.ma_record_tabular("loss/mapping_l2_loss", np.mean(res_dict["mapping_l2_loss"]), max_len)
                logger.ma_record_tabular("learning/m_grad_norm", np.mean(res_dict["m_grad_norm"]), max_len)
                logger.ma_record_tabular("lr/gen_lr", np.mean(res_dict["gen_lr"]), max_len)
                if args.cycle_loss:
                    logger.ma_record_tabular("loss/obs_gen_loss", np.mean(res_dict["obs_gen_loss"]), max_len)
                    logger.ma_record_tabular("loss/state_mapping_likelihood",
                                             np.mean(res_dict["state_mapping_likelihood"]), max_len)
                # if np.mean(res_dict["m_grad_norm"]) > 10000:
                #     logger.warn("grad norm is", np.mean(res_dict["m_grad_norm"]))
                #     import pdb; pdb.set_trace()

            if args.emb_dynamic:
                time_record('update_dynamic')

                # obs_train, acs_train, obs_next_train = obs_acs_reshape(obs_real, acs_real)
                # data_mse_loss, max_error = transition_learner.update_transition(obs_train[:args.dyn_batch_size],
                #                                                                 acs_train[:args.dyn_batch_size],
                #                                                                 obs_next_train[:args.dyn_batch_size])
                # logger.record_tabular("performance/mse_loss_data", np.mean(data_mse_loss))
                # logger.record_tabular("performance_dynamic/mse_max_error_data", max_error)
                obs_sim_dyn, _, acs_sim_dyn, _ = sample_sim_training_data()
                # update_dynamic(obs_sim_dyn, acs_sim_dyn, '-sim')
                # update_dynamic(ob_real2sim, acs_real, '-policy_data')  # action in dataset

                ob_real2sim_trans = np.reshape(ob_real2sim, [-1, ob_real2sim.shape[-1]])
                de_normalize_obs_trans = data_denormalization(ob_real2sim_trans)
                time_record("one_step_transition")
                if not is_robot_env:
                    policy_infer_acs = model.step(de_normalize_obs_trans,
                                                  deterministic=args.deter_policy)[0]
                    if args.clip_acs:
                        policy_infer_acs = np.clip(policy_infer_acs, env.action_space.low, env.action_space.high)
                    policy_infer_acs = np.reshape(policy_infer_acs,  list(ob_real2sim.shape[:2]) + [acs_real.shape[-1]])
                    acs_dyn_train = np.concatenate([acs_sim_dyn, acs_real, policy_infer_acs], axis=0)
                    obs_dyn_train = np.concatenate([obs_sim_dyn, ob_real2sim, ob_real2sim], axis=0)
                    obs_input, acs_input, next_obs_input, full_states, _ = safe_one_step_transition(obs_dyn_train,
                                                                                                    acs_dyn_train)
                else:
                    obs_input_r2s_data, acs_input_r2s_data, \
                    next_obs_input_r2s_data, full_states, _ = safe_one_step_transition(ob_real2sim, acs_real)
                    policy_infer_acs = model.full_state_step(full_states, deterministic=args.deter_policy)[0]
                    if args.clip_acs:
                        policy_infer_acs = np.clip(policy_infer_acs, env.action_space.low, env.action_space.high)

                    obs_input_r2s_infer, acs_input_r2s_infer, \
                    next_obs_input_r2s_infer, _, _ = safe_one_step_transition(obs_input_r2s_data[None], policy_infer_acs[None])

                    obs_input_data, acs_input_data, \
                    next_obs_input_data, _, _ = safe_one_step_transition(obs_input_r2s_data[None], policy_infer_acs[None])

                    obs_input = np.concatenate([obs_input_data, obs_input_r2s_data, obs_input_r2s_infer], axis=0)
                    acs_input = np.concatenate([acs_input_data, acs_input_r2s_data, acs_input_r2s_infer], axis=0)
                    next_obs_input = np.concatenate([next_obs_input_data, next_obs_input_r2s_data, next_obs_input_r2s_infer], axis=0)

                time_record_end("one_step_transition")

                update_dynamic(obs_input, acs_input, next_obs_input, '-merge')
                # if inc_batch_counter >= 100:
                #     transition_learner.lr *= 0.9
                #     logger.record_tabular("performance/lr_dyn", transition_learner.lr)
                #     inc_batch_counter = 0
                transition_learner.copy_params_to_target()
                time_record_end('update_dynamic')

        if args.dynamic_test:
            map_holder = MappingHolder(stack_imgs=args.stack_imgs, is_dynamics_emb=args.emb_dynamic,
                                       var_seq=var_seq, default_rnn_state=default_zero_state,

                                       obs_shape=env.state_space.shape[0], adjust_allowed=adjust_allowed,
                                       init_first_state=args.init_first_state, data_mean=state_mean_std[0],
                                           data_std=state_mean_std[1])
            ret_dict = runner.run_traj(deter=False, mapping_holder=map_holder,
                                       render_img=False, run_in_realworld=True)
            while ret_dict[runner.TRAJ_LEN] == 0:
                ret_dict = runner.run_traj(deter=False, mapping_holder=map_holder,
                                           render_img=False, run_in_realworld=True)

            obs_trans, acs_trans, next_obs, _, end_idx = one_step_transition(np.expand_dims(ret_dict[runner.OB_TRAJ], axis=0),
                                                                        np.expand_dims(ret_dict[runner.AC_TRAJ], axis=0))
            mse_loss_dynamic, max_error_dynamic = transition_learner.\
                update_transition(obs_trans[:args.dyn_batch_size * 10],
                                  acs_trans[:args.dyn_batch_size * 10],
                                  next_obs[:args.dyn_batch_size * 10])
            logger.record_tabular("performance_dynamic/mse_loss_dynamic (before)", np.mean(mse_loss_dynamic))
            logger.record_tabular("performance_dynamic/mse_max_error_dynamic (before)", max_error_dynamic)
            for mse_iters in range(10):
                mse_loss_dynamic = transition_learner.\
                    update_transition(obs_trans[:args.dyn_batch_size * 10],
                                      acs_trans[:args.dyn_batch_size * 10],
                                      next_obs[:args.dyn_batch_size * 10])
                max_error_dynamic = transition_learner.\
                    update_transition(obs_trans[:args.dyn_batch_size * 10],
                                      acs_trans[:args.dyn_batch_size * 10],
                                      next_obs[:args.dyn_batch_size * 10])
                if mse_loss_dynamic[0] < (adjust_allowed * 0.8) ** 2:
                    break
            logger.record_tabular("performance_dynamic/mse_loss_dynamic (after)", np.mean(mse_loss_dynamic))
            logger.record_tabular("performance_dynamic/mse_max_error_dynamic (after)", max_error_dynamic)

        if iter % 50 == 0:
            tester.sync_log_file()
        tester.update_fph(cum_epochs=iter - start_epoch)

        # evaluation
        if not args.dis_test:
            logger.record_tabular(key='performance/r2s_adjusted_r2', val=compute_adjusted_r2(obs_real, ob_real2sim))
            logger.record_tabular(key='performance/r2s_rmse', val=compute_rmse(obs_real, ob_real2sim))
            logger.record_tabular(key='performance/r2s_rmse-percent', val=compute_rmse_d_bias(obs_real, ob_real2sim))
            logger.record_tabular(key='performance/r2s2r_image_mse', val=compute_image_mse(imgs_real, ob_real2sim2real))
        if iter % 200 == 0 and not args.dis_test:
            time_record('evaluation')
            assert np.where(np.all(ob_real2sim[np.all(obs_real == 0, axis=-1)] != 0, axis=-1))[0].shape[0] == 0
            assert np.where(np.all(obs_real[np.all(ob_real2sim == 0, axis=-1)] != 0, axis=-1))[0].shape[0] == 0
            # 所有img == 0 的帧 （最后三维是长宽和通道数），对应的重构图像也应该全部是0
            if np.where(
                np.all(ob_real2sim2real[np.all(imgs_real == 0, axis=(-1, -2, -3))] != 0,
                       axis=(-1, -2, -3)))[0].shape[0] != 0:
                for i in range(args.trajectory_batch):
                    should_be_zero_obs = ob_real2sim2real[i][np.all(imgs_real == 0, axis=(-1, -2, -3))]
                    invalid_idx = np.where(np.all(should_be_zero_obs != 0, axis=(-1, -2, -3)))[0]
                    for idx in invalid_idx:
                        logger.info("idx :", idx)
                        logger.info("ob_real2sim2real:", should_be_zero_obs[idx, :10, :10, 0])
                raise RuntimeError
            assert np.where(
                np.all(imgs_real[np.all(ob_real2sim2real == 0, axis=(-1, -2, -3))] != 0, axis=(-1, -2, -3)))[0].shape[0] == 0
            mapping_reward = []
            mapping_r2 = []
            mapping_r2_dyn = []
            mapping_max_se_dyn = []

            for _ in range(test_episode):
                map_holder = MappingHolder(stack_imgs=args.stack_imgs, is_dynamics_emb=args.emb_dynamic,
                                           var_seq=var_seq, default_rnn_state=default_zero_state,
                                           obs_shape=env.state_space.shape[0], adjust_allowed=adjust_allowed,
                                           init_first_state=args.init_first_state, data_mean=state_mean_std[0],
                                           data_std=state_mean_std[1])
                ret_dict = runner.run_traj(deter=False, mapping_holder=map_holder, render_img=True, run_in_realworld=True)
                while ret_dict[runner.TRAJ_LEN] == 0:
                    ret_dict = runner.run_traj(deter=False, mapping_holder=map_holder, render_img=True,
                                               run_in_realworld=True)

                ob_traj = ret_dict[runner.OB_TRAJ]
                r2s_ob_traj = ret_dict[runner.R2S_OB_TRAJ]
                rew = ret_dict[runner.TOTAL_REW]
                ac_traj = ret_dict[runner.AC_TRAJ]
                eval_r2 = compute_adjusted_r2(ob_traj, r2s_ob_traj)
                mapping_r2.append(eval_r2)
                mapping_reward.append(rew)
                obs_sample, acs_sample, obs_next_sample = obs_acs_reshape(np.expand_dims(ob_traj, axis=0),
                                                                          np.expand_dims(ac_traj, axis=0))

                for random_idx in range(obs_sample.shape[0]):
                    de_normalize_ob = obs_sample[random_idx]
                    ac = acs_sample[random_idx]
                    de_normalize_next_ob = obs_next_sample[random_idx]
                    # transition_consistency_test(data_normalization(de_normalize_ob), ac,
                    # data_normalization(de_normalize_next_ob))
                if obs_sample.shape[0] == 0:
                    continue
                if args.emb_dynamic:
                    norm_obs_sample = data_normalization(obs_sample)
                    norm_obs_next_sample = data_normalization(obs_next_sample)
                    obs_nex_pred = transition_learner.pred(norm_obs_sample, acs_sample)
                    eval_r2_dyn = compute_adjusted_r2(norm_obs_next_sample, obs_nex_pred)
                    # update_dynamics_range_min = np.minimum(update_dynamics_range_min, obs_sample.min(axis=0))
                    # update_dynamics_range_max = np.maximum(update_dynamics_range_max, obs_sample.max(axis=0))
                    dyn_mse = np.max(np.square(obs_next_sample - obs_nex_pred))
                    mapping_max_se_dyn.append(dyn_mse)

                    if eval_r2_dyn is not None:
                        mapping_r2_dyn.append(eval_r2_dyn)
                    logger.record_tabular(key='tune_params/max_range', val=np.max(norm_obs_sample - update_dynamics_range_max))
                    logger.record_tabular(key='tune_params/min_range', val=np.min(norm_obs_sample - update_dynamics_range_min))

            logger.record_tabular(key='performance/r2s_adjusted_r2-sample', val=np.mean(mapping_r2))
            logger.record_tabular(key='performance/reward', val=np.array(mapping_reward).mean())
            logger.record_tabular(key='performance/reward_ratio', val=np.array(mapping_reward).mean() / expert_reward)
            logger.record_tabular(key='performance/reward_std', val=np.array(mapping_reward).std())
            if len(mapping_max_se_dyn) > 0:
                logger.record_tabular('performance/dynamic_max_error-sample', np.max(mapping_max_se_dyn))

            if len(mapping_r2_dyn) > 0:
                logger.record_tabular('performance/dynamic_adjusted_r2-sample', np.mean(mapping_r2_dyn))
            if args.emb_dynamic:
                dim_max_error = np.max(np.square(obs_next_sample - obs_nex_pred), axis=0)
                obs_next_sample_clip = np.clip(obs_next_sample, update_dynamics_range_min_trans_learn, update_dynamics_range_max_trans_learn)
                dim_max_error_clip = np.max(np.square(obs_next_sample_clip - obs_nex_pred), axis=0)
                for id in range(len(dim_max_error_clip)):
                    logger.record_tabular('performance_dynamics/dim_max_error_{}'.format(id), dim_max_error[id])
                    logger.record_tabular('performance_dynamics/dim_max_error_{}_clip'.format(id), dim_max_error_clip[id])
            if best_perf < np.array(mapping_reward).mean() / expert_reward:
                update_ckp = True
                best_perf = np.array(mapping_reward).mean() / expert_reward
            logger.dump_tabular()
            if args.output_image and not args.dis_test:
                if iter % 600 == 0:
                    for img_index in np.arange(0, args.max_sequence, 10):
                        real_imgae = imgs_real[0, img_index]
                        generated_image = ob_real2sim2real[0, img_index]
                        output_image = np.concatenate((real_imgae, generated_image), axis=1)
                        os.makedirs(tester.results_dir + "imgs/timestep_{}".format(iter), exist_ok=True)
                        cv2.imwrite(tester.results_dir + "imgs/timestep_{}/{}.jpg".format(iter, img_index),
                                    (output_image * 255).astype(np.uint8))
            time_record_end('evaluation')
        if iter > 1000 and (iter % 1000 == 0) and args.do_save_checkpoint:
            tester.save_checkpoint()
            update_ckp = False
        time_record_end('epoch')
        logger.dump_tabular()

if __name__ == '__main__':
    main()
