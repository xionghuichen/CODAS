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
    adjust_allowed = args.adjust_allowed
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
        norm_obs = data_normalization(obs)
        return obs,norm_obs, imgs, acs, lengths

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



    max_error_list = deque(maxlen=50)
    max_error_list.append(np.inf)

    # step 5: unit test
    obs, acs = [], []


    if args.load_date is not '':
        from RLA.easy_log.tester import load_from_record_date
        load_from_record_date(task_name=task_name_gen(), record_date=args.load_date)
        start_epoch = tester.time_step_holder.get_time()
    else:
        start_epoch = 0

    tester.time_step_holder.set_time(start_epoch)

    time_record('epoch')
    tester.time_step_holder.set_time(iter)
    logger.info("iter {}".format(iter))
    acs_real, obs_real, imgs_real, obs_real = None, None, None, None
    ob_real2sim, ob_real2sim2real = None, None
    # 这里和img是否对应？
    raw_obs_real, obs_real, imgs_real, acs_real, lengths = time_used_wrap('VAE-sample', sample_next_batch_data, LossType.VAE, 0)
    raw_obs_sim, obs_sim, _, acs_sim, lengths = time_used_wrap('GAN-sample', sample_next_batch_data, LossType.GAN, 0)
    res_dict = var_seq.infer_data(S_r=obs_real, O_r=imgs_real, A_r=acs_real,
                                    S_sim=obs_sim, A_sim=acs_sim,
                                    adjust_allowed=adjust_allowed)

    ob_real2sim2real = res_dict["hat_O_r"]
    normed_state_sim = res_dict["hat_S_r"]
    normed_state_sim_clip = np.clip(normed_state_sim, update_dynamics_range_min_trans_learn,
                                      update_dynamics_range_max_trans_learn)
    denormed_state_sim = data_denormalization(normed_state_sim_clip)

    mapping_reward = []
    mapping_r2 = []
    mapping_r2_dyn = []
    mapping_max_se_dyn = []

    map_holder = MappingHolder(stack_imgs=args.stack_imgs, is_dynamics_emb=args.emb_dynamic,
                                    var_seq=var_seq, default_rnn_state=default_zero_state,
                                    obs_shape=env.state_space.shape[0], adjust_allowed=adjust_allowed,
                                    init_first_state=args.init_first_state, data_mean=state_mean_std[0],
                                    data_std=state_mean_std[1])
    ret_dict = runner.run_traj(deter=True, mapping_holder=map_holder, render_img=True, run_in_realworld=True)
    while ret_dict[runner.TRAJ_LEN] == 0:
        ret_dict = runner.run_traj(deter=False, mapping_holder=map_holder, render_img=True,
                                        run_in_realworld=True)

    ob_traj = ret_dict[runner.OB_TRAJ]
    r2s_ob_traj = ret_dict[runner.R2S_OB_TRAJ]
    print(r2s_ob_traj.shape ,obs_real.shape, imgs_real.shape) # (200, 63) (20, 200, 63) (20, 200, 64, 64, 3)

    rew = ret_dict[runner.TOTAL_REW]
    ac_traj = ret_dict[runner.AC_TRAJ]

    mapping_reward.append(rew)
    obs_sample, acs_sample, obs_next_sample = obs_acs_reshape(np.expand_dims(ob_traj, axis=0),
                                                                    np.expand_dims(ac_traj, axis=0))

    if args.emb_dynamic:
        norm_obs_sample = data_normalization(obs_sample)
        norm_obs_next_sample = data_normalization(obs_next_sample)
        obs_nex_pred = transition_learner.pred(norm_obs_sample, acs_sample)
        eval_r2_dyn = compute_adjusted_r2(norm_obs_next_sample, obs_nex_pred)
        dyn_mse = np.max(np.square(obs_next_sample - obs_nex_pred))
        mapping_max_se_dyn.append(dyn_mse)

        if eval_r2_dyn is not None:
            mapping_r2_dyn.append(eval_r2_dyn)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_save_dir = os.path.join("videos", args.env_id)
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)
    #write real video
    
    video_save_path = os.path.join(video_save_dir, "real.mp4")
    video_writer = cv2.VideoWriter(video_save_path, fourcc, 30, (256, 256))
    for state in raw_obs_real[0]:
        env._set_ob(state)
        img = env.render(mode="rgb_array", width=256, height=256, camera_name='track')        
        video_writer.write(img)

    video_writer.release()
        
    #write re-rendered video
    video_save_path = os.path.join(video_save_dir, "rerender.mp4")
    video_writer = cv2.VideoWriter(video_save_path, fourcc, 30, (256, 256))
    # from collections import deque
    hist_state = deque(maxlen=10)
    for state in denormed_state_sim[0]:
        hist_state.append(state)
        env._set_ob(np.mean(np.array(hist_state), axis=0))
        img = env.render(mode="rgb_array", width=256, height=256, camera_name='track')        
        video_writer.write(img)

    video_writer.release()

    video_save_path = os.path.join(video_save_dir, "recon.mp4")
    video_writer = cv2.VideoWriter(video_save_path, fourcc, 30, (args.image_size, args.image_size))
    #write re constructed video
    for img_index in np.arange(0, args.max_sequence):
        generated_image = (ob_real2sim2real[0, img_index]*255.0).astype(np.uint8)
        video_writer.write(generated_image)
    video_writer.release()



if __name__ == '__main__':
    main()
