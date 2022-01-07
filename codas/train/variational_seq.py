import tensorflow as tf
import numpy as np
import math

from codas.utils import tf_util as U
from codas.utils.tf_basic import TfBasicClass
from codas.utils.config import *
from RLA.easy_log.tester import tester
from codas.train.mapping_func import Sim2Real, Real2Sim
from codas.utils.tf_func import mask_filter
from codas.utils.structure import *
from codas.utils.functions import robust_append_dict
from codas.train.discriminator import TrajDiscriminator

def logsigmoid(a):
    # Equivalent to tf.log(tf.sigmoid(a))
    return -tf.nn.softplus(-a)


# Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51
def logit_bernoulli_entropy(logits):
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


class VarSeq(object):
    def __init__(self, sess, sequence_length: int,
                 rollout_step: int, img_shape,
                 batch_size: int,
                 lr_dis: int, lr_gen: int, l2_coeff: float, dis_l2_coeff: float,
                 total_timesteps: int, decay_ratio: float,
                 dis_test: bool,
                 embedding: TfBasicClass, sim2real_mapping: TfBasicClass, real2sim_mapping: TfBasicClass,
                 reconstruct_clip: float, policy: TfBasicClass,
                 discriminator: TfBasicClass, obs_discriminator: TfBasicClass, grad_clip_norm: float,

                 emb_dynamic: bool, cycle_loss: bool,
                 encoder: TfBasicClass, init_first_state: bool,
                 decoder: TfBasicClass,  label_image_test: bool,
                 ob_shape, ac_shape, lambda_a: float, lambda_b: float,
                 minibatch_size: int, merge_d_train: bool,  stack_imgs:int, random_set_to_zero:bool, scope='var_seq'):
        self.scope = scope
        self.sess = sess
        self.merge_d_train = merge_d_train
        self.random_set_to_zero = random_set_to_zero
        self.minibatch_size = minibatch_size
        self.l2_coeff = l2_coeff
        self.dis_l2_coeff = dis_l2_coeff
        self.cycle_loss = cycle_loss
        self.reconstruct_clip = reconstruct_clip
        self.emb_dynamic = emb_dynamic
        self.sequence_length = sequence_length
        self.init_first_state = init_first_state
        self.rollout_step = rollout_step
        assert self.sequence_length % self.rollout_step == 0
        self.rollout_times = int(self.sequence_length / self.rollout_step)
        self.grad_clip_norm = grad_clip_norm
        self.policy = policy
        self.grad_clip = 5.0
        self.decay_ratio = decay_ratio
        self.lr_dis = lr_dis
        self.lr_gen = lr_gen
        self.embedding = embedding
        self.sim2real_mapping = sim2real_mapping
        self.real2sim_mapping = real2sim_mapping
        self.discriminator = discriminator
        self.obs_discriminator = obs_discriminator
        self.batch_size = batch_size
        self.encoder = encoder
        self.decoder = decoder
        self.label_image_test = label_image_test
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.total_timesteps = total_timesteps
        self.ob_shape = ob_shape
        self.dis_test = dis_test
        self.ac_shape = ac_shape
        self.stack_imgs = stack_imgs
        self.policy_loss = None
        self.batch_zero_state = None
        self.batch_zero_O = None
        self.batch_zero_A = None

        self.img_shape = img_shape
        self.img_shape_to_list = [self.img_shape[ImgShape.WIDTH], self.img_shape[ImgShape.HEIGHT], self.img_shape[ImgShape.CHANNEL]]

    def get_variable_mask(self, data):
        with tf.variable_scope('mask', reuse=False):
            return tf.expand_dims(tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2)),
                                  axis=-1)  # 第3维度是数据特征，如果有非0的数据，就会表示为1

    def model_setup(self):
        assert isinstance(self.sim2real_mapping, Sim2Real)
        assert isinstance(self.real2sim_mapping, Real2Sim)
        self.S_r_ph = U.get_placeholder(name='S_r_ph', dtype=tf.float32, shape=[None, None, self.ob_shape])
        # try use float 16 to reduce memory usage
        self.O_r_ph = U.get_placeholder(name='O_r_ph', dtype=tf.float32, shape=[None, None] + self.img_shape_to_list)
        self.O_r_first_ph = U.get_placeholder(name='O_r_first_ph', dtype=tf.float32, shape=[None] + self.img_shape_to_list)
        self.A_r_ph = U.get_placeholder(name='A_r', dtype=tf.float32, shape=[None, None, self.ac_shape])
        self.A_r_first_ph = U.get_placeholder(name='A_r_first_ph', dtype=tf.float32, shape=[None, self.ac_shape])
        self.S_sim_ph = U.get_placeholder(name='S_sim_ph', dtype=tf.float32, shape=[None, None, self.ob_shape])

        self.r2s_prev_hidden_state_ph = U.get_placeholder(name='r2s_prev_hidden_state', dtype=tf.float32,
                                                          shape=[None, self.real2sim_mapping.hidden_state_size])
        self.s2r2s_prev_hidden_state_ph = U.get_placeholder(name='s2r2s_prev_hidden_state', dtype=tf.float32,
                                                          shape=[None, self.real2sim_mapping.hidden_state_size])
        if isinstance(self.discriminator, TrajDiscriminator):
            self.dis_prev_hidden_state_ph = U.get_placeholder(name='dis_prev_hidden_state', dtype=tf.float32,
                                                              shape=[None, self.discriminator.hidden_state_size])
        else:
            self.dis_prev_hidden_state_ph = U.get_placeholder(name='dis_prev_hidden_state', dtype=tf.float32,
                                                              shape=[None, self.real2sim_mapping.hidden_state_size])
        self.A_sim_ph = U.get_placeholder(name='A_sim_ph', dtype=tf.float32, shape=[None, None, self.ac_shape])
        self.A_sim_first_ph = U.get_placeholder(name='A_sim_first_ph', dtype=tf.float32, shape=[None, self.ac_shape])
        self.adjust_allowed_ph = tf.placeholder(tf.float32, shape=[1], name='adjust_allowed_ph')

        self.global_steps = tf.placeholder(tf.int32, shape=[], name='steps')

        def get_variable_length(data):
            with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
                used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))  # 第3维度是数据特征，如果有非0的数据，就会表示为1
                length = tf.reduce_sum(used, reduction_indices=1)  # 第2维度，下标1，表示的是时间步，当数据为1的时候，说明该维度的时间步是正确的
                length = tf.cast(length, tf.int32)  # 转化为整型
            return length
        # pipeline
        # 1. sequential VAE:
        # O_r, A_r_prev -> hat_S_r
        # \hat_S_r, A_r_prev, O_r_prev -> \hat_O_r
        # 2. discriminator
        # \hat_S_r, A_r <-> S_sim, A_sim
        # 3. cycle generator:
        # S_sim, A_sim_prev -> \hat_O_sim
        # cycle_\hat_O_sim, A_sim_prev -> cycle_hat_S
        # 4. cycle discriminator:
        # cycle_\hat_O_sim, A_sim <-> \hat_O_r, A_r
        O_r = self.O_r_ph
        O_r_prev = self.O_r_ph
        O_r_prev = tf.concat([tf.expand_dims(self.O_r_first_ph, axis=1), O_r_prev[:, :-1]], axis=1)

        def stack_img_func(imgs):
            # padding zeros
            def stack_idx(idx):
                pre_pad_img = tf.zeros([tf.shape(imgs)[0], idx] + imgs.shape[2:].as_list(), dtype=imgs.dtype)
                post_pad_img = tf.zeros([tf.shape(imgs)[0], self.stack_imgs - 1 - idx] + imgs.shape[2:].as_list(),
                                        dtype=imgs.dtype)
                stacked_imgs = tf.concat([pre_pad_img, imgs, post_pad_img], axis=1)
                return stacked_imgs

            idx_list = tuple(list(range(self.stack_imgs)))
            st_imgs = list(map(stack_idx, idx_list))
            stack_imgs = tf.concat(st_imgs, axis=-1)[:, :-1 * (self.stack_imgs - 1)]
            return stack_imgs
        if self.stack_imgs > 1:
            self.stack_O_r = stack_img_func(O_r)
            self.stack_O_r_prev = stack_img_func(O_r_prev)
        else:
            self.stack_O_r = O_r
            self.stack_O_r_prev = O_r_prev
        A_r = self.A_r_ph
        A_r_prev = self.A_r_ph
        # self.A_r_prev = tf.concat([tf.zeros([tf.shape(A_r_prev)[0], 1, self.ac_shape]), A_r_prev[:, :-1]], axis=1)
        self.A_r_prev = tf.concat([tf.expand_dims(self.A_r_first_ph, axis=1), A_r_prev[:, :-1]], axis=1)
        # self.A_r_prev = self.A_r_first_ph # tf.concat([self.A_r_prev_pad_ph, A_r_prev[:, :-1]], axis=1)
        S_r = self.S_r_ph
        self.var_length = get_variable_length(S_r)
        self.mask = self.get_variable_mask(S_r)
        S_sim = self.S_sim_ph
        A_sim = self.A_sim_ph
        # A_sim_prev = tf.concat([tf.zeros([tf.shape(A_sim)[0], 1, self.ac_shape]), A_sim[:, :-1]], axis=1)
        A_sim_prev = tf.concat([tf.expand_dims(self.A_sim_first_ph, axis=1), A_sim[:, :-1]], axis=1)
        # A_sim_prev = self.A_sim_first_ph
        self.var_length_sim = var_length_sim = get_variable_length(S_sim)
        self.sim_mask = self.get_variable_mask(S_sim)


        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # encode (s, a) pairs into rnn

            self.encoded_O_r = self.encoder.obj_graph_construct(self.stack_O_r)
            self.encoded_pair = self.embedding.obj_graph_construct(tf.concat([self.encoded_O_r, self.A_r_prev], 2))

            # S_r_ph is used just for var_length calculation
            self.hat_S_r_distribution, self.all_r2s_hidden_state, self.r2s_hidden_state = \
                self.real2sim_mapping.obj_graph_construct((self.encoded_pair, self.A_r_prev, self.var_length,
                                                           self.r2s_prev_hidden_state_ph,
                                                           self.adjust_allowed_ph))

            self.batch_zero_state_op = self.real2sim_mapping.zero_state(batch_size=self.batch_size)

            self.hat_S_r = hat_S_r = self.hat_S_r_distribution.sample()
            self.hat_S_r_mean = self.hat_S_r_distribution.mean()
            assert self.hat_S_r_distribution.reparameterization_type == tf.contrib.distributions.FULLY_REPARAMETERIZED
            self.encoded_O_r_prev = self.encoder.obj_graph_construct(self.stack_O_r_prev)
            self.encoded_hat_O_r = self.sim2real_mapping.obj_graph_construct([hat_S_r, self.encoded_O_r_prev, self.A_r_prev])
            self.hat_O_r = self.decoder.obj_graph_construct([self.encoded_hat_O_r, self.img_shape_to_list])
            self.sim2real_encoded_image = None
            self.sim2real_encoded_pair = None
            self.sim2real2sim_state = None
            cycle_hat_S_sim = None
            if self.cycle_loss:
                assert not self.sim2real_mapping.real_ob_input, "have not implemented the RNN version of sim2real inference."
                self.sim2real_obs_emb = self.sim2real_mapping.obj_graph_construct([S_sim, self.stack_O_r_prev, A_sim_prev])
                self.cycle_hat_O_sim = self.decoder.obj_graph_construct([self.sim2real_obs_emb, self.img_shape_to_list])
                self.cycle_encoded_hat_O_sim = self.encoder.obj_graph_construct(self.cycle_hat_O_sim)
                self.cycle_encoded_pair = self.embedding.obj_graph_construct(tf.concat([self.cycle_encoded_hat_O_sim, A_sim_prev], 2))

                self.cycle_hat_S_distribution, self.all_s2r2s_hidden_state, self.s2r2s_hidden_state = \
                    self.real2sim_mapping.obj_graph_construct((self.cycle_encoded_pair, A_sim_prev, var_length_sim,
                                                               self.s2r2s_prev_hidden_state_ph, self.adjust_allowed_ph))
                # cycle_hat_S_sim = self.cycle_hat_S_distribution.sample()

            with tf.variable_scope('dis', reuse=False):
                # WARNING: Feeding the newest sample to ob_real2sim_sample makes the performance more stable
                # compared with the data sampled from the replay buffer (i.e., self.ob_real2sim_traj_ph).

                if self.dis_test:
                    dis_fake = self.discriminator.obj_graph_construct((S_r, A_r))
                else:
                    dis_fake = self.discriminator.obj_graph_construct((hat_S_r, A_r))

                dis_real = self.discriminator.obj_graph_construct((S_sim, A_sim))
                if self.cycle_loss:
                    assert not self.dis_test
                    obs_dis_fake = self.obs_discriminator.obj_graph_construct((self.cycle_hat_O_sim, A_sim))
                    obs_dis_real = self.obs_discriminator.obj_graph_construct((self.stack_O_r, A_r))
                else:
                    obs_dis_fake = obs_dis_real = None
                self.obs_dis_real_prob = None
                self.obs_dis_fake_prob = None
                if tester.hyper_param["gan_loss"] == GanLoss.MINIMAX:
                    self.dis_real_prob = tf.nn.sigmoid(dis_real)
                    self.dis_fake_prob = tf.nn.sigmoid(dis_fake)
                    if self.cycle_loss:
                        self.obs_dis_real_prob = tf.nn.sigmoid(obs_dis_real)
                        self.obs_dis_fake_prob = tf.nn.sigmoid(obs_dis_fake)
                elif tester.hyper_param["gan_loss"] == GanLoss.WGAN:
                    self.dis_real_prob = dis_real
                    self.dis_fake_prob = dis_fake

                with tf.variable_scope('loss_and_train', reuse=False):
                    dis_real_filter = mask_filter(dis_real, self.sim_mask[..., 0])
                    dis_fake_filter = mask_filter(dis_fake, self.mask[..., 0])
                    self.cycle_dis_loss = None
                    dis_var_list = self.discriminator.trainable_variables()
                    if tester.hyper_param["gan_loss"] == GanLoss.MINIMAX:
                        def minimax_loss(dis_real_input, dis_fake_input):
                            generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_input,
                                                                                     labels=tf.zeros_like(
                                                                                         dis_fake_input))
                            generator_loss = tf.reduce_mean(generator_loss)
                            # expert 标记为1
                            # -log(sigmoid(x))
                            expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real_input,
                                                                                  labels=tf.ones_like(dis_real_input))
                            expert_loss = tf.reduce_mean(expert_loss)
                            # Build entropy loss
                            logits = tf.concat([dis_fake_input, dis_real_input], 0)
                            entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
                            entropy_loss = - 0.001 * entropy
                            return expert_loss + generator_loss + entropy_loss

                        self.minimax_loss = minimax_loss(dis_real_filter, dis_fake_filter)
                        if self.cycle_loss:
                            obs_dis_real_filter = mask_filter(obs_dis_real, self.mask[..., 0])
                            obs_dis_fake_filter = mask_filter(obs_dis_fake, self.sim_mask[..., 0])
                            self.cycle_dis_loss = minimax_loss(obs_dis_real_filter, obs_dis_fake_filter)

                    elif tester.hyper_param["gan_loss"] == GanLoss.WGAN:
                        def wgan_loss(dis_real_input, dis_fake_input):
                            generator_loss = tf.reduce_mean(dis_fake_input)
                            expert_loss = - tf.reduce_mean(dis_real_input)
                            loss_d = generator_loss + expert_loss
                            alpha = tf.random_uniform(shape=[tf.shape(self.S_sim_ph)[0], tf.shape(self.S_sim_ph)[1], 1], minval=0., maxval=1.)
                            # interpolates_S_sim = alpha * S_sim  + (1 - alpha) * S_sim
                            # interpolates_A_sim = alpha * A_sim  + (1 - alpha) * A_sim
                            # dis_real_noise = self.discriminator.obj_graph_construct((interpolates_S_sim, interpolates_A_sim))
                            interpolates_S_fake = (1 - alpha) * hat_S_r + (1 - alpha) * S_sim
                            interpolates_A_fake = (1 - alpha) * A_r + (1 - alpha) * A_sim
                            interpolates_S_fake = mask_filter(interpolates_S_fake, self.mask[..., 0] * self.sim_mask[..., 0])
                            interpolates_A_fake = mask_filter(interpolates_A_fake, self.mask[..., 0] * self.sim_mask[..., 0])
                            dis_fake_noise = self.discriminator.obj_graph_construct((interpolates_S_fake, interpolates_A_fake))
                            grad = tf.gradients(dis_fake_noise, [interpolates_S_fake, interpolates_A_fake])
                            gps = []
                            for g in grad:
                                slop = tf.sqrt(tf.reduce_sum(tf.square(g), axis=-1))
                                gp = tf.reduce_mean((slop - 1.) ** 2)
                                gps.append(gp)

                            loss_d += 10 * tf.reduce_mean(gps)

                            return loss_d
                        self.minimax_loss = wgan_loss(dis_real_filter, dis_fake_filter)
                    else:
                        raise NotImplementedError

                    self.l2_reg_dis_loss = tf.add_n([tf.nn.l2_loss(var) for var in dis_var_list]) / len(dis_var_list)
                    self.dis_loss = self.minimax_loss + self.l2_reg_dis_loss * self.dis_l2_coeff
                    self.dis_learning_rate = tf.train.polynomial_decay(learning_rate=self.lr_dis,
                                                                       end_learning_rate=self.lr_dis * (
                                                                                   1 - self.decay_ratio),
                                                                       global_step=self.global_steps,
                                                                       decay_steps=self.total_timesteps,
                                                                       power=1, cycle=True)
                    dis_optimizer = tf.train.RMSPropOptimizer(self.dis_learning_rate)
                    self.dis_optim = dis_optimizer.minimize(self.dis_loss, var_list=dis_var_list)
                    dis_grad_and_vars = dis_optimizer.compute_gradients(self.dis_loss, dis_var_list)
                    _, self.d_grad_norm = tf.clip_by_global_norm(dis_grad_and_vars, 10)
                    if self.cycle_loss:
                        obs_dis_var_list = self.obs_discriminator.trainable_variables()
                        obs_dis_optimizer = tf.train.AdamOptimizer(self.dis_learning_rate * 10, beta1=0.5)
                        self.obs_dis_optim = obs_dis_optimizer.minimize(self.cycle_dis_loss, var_list=obs_dis_var_list)
                        obs_dis_grad_and_vars = obs_dis_optimizer.compute_gradients(self.cycle_dis_loss, obs_dis_var_list)
                        _, self.cycle_d_grad_norm = tf.clip_by_global_norm(obs_dis_grad_and_vars, 10)
                if tester.hyper_param["gan_loss"] == GanLoss.MINIMAX:
                    self.dis_accuracy_real = tf.nn.sigmoid(dis_real_filter) > 0.5
                    self.dis_accuracy_fake = tf.nn.sigmoid(dis_fake_filter) < 0.5
                    if self.cycle_loss:
                        self.obs_dis_accuracy_real = tf.nn.sigmoid(obs_dis_real_filter) > 0.5
                        self.obs_dis_accuracy_fake = tf.nn.sigmoid(obs_dis_fake_filter) < 0.5
                elif tester.hyper_param["gan_loss"] == GanLoss.WGAN:
                    self.dis_accuracy_real = dis_real_filter > 0.5
                    self.dis_accuracy_fake = dis_fake_filter < 0.5

            def safe_log(ops):
                assert not tester.hyper_param['safe_log'], "clip-log lead to worse performance."
                if tester.hyper_param['safe_log']:
                    return tf.clip_by_norm(tf.log(ops + 1e-6), 2)
                else:
                    return tf.log(ops + 1e-6)

            with tf.variable_scope('gen', reuse=False):
                with tf.variable_scope('reconstruction', reuse=False):
                    # TODO: check NAN (since mask is deleted)
                    self.logits_flat = logits_flat = tf.layers.flatten(
                        mask_filter(self.hat_O_r, self.mask[..., 0]))
                    self.labels_flat = labels_flat = tf.layers.flatten(
                        mask_filter(O_r, self.mask[..., 0]))
                    self.real2sim2real_logprob = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_flat,
                                                                                         labels=labels_flat)
                    self._recon_loss = tf.reduce_mean(self.real2sim2real_logprob)
                    self.mapping_likelihood = - self._recon_loss
                    self._output_image = tf.nn.sigmoid(self.hat_O_r)
                    self.hat_O_r_mask = tf.einsum('nsxyz,nsc->nsxyz', self._output_image, self.mask)
                    if self.cycle_loss:

                        # self.s2r2s_logprob_before_filter = tf.expand_dims(safe_log(self.cycle_hat_S_distribution.prob(S_sim)),
                        #     axis=-1)
                        self.s2r2s_logprob_before_filter = tf.sqrt(tf.reduce_mean(tf.square(self.cycle_hat_S_distribution.mean() - S_sim), axis=-1))
                        self.s2r2s_logprob = -1 * mask_filter(self.s2r2s_logprob_before_filter, self.sim_mask[..., 0])
                        self.state_mapping_likelihood = tf.reduce_mean(self.s2r2s_logprob, axis=0)
                        self.s2r2s_real2sim2_real_mean = self.cycle_hat_S_distribution.mean() * self.sim_mask

                    if self.label_image_test:
                        self.real2sim2real_logprob_before_filter = tf.expand_dims(safe_log(
                            self.hat_S_r_distribution.prob(S_r)), axis=-1)
                        self.real2sim2real_logprob = mask_filter(self.real2sim2real_logprob_before_filter,
                                                                 self.mask[..., 0])
                        if self.reconstruct_clip > 0:
                            self.real2sim2real_logprob = tf.minimum(self.real2sim2real_logprob, self.reconstruct_clip)
                        self.mapping_likelihood = tf.reduce_mean(self.real2sim2real_logprob, axis=0)

                    self.hat_S_r_mask = hat_S_r * self.mask
                    self.hat_S_r_mean_mask = self.hat_S_r_distribution.mean() * self.mask
                    self.hat_S_r_std_mask = self.hat_S_r_distribution.stddev() * self.mask

                with tf.variable_scope('loss_and_train', reuse=False):
                    dis_fake_filter = mask_filter(dis_fake, self.mask[..., 0])
                    if tester.hyper_param["gan_loss"] == GanLoss.WGAN:
                        generator_loss = dis_fake_filter
                    elif tester.hyper_param["gan_loss"] == GanLoss.MINIMAX:
                        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_filter,
                                                                                 labels=tf.zeros_like(dis_fake_filter))
                    else:
                        raise NotImplementedError
                    self.gen_loss = - tf.reduce_mean(generator_loss)

                    self.mapping_loss = self.lambda_a * self.gen_loss - self.lambda_b * self.mapping_likelihood
                    if self.cycle_loss:
                        obs_dis_fake_filter = mask_filter(obs_dis_fake, self.sim_mask[..., 0])
                        obs_generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=obs_dis_fake_filter,
                                                                                 labels=tf.zeros_like(
                                                                                     obs_dis_fake_filter))
                        self.obs_gen_loss = - tf.reduce_mean(obs_generator_loss)
                        self.mapping_loss += self.lambda_a * self.obs_gen_loss - self.lambda_b * self.state_mapping_likelihood
                    mapping_var_list = self.real2sim_mapping.trainable_variables() + self.embedding.trainable_variables()
                    if not self.label_image_test:
                        mapping_var_list += self.sim2real_mapping.trainable_variables()
                    if self.cycle_loss:
                        mapping_var_list += self.sim2real_mapping.trainable_variables()
                        mapping_var_list += self.decoder.trainable_variables()
                    mapping_var_list += self.encoder.trainable_variables()
                    if not self.label_image_test:
                        mapping_var_list += self.decoder.trainable_variables()
                    self.mapping_var_list = mapping_var_list
                    self.l2_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in mapping_var_list]) / len(mapping_var_list)
                    self.mapping_loss += self.l2_coeff * self.l2_reg_loss
                    self.gen_learning_rate = tf.train.polynomial_decay(learning_rate=self.lr_gen,
                                                                       end_learning_rate=self.lr_gen * (
                                                                                   1 - self.decay_ratio),
                                                                       global_step=self.global_steps,
                                                                       decay_steps=self.total_timesteps,
                                                                       power=1, cycle=True)
                    self.mapping_optim = tf.train.RMSPropOptimizer(self.gen_learning_rate)
                    self.grad_and_vars = grad_and_vars = self.mapping_optim.compute_gradients(self.mapping_loss,
                                                                                              mapping_var_list)
                    #
                    grads = tf.gradients(self.mapping_loss, mapping_var_list)

                    # tf.clip_by_norm(grad_and_vars[0], 10)
                    grads, self.m_grad_norm = tf.clip_by_global_norm(grads, 1000)
                    if self.grad_clip_norm > 0:
                        grad_and_vars = [(tf.clip_by_norm(gv[0], self.grad_clip_norm), gv[1]) for gv in grad_and_vars]
                        self.m_grad_clip_by_norm = [(tf.clip_by_norm(gv[0], self.grad_clip_norm), gv[1]) for gv in
                                                    grad_and_vars]
                    else:
                        grad_and_vars = list(zip(grads, mapping_var_list))

                    self.train_ops = self.mapping_optim.apply_gradients(grad_and_vars)
                # tf.linalg.norm(grad_and_vars[0][0])
                # self.m_grad_norm = tf.reduce_mean([(tf.linalg.norm(gv[0])) for gv in grad_and_vars])

        def log_grad_summary(input_grad_and_vars):
            for grad, var in input_grad_and_vars:
                if 'LayerNorm' in var.name:
                    continue
                if grad is None:
                    continue
                name = str(var.name).split('/')
                construct_name = '-'.join(name)
                if 'dense' in name:
                    dense_index = name.index('dense')
                    construct_name = '-'.join(name[:dense_index])
                    post_fix_name = '-'.join(name[dense_index:])
                    # for correctly mask, NAN gradient should not appear
                    tf.summary.histogram(construct_name + '/grad-' + post_fix_name, var)
                    tf.summary.histogram(construct_name + '/weight-' + post_fix_name, var)
                else:
                    tf.summary.histogram(construct_name + '/weight', var)
                    tf.summary.histogram(construct_name + '/grad', grad)
        log_grad_summary(grad_and_vars)
        with tf.variable_scope('r2sr2', reuse=False):
            tf.summary.histogram("log_prob", self.real2sim2real_logprob)
        self.summary = tf.summary.merge_all()

        self.dis_summary = []
        with tf.variable_scope('dis', reuse=False):
            tf.summary.histogram("real", tf.nn.sigmoid(dis_real_filter))
            tf.summary.histogram("fake", tf.nn.sigmoid(dis_fake_filter))
            if self.cycle_loss:
                tf.summary.histogram("obs_real", tf.nn.sigmoid(obs_dis_real_filter))
                tf.summary.histogram("obs_fake", tf.nn.sigmoid(obs_dis_fake_filter))
            # dis_grad_and_vars = self.dis_optim.compute_gradients(self.dis_loss, dis_var_list)
            log_grad_summary(dis_grad_and_vars)

        self.dis_summary = tf.summary.merge_all(scope='dis')
        self.do_nothing_op = tf.no_op()

    def infer_step(self, imgs, acs, prev_state, adjust_allowed, stoc_infer=True):
        feed_dict = {
            self.A_r_prev: acs[None, :],
            self.r2s_prev_hidden_state_ph: prev_state,
            self.adjust_allowed_ph: np.expand_dims(adjust_allowed, axis=0),
            self.var_length: [1],
            self.stack_O_r: imgs[None, :]
        }
        if stoc_infer:
            hat_S_op = self.hat_S_r
        else:
            hat_S_op = self.hat_S_r_mean
        hat_S, next_hidden = self.sess.run([hat_S_op, self.r2s_hidden_state], feed_dict=feed_dict)
        hat_S = hat_S[0]
        if self.emb_dynamic:
            next_hidden[:, -1 * self.ob_shape:] = hat_S

        return hat_S, next_hidden

    def data_reshape(self, seq_data):
        return seq_data.reshape([seq_data.shape[0] * self.rollout_times, self.rollout_step] + list(seq_data.shape[2:]))

    def data_preprocess(self, S_r, O_r, A_r, S_sim, A_sim, var_length, var_length_sim, all_hidden_state,
                        all_cycle_hidden_state):
        all_hidden_state_traj = all_hidden_state
        # sub_traj filter
        batch_length = []
        for i in range(self.rollout_times):
            batch_length.append(np.clip(var_length - i * self.rollout_step, 0, self.rollout_step))
        batch_length = np.concatenate(np.array(batch_length).T, axis=0)
        filter_idx = np.where(batch_length > 0)[0]

        batch_length_sim = []
        for i in range(self.rollout_times):
            batch_length_sim.append(np.clip(var_length_sim - i * self.rollout_step, 0, self.rollout_step))
        batch_length_sim = np.concatenate(np.array(batch_length_sim).T, axis=0)
        filter_idx_sim = np.where(batch_length_sim > 0)[0]
        all_hidden_state, S_r_rollout, A_r_rollout, O_r_rollout = map(self.data_reshape, (all_hidden_state, S_r, A_r, O_r))
        S_sim_rollout, A_sim_rollout = map(self.data_reshape, (S_sim, A_sim))
        mask_zero_idx = [i * self.rollout_times for i in range(self.batch_size)]

        # construct the ''first-timestep'' data
        def first_data_gen(zero_data, input_data):
            first_data = np.concatenate([np.expand_dims(zero_data, axis=0), input_data[:-1, -1]], axis=0)
            first_data[mask_zero_idx] = zero_data
            return first_data

        A_r_first = first_data_gen(self.get_batch_zero_A()[0], A_r_rollout)
        A_sim_first = first_data_gen(self.get_batch_zero_A()[0], A_sim_rollout)
        O_r_first = first_data_gen(self.get_batch_zero_O()[0], O_r_rollout)
        prev_hidden_state = first_data_gen(self.get_batch_zero_state()[0], all_hidden_state)
        if self.init_first_state :
            prev_hidden_state[mask_zero_idx][:, -1 * self.ob_shape:] = S_r[:, 0]
        # random set to zero data.
        if self.random_set_to_zero:
            A_r_first[:] = self.get_batch_zero_A()[0]
            A_sim_first[:] = self.get_batch_zero_A()[0]
            O_r_first[:] = self.get_batch_zero_O()[0]
        # prev_hidden_state[:] = self.get_batch_zero_state()[0]
        # A_r_first = A_sim_first = self.get_batch_zero_A()
        # O_r_first = self.get_batch_zero_O()
        # prev_hidden_state = self.get_batch_zero_state()
        # filter zero data


        def filter_zero_data(input_data, filter):
            return input_data[filter]

        prev_hidden_state, S_r_rollout, \
        A_r_rollout, O_r_rollout, A_r_first, O_r_first = map(lambda x: filter_zero_data(x, filter_idx),
                                                             (prev_hidden_state, S_r_rollout, A_r_rollout, O_r_rollout,
                                                                                    A_r_first, O_r_first))

        S_sim_rollout, A_sim_rollout, A_sim_first = map(lambda x: filter_zero_data(x, filter_idx_sim), (S_sim_rollout, A_sim_rollout,
                                                                                            A_sim_first))

        # unit test
        # all_prev_hidden_state = np.concatenate(
        #     [np.expand_dims(self.get_batch_zero_state(), axis=1), all_hidden_state_traj], axis=1)[:, :-1]
        # all_prev_hidden_state = self.data_reshape(all_prev_hidden_state)
        # all_prev_hidden_state = filter_zero_data(all_prev_hidden_state, filter_idx)
        # np.all(all_prev_hidden_state[:, 0] == prev_hidden_state)
        # prev_A_r = np.concatenate([np.expand_dims(self.get_batch_zero_A(), axis=1), A_r], axis=1)[:, :-1]
        # prev_A_r = self.data_reshape(prev_A_r)
        # prev_A_r = filter_zero_data(prev_A_r, filter_idx)
        # np.all(prev_A_r[:, 0] == A_r_first)

        if self.cycle_loss:
            prev_cycle_hidden_state = first_data_gen(self.get_batch_zero_state()[0],
                                                     self.data_reshape(all_cycle_hidden_state))[filter_idx_sim]
        else:
            prev_cycle_hidden_state = prev_hidden_state
        return O_r_rollout, O_r_first, A_r_rollout, A_r_first, A_sim_rollout, A_sim_first, S_sim_rollout, S_r_rollout, \
               prev_hidden_state, prev_cycle_hidden_state

    def dis_train(self, O_r_rollout, O_r_first, A_r_rollout, A_r_first, A_sim_rollout, A_sim_first, S_sim_rollout,
                  S_r_rollout, prev_hidden_state, prev_cycle_hidden_state,
                  global_steps, adjust_allowed, add_summary=False, update=True):
        res_dict = {}
        group_traj_len = S_r_rollout.shape[0]
        id_list = np.arange(0, group_traj_len)
        np.random.shuffle(id_list)
        id_list = np.concatenate([id_list, id_list])
        group_traj_len_sim = S_sim_rollout.shape[0]
        id_list_sim = np.arange(0, group_traj_len_sim)
        np.random.shuffle(id_list_sim)
        id_list_sim = np.concatenate([id_list_sim, id_list_sim])
        if self.merge_d_train or self.minibatch_size == -1:
            minibatch_size = max(group_traj_len, group_traj_len_sim)
        else:
            minibatch_size = self.minibatch_size

        for i in range(int(np.ceil(group_traj_len / minibatch_size))):
            feed_dict = {
                self.S_r_ph: S_r_rollout[id_list[i*minibatch_size:(i+1)*minibatch_size]],
                self.A_r_ph: A_r_rollout[id_list[i*minibatch_size:(i+1)*minibatch_size]],
                self.A_r_first_ph: A_r_first[id_list[i*minibatch_size:(i+1)*minibatch_size]],
                self.S_sim_ph: S_sim_rollout[id_list_sim[i*minibatch_size:(i+1)*minibatch_size]],
                self.A_sim_ph: A_sim_rollout[id_list_sim[i*minibatch_size:(i+1)*minibatch_size]],
                self.A_sim_first_ph: A_sim_first[id_list_sim[i*minibatch_size:(i+1)*minibatch_size]],
                self.global_steps: global_steps,
                self.r2s_prev_hidden_state_ph: prev_hidden_state[id_list[i*minibatch_size:(i+1)*minibatch_size]],
                self.O_r_ph: O_r_rollout[id_list[i*minibatch_size:(i+1)*minibatch_size]],
                self.O_r_first_ph: O_r_first[id_list[i*minibatch_size:(i+1)*minibatch_size]],
                self.adjust_allowed_ph: np.expand_dims(adjust_allowed, axis=0),
            }
            if self.cycle_loss:
                feed_dict.update({
                    self.s2r2s_prev_hidden_state_ph: prev_cycle_hidden_state[
                        id_list_sim[i * minibatch_size:(i + 1) * minibatch_size]]})
            ops = [self.dis_loss, self.r2s_hidden_state, self.dis_accuracy_real, self.dis_accuracy_fake,
                                 self.dis_learning_rate, self.d_grad_norm, self.dis_real_prob, self.dis_fake_prob,
                   self.minimax_loss, self.l2_reg_dis_loss]
            if self.cycle_loss:
                ops += [self.obs_dis_accuracy_real, self.obs_dis_accuracy_fake, self.cycle_dis_loss, self.cycle_d_grad_norm,
                       self.obs_dis_real_prob, self.obs_dis_fake_prob]
            else:
                ops += [self.do_nothing_op, self.do_nothing_op, self.do_nothing_op, self.do_nothing_op, self.do_nothing_op,
                        self.do_nothing_op]
            if add_summary:
                ops.append(self.dis_summary)
            else:
                ops.append(self.do_nothing_op)
            if update:
                ops.append(self.dis_optim)
            else:
                ops.append(self.do_nothing_op)
            if self.cycle_loss:
                if update:
                    ops.append(self.obs_dis_optim)
                else:
                    ops.append(self.do_nothing_op)
            if self.cycle_loss:
                res = self.sess.run(ops, feed_dict=feed_dict)[:-2]
            else:
                res = self.sess.run(ops, feed_dict=feed_dict)[:-1]
            robust_append_dict(res_dict, "dis_loss", res[0])
            robust_append_dict(res_dict, "dis_accuracy_real", res[2])
            robust_append_dict(res_dict, "dis_accuracy_fake", res[3])
            robust_append_dict(res_dict, "dis_lr", res[4])
            robust_append_dict(res_dict, "d_grad_norm", res[5])
            robust_append_dict(res_dict, "dis_real_prob", res[6])
            robust_append_dict(res_dict, "dis_fake_prob", res[7])
            robust_append_dict(res_dict, "minimax_loss", res[8])
            robust_append_dict(res_dict, "l2_reg_dis_loss", res[9])
            if self.cycle_loss:
                robust_append_dict(res_dict, "obs_dis_accuracy_real", res[10])
                robust_append_dict(res_dict, "obs_dis_accuracy_fake", res[11])
                robust_append_dict(res_dict, "img_dis_loss", res[12])
                robust_append_dict(res_dict, "cycle_d_grad_norm", res[13])
            robust_append_dict(res_dict, "summary", res[16])

        return res_dict

    def infer_data(self, S_r, O_r, A_r, S_sim, A_sim, adjust_allowed):
        prev_states = self.get_batch_zero_state()
        feed_dict = {
            self.S_r_ph: S_r,
            self.A_r_ph: A_r,
            self.A_r_first_ph: self.get_batch_zero_A(),
            self.O_r_ph: O_r,
            self.O_r_first_ph: self.get_batch_zero_O(),
            self.r2s_prev_hidden_state_ph: prev_states,
            self.S_sim_ph: S_sim,
            self.adjust_allowed_ph: np.expand_dims(adjust_allowed, axis=0)}
        ops = [self.all_r2s_hidden_state, self.hat_S_r_mask, self.hat_O_r_mask, self.var_length, self.var_length_sim]

        ops.append(self.do_nothing_op)
        if self.cycle_loss:
            feed_dict.update({
                self.A_sim_ph: A_sim,
                self.A_sim_first_ph: self.get_batch_zero_A(),
                self.s2r2s_prev_hidden_state_ph: prev_states,
            })
            ops.extend([self.all_s2r2s_hidden_state, self.var_length_sim])
        else:
            ops.extend([self.do_nothing_op, self.do_nothing_op])
        traj_res = self.sess.run(ops, feed_dict=feed_dict)
        res_dict = {}

        res_dict["all_hidden_state"] = traj_res[0]
        res_dict["var_length"] = traj_res[3]
        res_dict["var_length_sim"] = traj_res[4]

        if self.cycle_loss:
            all_cycle_hidden_state = traj_res[6]
        else:
            all_cycle_hidden_state = traj_res[0]

        res_dict["all_cycle_hidden_state"] = all_cycle_hidden_state
        res_dict["hat_S_r"] = traj_res[1]
        res_dict["hat_O_r"] = traj_res[2]
        res_dict["hat_A_r_mean"] = traj_res[5]

        return res_dict

    def mapping_train(self, O_r_rollout, O_r_first, A_r_rollout, A_r_first, A_sim_rollout, A_sim_first, S_sim_rollout,
                      S_r_rollout, prev_hidden_state, prev_cycle_hidden_state,
                      global_steps, adjust_allowed, add_summary=False):
        res_dict = {}
        group_traj_len = S_r_rollout.shape[0]
        id_list = np.arange(0, group_traj_len)
        np.random.shuffle(id_list)
        id_list = np.concatenate([id_list, id_list])
        group_traj_len_sim = S_sim_rollout.shape[0]
        id_list_sim = np.arange(0, group_traj_len_sim)
        np.random.shuffle(id_list_sim)
        id_list_sim = np.concatenate([id_list_sim, id_list_sim])
        if self.minibatch_size == -1:
            minibatch_size = max(group_traj_len, group_traj_len_sim)
        else:
            minibatch_size = self.minibatch_size
        for i in range(int(np.ceil(group_traj_len/minibatch_size))):
            feed_dict = {
                self.S_r_ph: S_r_rollout[id_list[i*minibatch_size:(i+1)*minibatch_size]],
                self.A_r_ph: A_r_rollout[id_list[i*minibatch_size:(i+1)*minibatch_size]],
                self.A_r_first_ph: A_r_first[id_list[i*minibatch_size:(i+1)*minibatch_size]],
                self.S_sim_ph: S_sim_rollout[id_list_sim[i*minibatch_size:(i+1)*minibatch_size]],
                self.A_sim_ph: A_sim_rollout[id_list_sim[i*minibatch_size:(i+1)*minibatch_size]],
                self.A_sim_first_ph: A_sim_first[id_list_sim[i*minibatch_size:(i+1)*minibatch_size]],
                self.global_steps: global_steps,
                self.r2s_prev_hidden_state_ph: prev_hidden_state[id_list[i*minibatch_size:(i+1)*minibatch_size]],
                self.O_r_ph: O_r_rollout[id_list[i*minibatch_size:(i+1)*minibatch_size]],
                self.O_r_first_ph: O_r_first[id_list[i*minibatch_size:(i+1)*minibatch_size]],
                self.adjust_allowed_ph: np.expand_dims(adjust_allowed, axis=0),
            }

            if self.cycle_loss:
                feed_dict.update({
                    self.s2r2s_prev_hidden_state_ph: prev_cycle_hidden_state[
                        id_list_sim[i * minibatch_size:(i + 1) * minibatch_size]],
                })

            ops = [self.hat_S_r_mask, self.hat_S_r_mean_mask, self.hat_S_r_std_mask,
                   self.r2s_hidden_state, self.hat_O_r_mask, self.gen_loss, self.mapping_likelihood,
                   self.mapping_loss, self.gen_learning_rate, self.l2_reg_loss, self.m_grad_norm]

            ops.extend([self.do_nothing_op, self.do_nothing_op, self.do_nothing_op])
            if self.cycle_loss:
                ops.extend([self.state_mapping_likelihood, self.obs_gen_loss])
            else:
                ops.extend([self.do_nothing_op, self.do_nothing_op])
            if add_summary:
                ops.append(self.summary)
            else:
                ops.append(self.do_nothing_op)
            ops.append(self.train_ops)
            res = self.sess.run(ops, feed_dict=feed_dict)[:-1]

            robust_append_dict(res_dict, "gen_loss", res[5])
            robust_append_dict(res_dict, "mapping_likelihood", res[6])
            robust_append_dict(res_dict, "all_loss", res[7])
            robust_append_dict(res_dict, "gen_lr", res[8])
            robust_append_dict(res_dict, "mapping_l2_loss", res[9])
            robust_append_dict(res_dict, "m_grad_norm", res[10])
            if self.cycle_loss:
                robust_append_dict(res_dict, "state_mapping_likelihood", res[14])
                robust_append_dict(res_dict, "obs_gen_loss", res[15])

            robust_append_dict(res_dict, "summary", res[16])
        return res_dict

    def get_batch_zero_state(self):
        if self.batch_zero_state is None:
            self.batch_zero_state = self.sess.run(self.batch_zero_state_op)
        return self.batch_zero_state

    def get_batch_zero_A(self):
        if self.batch_zero_A is None:
            self.batch_zero_A = np.zeros([self.batch_size, self.ac_shape])
        return self.batch_zero_A

    def get_batch_zero_O(self):
        if self.batch_zero_O is None:
            self.batch_zero_O = np.zeros([self.batch_size] + self.img_shape_to_list)
        return self.batch_zero_O
