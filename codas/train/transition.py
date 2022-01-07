
from codas.utils.tf_basic import TfBasicClass
import tensorflow as tf
from codas.utils import tf_util as U
from codas.utils.tf_util import one_dim_layer_normalization
import numpy as np
from RLA.easy_log import logger

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def soft_clip(input, min_v, max_v):
    output = max_v - tf.nn.softplus(max_v - input)
    output = min_v + tf.nn.softplus(output - min_v)
    # output = tf.clip_by_value(input, min_v, max_v)
    return output


class Transition(TfBasicClass):
    def __init__(self, transition_hidden_dims, transition_trainable,
                 obs_min, obs_max, ob_shape, act_fn=tf.nn.tanh, scope='transition'):
        self.act_fn = act_fn
        self.transition_hidden_dims = transition_hidden_dims
        self.transition_trainable = transition_trainable
        self.ob_shape = ob_shape
        self.obs_min = obs_min
        self.obs_max = obs_max

        TfBasicClass.__init__(self, scope)

    def _obj_construct(self, transition_input, *args, **kwargs):
        ob_sim, ac = tuple(transition_input)
        norm_obs_sim = ob_sim
        if not self.transition_trainable:
            norm_obs_sim = tf.clip_by_value(norm_obs_sim, self.obs_min, self.obs_max)

        transition_input = tf.concat([norm_obs_sim, ac], axis=-1)
        next_ob_sim = tf.layers.dense(transition_input, self.transition_hidden_dims[0], activation=self.act_fn,
                                      trainable=self.transition_trainable)

        for hidden_dim in self.transition_hidden_dims[1:]:
            next_ob_sim = tf.layers.dense(next_ob_sim, hidden_dim, activation=self.act_fn,
                                          trainable=self.transition_trainable)

        next_ob_sim = tf.layers.dense(next_ob_sim, self.ob_shape, trainable=self.transition_trainable)
        next_ob_sim = next_ob_sim
        # if not self.transition_trainable:
        #     next_ob_sim = tf.clip_by_value(next_ob_sim, self.obs_min, self.obs_max)
        next_ob_sim = (tf.nn.tanh(next_ob_sim) + 1.001) / 2.0 * (self.obs_max - self.obs_min) + self.obs_min
        # next_ob_sim = soft_clip(next_ob_sim, self.obs_min, self.obs_max)
        # next_ob_sim = tf.nn.tanh(next_ob_sim) * 20
        return next_ob_sim

    def pretrained_value_assignment(self):
        pretrained_vars = self.global_variables()
        assignment_ops = []
        phs = []
        for var in pretrained_vars:
            ph = tf.placeholder(shape=var.shape, dtype=tf.float32,
                                name=var.name.replace('/', '-').replace(':', '_') + '-ph')
            op = var.assign(ph)
            phs.append(ph)
            assignment_ops.append(op)
        return assignment_ops, phs


class TransitionDecoder(TfBasicClass):
    def __init__(self, ob_shape, hidden_dims, obs_min, obs_max, scope='transition_decoder'):
        self.ob_shape = ob_shape
        self.obs_min  = obs_min
        self.obs_max = obs_max

        self.hidden_dims = hidden_dims
        TfBasicClass.__init__(self, scope)

    def _obj_construct(self, source_input, *args, **kwargs):
        rnn_output, transition_predict, ob_real_emb, ac, adjust_allowed = tuple(source_input)
        # transition_predict = tf.stop_gradient(transition_predict)
        # mu = tf.layers.dense(output, self.output_size)
        # log_std = tf.layers.dense(output, self.output_size)
        # log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX) # * ((1 - self.mask) * 1e-10)
        # std = (tf.exp(log_std) + 1e-6)
        decode = tf.concat([rnn_output, transition_predict,  ob_real_emb, ac], axis=-1)
        for hidden_dim in self.hidden_dims:
            decode = tf.layers.dense(decode, hidden_dim)
            decode = one_dim_layer_normalization(decode)
            decode = tf.nn.tanh(decode)
        # infer by rnn
        # mu = tf.layers.dense(decode, self.ob_shape, name='mu')
        # decode = tf.concat([rnn_output, transition_predict], axis=-1)
        # transition_code = tf.layers.dense(decode, rnn_output.shape[-1], tf.nn.tanh)
        # delta_weight = tf.layers.dense(transition_code, self.ob_shape)

        # # learn to weight
        # # transition_code = transition_predict * transition_code
        # # decode = tf.concat([rnn_output, transition_code], axis=-1)
        # mu = tf.layers.dense(decode, self.ob_shape, name='mu')
        # mu = mu * delta_weight + transition_predict

        # concat

        # transition_code = tf.layers.dense(decode, rnn_output.shape[-1], tf.nn.tanh)
        # transition_code = tf.layers.dense(transition_code, rnn_output.shape[-1], tf.nn.tanh)
        # decode = tf.concat([rnn_output, transition_code], axis=-1)
        # mu = tf.layers.dense(decode, self.ob_shape, name='mu')

        # add
        mu = tf.layers.dense(decode, self.ob_shape, name='mu')
        mu = soft_clip(mu, self.obs_min, self.obs_max)
        # mu = tf.nn.tanh(mu)
        # mu = mu * adjust_allowed + transition_predict

        log_std = tf.layers.dense(decode, self.ob_shape, name='logstd')
        log_std = soft_clip(log_std, LOG_STD_MIN, LOG_STD_MAX)  # * ((1 - self.mask) * 1e-10)
        std = (tf.exp(log_std) + 1e-6)
        self.std = std
        return mu, std


class TransitionLearner(object):

    def __init__(self, transition:Transition, transition_target: Transition, ob_shape, ac_shape, batch_size, lr, l2_loss, sess, scope='transition_learner'):
        self.transition = transition
        self.transition_target = transition_target
        self.lr = lr
        self.sess = sess
        self.ob_shape = ob_shape
        self.scope = scope
        self.l2_loss = l2_loss
        self.batch_size = batch_size
        self.ac_shape = ac_shape

    def model_setup(self):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # s, a ph.
            self.acs_ph = U.get_placeholder(name='acs', dtype=tf.float32, shape=[None, self.ac_shape])
            self.obs_ph = U.get_placeholder(name='obs', dtype=tf.float32, shape=[None, self.ob_shape])
            self.next_obs_ph = U.get_placeholder(name='next_obs', dtype=tf.float32, shape=[None, self.ob_shape])
            self.lr_ph = U.get_placeholder(name='dyn_lr', dtype=tf.float32, shape=[])
            self.nex_obs_pred = self.transition.obj_graph_construct([self.obs_ph, self.acs_ph])
            self.nex_obs_pred_target = self.transition_target.obj_graph_construct([self.obs_ph, self.acs_ph])
            # mse loss.
            self.mse_loss = tf.losses.mean_squared_error(labels=self.next_obs_ph, predictions=self.nex_obs_pred)
            self.max_error = tf.reduce_max(tf.square(self.next_obs_ph - self.nex_obs_pred))
            vars = self.transition.trainable_variables()
            self.l2_reg = tf.add_n([tf.nn.l2_loss(var) for var in vars]) / len(vars)
            loss_with_reg = self.mse_loss + self.l2_reg * self.l2_loss
            # opt.
            self.optim = tf.train.AdamOptimizer(self.lr_ph).minimize(loss_with_reg,)
            # copy param to target.
            target_vars = self.transition_target.global_variables()
            vars = self.transition.trainable_variables()
            assigns = []
            for target_var, var in zip(target_vars, vars):
                assigns.append(tf.assign(target_var, var))
            self.assign_op = assigns  # tf.group(assigns)
            self.trans_init_ops, self.trans_init_phs = self.transition.pretrained_value_assignment()
            self.target_trans_init_ops, self.target_trans_init_phs = self.transition_target.pretrained_value_assignment()

    def update_transition(self, obs, acs, next_obs, lr=None):
        idx = np.arange(obs.shape[0])
        np.random.shuffle(idx)
        start_id = 0
        mse_loss_list = []
        l2_loss_list = []
        max_error_list = []
        idx = np.concatenate([idx, idx])
        if lr is None:
            lr = self.lr
        assert self.batch_size <= obs.shape[0], f"bathsize {self.batch_size}, obs.shape {obs.shape[0]}"
        while start_id + self.batch_size <= obs.shape[0]:
            sample_obs = obs[idx[start_id:start_id + self.batch_size]]
            sample_acs = acs[idx[start_id:start_id + self.batch_size]]
            sample_next_obs = next_obs[idx[start_id:start_id + self.batch_size]]
            start_id += self.batch_size
            feed_dict = {
                self.obs_ph : sample_obs,
                self.acs_ph: sample_acs,
                self.next_obs_ph: sample_next_obs,
                self.lr_ph: lr,
            }
            mse_loss, max_error, l2_reg = self.sess.run([self.mse_loss, self.max_error, self.l2_reg, self.optim], feed_dict=feed_dict)[:-1]
            mse_loss_list.append(mse_loss)
            max_error_list.append(max_error)
            l2_loss_list.append(l2_reg)
        return np.mean(mse_loss_list), np.mean(max_error_list), np.mean(l2_loss_list)

    # def update_transition(self, obs, acs, next_obs):
    #     feed_dict = {
    #         self.obs_ph : obs,
    #         self.acs_ph: acs,
    #         self.next_obs_ph: next_obs
    #         }
    #     return self.sess.run([self.mse_loss, self.max_error, self.optim], feed_dict=feed_dict)[:-1]

    def pred_consistent_test(self, obs, acs):
        feed_dict = {
            self.obs_ph : obs,
            self.acs_ph: acs,
        }
        pred, pred_target = self.sess.run([self.nex_obs_pred, self.nex_obs_pred_target], feed_dict=feed_dict)
        assert np.allclose(pred, pred_target)

    def pred(self, obs, acs):
        feed_dict = {
            self.obs_ph : obs,
            self.acs_ph: acs,
        }
        return self.sess.run(self.nex_obs_pred_target, feed_dict=feed_dict)


    def copy_params_to_target(self):
        self.sess.run(self.assign_op)

    def pretrained_value_assignment(self, *weights):
        feed_dict = {}
        for target_ph, ph, w in zip(self.target_trans_init_phs, self.trans_init_phs, weights):
            feed_dict[ph] = w
            feed_dict[target_ph] = w
        self.sess.run([self.trans_init_ops, self.target_trans_init_ops], feed_dict=feed_dict)





