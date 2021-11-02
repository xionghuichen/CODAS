
from SRG.utils.tf_basic import TfBasicClass
from RLA.easy_log.tester import tester

from codas.utils.tf_util import one_dim_layer_normalization
import numpy as np
import tensorflow as tf

from codas.utils.tf_basic import TfBasicClass
from codas.utils.config import *
from codas.utils import tf_util

class BaseDiscriminator(TfBasicClass):
    def __init__(self, hid_dims, discre_struc, output_size, scope):
        self.hid_dims = hid_dims
        self.discre_struc = discre_struc

        self.output_size = output_size
        TfBasicClass.__init__(self, scope)

    def _obj_construct(self, source_input, *args, **kwargs):
        pass


from codas.rnn_cells.base import CustomMultiCell, CustomDynCell

class TrajDiscriminator(BaseDiscriminator):
    def __init__(self, hid_dims, emb_hid_dim, input_type, output_size, layer_norm,
                 scope, rnn_hidden_dims, rnn_cell, act_fn=tf.nn.relu):
        BaseDiscriminator.__init__(self, hid_dims, input_type, output_size, scope)

        self.emb_hid_dim = emb_hid_dim
        self.layer_norm = layer_norm
        self.act_fn = act_fn
        self.rnn_cell = rnn_cell
        self.rnn_hidden_dims = rnn_hidden_dims

    @property
    def hidden_state_size(self):
        return sum(self.rnn_hidden_dims)

    def _obj_construct(self, source_input, *args, **kwargs):
        if self.discre_struc == DiscriminatorStructure.OB_AC_CONCATE:
            ob, ac = source_input
            self.var_length = self.get_variable_length(ob)
            _input = tf.concat([ob, ac], axis=-1)
            x = tf.layers.dense(_input, self.emb_hid_dim, activation=self.act_fn)
        else:
            raise NotImplementedError


        for dim in self.hid_dims:
            x = tf.layers.dense(x, dim)
            if self.layer_norm:
                x = one_dim_layer_normalization(x)
            x = self.act_fn(x)
        cells = []
        for h in self.rnn_hidden_dims:
            cells.append(self.rnn_cell(h))
        cell = CustomMultiCell(cells, state_is_tuple=False, cell_unit_size=self.rnn_hidden_dims)
        # the rnn is initialized with zero vector
        # (a trick to reset the hidden states of the traj-discriminator for robust learning)
        output, last_hidden_state = tf.nn.dynamic_rnn(cell, x, sequence_length=self.var_length, dtype=tf.float32)
        output = output[0]
        p_h3 = tf.layers.dense(output, self.output_size, activation=tf.identity)
        if tester.hyper_param['gan_loss'] == GanLoss.MINIMAX:
            assert self.output_size == 1
        return p_h3

    def get_structure(self):
        return self.discre_struc

    def get_variable_length(self, data):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
            length = tf.reduce_sum(used, reduction_indices=1)
            length = tf.cast(length, tf.int32)
        return length


class StateDistributionDiscriminator(BaseDiscriminator):
    def __init__(self, hid_dims, emb_hid_dim, input_type, output_size, layer_norm, scope, act_fn=tf.nn.relu):
        BaseDiscriminator.__init__(self, hid_dims, input_type, output_size, scope)

        self.emb_hid_dim = emb_hid_dim
        self.layer_norm = layer_norm
        self.act_fn = act_fn

    def _obj_construct(self, source_input, *args, **kwargs):
        if self.discre_struc == DiscriminatorStructure.OB:
            _input = source_input
            x = tf.layers.dense(_input, self.emb_hid_dim, activation=self.act_fn)
        elif self.discre_struc == DiscriminatorStructure.OB_AC_CONCATE:
            ob, ac = source_input
            _input = tf.concat([ob, ac], axis=-1)
            x = tf.layers.dense(_input, self.emb_hid_dim, activation=self.act_fn)
        else:
            raise NotImplementedError
        for dim in self.hid_dims:
            # dx = tf.layers.dense(x, dim)
            # if self.layer_norm:
            #     dx = one_dim_layer_normalization(dx)
            # dx = self.act_fn(dx)
            # x = x + dx
            x = tf.layers.dense(x, dim)
            if self.layer_norm:
                x = one_dim_layer_normalization(x)
            x = self.act_fn(x)
        # x = tf.layers.dense(x, self.emb_hid_dim, activation=tf.nn.tanh)
        p_h3 = tf.layers.dense(x, self.output_size, activation=tf.identity)

        if tester.hyper_param['gan_loss'] == GanLoss.MINIMAX:
            assert self.output_size == 1
            # discriminator = tf.sigmoid(p_h3)
        # elif tester.hyper_param['gan_loss'] == GanLoss.LSGAN_LOGIT:
        #     discriminator = p_h3
        # else:
        #     raise NotImplementedError
        return p_h3

    def get_structure(self):
        return self.discre_struc


class ImgDiscriminator(StateDistributionDiscriminator):
    def __init__(self, *args, **kwargs):
        StateDistributionDiscriminator.__init__(self, *args, **kwargs)

    def _obj_construct(self, source_input, *args, **kwargs):
        ac = None
        if self.discre_struc == DiscriminatorStructure.OB:
            imgs = source_input
        elif self.discre_struc == DiscriminatorStructure.OB_AC_CONCATE:
            imgs, ac = source_input
        else:
            raise NotImplementedError
        kwargs = dict(strides=2, activation=tf.nn.relu)
        ndims = imgs.get_shape().ndims
        if ndims == 5:
            hidden = tf.reshape(imgs, [-1] + imgs.shape[2:].as_list())
        elif ndims == 4:
            hidden = imgs
        else:
            raise NotImplemented
        hidden = tf.layers.conv2d(hidden, 32, 4, name='enc_conv1', **kwargs)
        hidden = tf.layers.conv2d(hidden, 64, 4, name='enc_conv2', **kwargs)
        hidden = tf.layers.conv2d(hidden, 128, 4, name='enc_conv3', **kwargs)
        hidden = tf.layers.conv2d(hidden, 256, 4, name='enc_conv4', **kwargs)
        hidden = tf.layers.flatten(hidden)
        assert hidden.shape[1:].as_list() == [1024], hidden.shape.as_list()
        # hidden = tf.layers.dense(hidden, 128, None, name='enc_fc5')
        if ndims == 5:
            hidden = tf.reshape(hidden, tf_util.shape(imgs)[:2] + [
                np.prod(hidden.shape[1:].as_list())])
        imgs_emb = hidden
        x = tf.layers.dense(tf.concat([imgs_emb, ac], axis=-1), 256, activation=tf.nn.tanh)
        p_h3 = tf.layers.dense(x, self.output_size, activation=tf.identity)

        if tester.hyper_param['gan_loss'] == GanLoss.MINIMAX:
            assert self.output_size == 1
            # discriminator = tf.sigmoid(p_h3)
        # elif tester.hyper_param['gan_loss'] == GanLoss.LSGAN_LOGIT:
        #     discriminator = p_h3
        # else:
        #     raise NotImplementedError
        return p_h3

    def get_structure(self):
        return self.discre_struc