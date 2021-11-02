from codas.rnn_cells.base import CustomMultiCell, CustomDynCell
from codas.utils import tf_util as U
from codas.utils.config import *
from codas.utils.tf_basic import TfBasicClass
from codas.utils.tf_util import one_dim_layer_normalization, shape

from RLA.easy_log.tester import tester

LOG_STD_MAX = 2
LOG_STD_MIN = -20


# for low-d state, no need for unet_structure, batch_norm, etc
# use simple fc layers
class Embedding(TfBasicClass):
    def __init__(self, hidden_dims, output_size,
                 act_fn=None, layer_norm=False,
                 scope='embedding'):
        self.hidden_dims = hidden_dims
        self.layer_norm = layer_norm
        self.scope = scope
        self.act_fn = act_fn
        self.output_size = output_size
        TfBasicClass.__init__(self, scope)

    def _obj_construct(self, input_, *args, **kwargs):
        ndims = input_.get_shape().ndims
        output = input_
        if ndims > 2:
            output = tf.reshape(output, [-1,] + [output.shape[-1]])
        for dim in self.hidden_dims:
            output = tf.layers.dense(output, dim)
            if self.layer_norm:
                output = one_dim_layer_normalization(output)
            output = self.act_fn(output)
        output = tf.layers.dense(output, self.output_size)
        if ndims > 2:
            output = tf.reshape(output, shape(input_)[:2] + [int(output.shape[-1])])
        return output


class RNNPolicy(TfBasicClass):
    def __init__(self, rnn_hidden_dims: [int], rnn_cell, seq_length: int, act_fn, mlp_layer,
                 input_shape, action_shape,output_hidden_dims, layer_norm, scope='rnn_policy', target_mapping=False):
        self.rnn_hidden_dims = rnn_hidden_dims
        self.scope = scope
        self.input_shape = input_shape
        self.output_hidden_dims = output_hidden_dims
        self.target_mapping = target_mapping
        self.rnn_cell = rnn_cell
        self.__RNN_cell_shape = None
        self.zero_state_op = None
        self.action_shape = action_shape
        self.seq_length = seq_length
        self.mlp_layer = mlp_layer
        self.act_fn = act_fn
        self.layer_norm = layer_norm
        TfBasicClass.__init__(self, scope)

    def get_variable_length(self, data):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
            length = tf.reduce_sum(used, reduction_indices=1)
            length = tf.cast(length, tf.int32)
        return length

    @property
    def RNN_cell_shape(self):
        assert self.__RNN_cell_shape is not None
        return self.__RNN_cell_shape

    def __rnn_output_decoder(self, inputs):
        encode_output = inputs
        encode_output = tf.reshape(encode_output, [-1, ] + [encode_output.shape[-1]])
        for dim in self.output_hidden_dims:
            encode_output = tf.layers.dense(encode_output, dim)
            if self.layer_norm:
                encode_output = one_dim_layer_normalization(encode_output)
            encode_output = self.act_fn(encode_output)
        output = encode_output
        if len(shape(inputs)) == 3:
            output = tf.reshape(output, shape(inputs)[:2] + [int(output.shape[-1])])

        ac = tf.layers.dense(output, self.action_shape)
        return ac

    def _obj_construct(self, input_, *args, **kwargs):
        ob_ac_embedding, ac, ob, self.mask, initial_state = input_
        cells = []
        self.var_length = self.get_variable_length(ob)
        if self.mlp_layer is not None:
            assert isinstance(self.mlp_layer, MlpEncoder)
            output = self.mlp_layer.obj_graph_construct(ob_ac_embedding)
            last_hidden_state = self.zero_state(tf.shape(output)[0])
        else:
            for h in self.rnn_hidden_dims:
                cells.append(self.rnn_cell(h))

            cell = CustomMultiCell(cells, state_is_tuple=False, cell_unit_size=self.rnn_hidden_dims)
            output, last_hidden_state = tf.nn.dynamic_rnn(cell, ob_ac_embedding,
                                                          sequence_length=self.var_length, initial_state=initial_state,
                                                          dtype=tf.float32)

            output, all_rnn_state = output
            self.cell = cell
        output = self.__rnn_output_decoder(output)
        return output, last_hidden_state, self.var_length

    # the following are compute-graph-ref based function, should be called after _obj_construct
    # these function return op based on the input (should be placeholder or tensor)
    @property
    def hidden_state_size(self):
        if self.mlp_layer is not None:
            return 1
        else:
            return sum(self.rnn_hidden_dims)

    # the functions postfix with run are runnable functions, which return the results of op.
    def zero_state_run(self, batch_size, sess):
        if self.zero_state_op is None:
            self.batch_size_ph = U.get_placeholder('zero_state_batch_size_ph', shape=(), dtype=tf.int32)
            self.zero_state_op = self.zero_state(batch_size=self.batch_size_ph)
        return sess.run(self.zero_state_op, feed_dict={self.batch_size_ph: batch_size})

    def zero_state(self, batch_size):
        if self.mlp_layer is not None:
            return tf.zeros(shape=(batch_size, 1))
        else:
            return self.cell.zero_state(batch_size=batch_size, dtype=tf.float32)

class MlpEncoder(TfBasicClass):
    def __init__(self, hidden_dims, act_fn, scope):
        self.hidden_dims = hidden_dims
        self.act_fn = act_fn
        TfBasicClass.__init__(self, scope)

    def _obj_construct(self, source_input, *args, **kwargs):
        ob_ac_embedding = source_input
        encode_ouput = ob_ac_embedding
        for dim in self.hidden_dims:
            encode_ouput = tf.layers.dense(encode_ouput, dim)
            encode_ouput = one_dim_layer_normalization(encode_ouput)
            encode_ouput = self.act_fn(encode_ouput)
        return encode_ouput


class Real2Sim(TfBasicClass):
    def __init__(self, rnn_hidden_dims: [int], rnn_cell, seq_length: int, act_fn,
                 ob_shape, action_shape, mlp_layer,
                 output_hidden_dims, layer_norm, emb_dynamic,
                 transition: TfBasicClass, transition_decoder:TfBasicClass,
                 scope='real2sim_mapping', target_mapping=False, num_gan_step = 1):
        self.rnn_hidden_dims = rnn_hidden_dims
        self.scope = scope
        self.ob_shape = ob_shape
        self.emb_dynamic = emb_dynamic
        self.output_hidden_dims = output_hidden_dims
        self.target_mapping = target_mapping
        self.rnn_cell = rnn_cell
        self.__RNN_cell_shape = None
        self.zero_state_op = None
        self.action_shape = action_shape
        self.seq_length = seq_length
        self.act_fn = act_fn
        self.layer_norm = layer_norm
        self.mlp_layer = mlp_layer
        self.transition = transition
        self.transition_decoder = transition_decoder
        self.num_gan_step = num_gan_step
        TfBasicClass.__init__(self, scope)


    @property
    def RNN_cell_shape(self):
        assert self.__RNN_cell_shape is not None
        return self.__RNN_cell_shape

    def __rnn_output_decoder(self, inputs):
        if self.emb_dynamic:
            mu, std, hidden_state = inputs
        else:
            encode_ouput, hidden_state = inputs
            # encode_ouput = tf.reshape(encode_ouput, [-1, ] + [encode_ouput.shape[-1]])
            for dim in self.output_hidden_dims:
                encode_ouput = tf.layers.dense(encode_ouput, dim)
                if self.layer_norm:
                    encode_ouput = one_dim_layer_normalization(encode_ouput)
                encode_ouput = tf.nn.tanh(encode_ouput)
            output = encode_ouput
            # output = tf.layers.dense(output, self.rnn_hidden_dims[-1], activation=tf.nn.tanh)
            mu = tf.layers.dense(output, self.ob_shape)
            log_std = tf.layers.dense(output, self.ob_shape)
            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = (tf.exp(log_std) + 1e-6)
        return tf.contrib.distributions.Normal(loc=mu, scale=std), hidden_state
        # return tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=std), hidden_state

    def _obj_construct(self, input_, *args, **kwargs):
        ob_ac_embedding, ac, self.var_length, initial_state, adjust_allowed = input_
        cells = []
        for h in self.rnn_hidden_dims:
            cells.append(self.rnn_cell(h))
        if self.mlp_layer is not None:
            assert isinstance(self.mlp_layer, MlpEncoder)
            output = self.mlp_layer.obj_graph_construct(ob_ac_embedding)
            last_hidden_state = self.zero_state(tf.shape(output)[0])
            all_hidden_state = tf.zeros(shape=tf.concat([tf.shape(output)[0:2], [self.ob_shape]], axis=0))
            output = (output, all_hidden_state)
        else:
            if self.emb_dynamic:
                cell = CustomDynCell(cells, state_is_tuple=False, cell_unit_size=self.rnn_hidden_dims,
                                     transition=self.transition, transition_decoder=self.transition_decoder,
                                     ob_shape=self.ob_shape)
                # adjust_allowed should be padded into a sequence for rnn rollout.
                adjust_allowed = tf.expand_dims(tf.ones(tf.shape(ob_ac_embedding)[:-1]), axis=-1) * adjust_allowed
                output, last_hidden_state = tf.nn.dynamic_rnn(cell, (ob_ac_embedding, ac, adjust_allowed),
                                                              sequence_length=self.var_length, initial_state=initial_state,
                                                              dtype=tf.float32)
            else:
                cell = CustomMultiCell(cells, state_is_tuple=False, cell_unit_size=self.rnn_hidden_dims)
                output, last_hidden_state = tf.nn.dynamic_rnn(cell, ob_ac_embedding, sequence_length=self.var_length,
                                                              initial_state=initial_state, dtype=tf.float32)
            self.cell = cell
        output, all_hidden_state = self.__rnn_output_decoder(output)
        return output, all_hidden_state, last_hidden_state

    # the following are compute-graph-ref based function, should be called after _obj_construct
    # these function return op based on the input (should be placeholder or tensor)
    @property
    def hidden_state_size(self):
        if self.mlp_layer is not None:
            return self.ob_shape
        else:
            if self.emb_dynamic:
                return sum(self.rnn_hidden_dims) + self.ob_shape
            else:
                return sum(self.rnn_hidden_dims)

    # the functions postfix with run are runnable functions, which return the results of op.
    def zero_state_run(self, batch_size, sess):
        if self.zero_state_op is None:
            self.batch_size_ph = U.get_placeholder('zero_state_batch_size_ph', shape=(), dtype=tf.int32)
            self.zero_state_op = self.zero_state(batch_size=self.batch_size_ph)
        return sess.run(self.zero_state_op, feed_dict={self.batch_size_ph: batch_size})

    def zero_state(self, batch_size):
        if self.mlp_layer is not None:
            return tf.zeros(shape=(batch_size, self.ob_shape))
        else:
            return self.cell.zero_state(batch_size=batch_size, dtype=tf.float32)


class Sim2Real(TfBasicClass):
    def __init__(self, hidden_dims, rnn_hidden_dims, emb_dim,
                 ob_shape, ac_shape, act_fn=None, rnn_cell=None,
                 layer_norm=False, real_ob_input=False, encoder=None,
                 scope='sim2real_mapping'):
        self.hidden_dims = hidden_dims
        self.rnn_hidden_dims = rnn_hidden_dims
        self.emb_dim = emb_dim
        self.scope = scope
        self.act_fn = act_fn
        self.layer_norm = layer_norm
        self.rnn_cell = rnn_cell
        self.real_ob_input = real_ob_input
        self.ob_shape = ob_shape
        self.ac_shape = ac_shape
        self.encoder = encoder
        TfBasicClass.__init__(self, scope)

    def get_variable_length(self, data):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2)) # 第3维度是数据特征，如果有非0的数据，就会表示为1
            length = tf.reduce_sum(used, reduction_indices=1) # 第2维度，下标1，表示的是时间步，当数据为1的时候，说明该维度的时间步是正确的
            length = tf.cast(length, tf.int32) # 转化为整型
        return length

    def _obj_construct(self, input_, *args, **kwargs):
        ob_real2sim, ob_real, ac = input_
        if not self.real_ob_input:
            ob_real = ob_real2sim
        if tester.hyper_param['res_struc'] == ResnetStructure.EMBEDDING_RAS:
            ob_real2sim = tf.concat([ob_real2sim, ac], axis=-1)
        elif tester.hyper_param["res_struc"] == ResnetStructure.EMBEDDING_RS:
            ob_real2sim = ob_real2sim
        else:
            raise NotImplementedError
        real2sim_embedding = tf.layers.dense(ob_real2sim, int(self.emb_dim / 2), activation=self.act_fn)
        real_embedding = tf.layers.dense(ob_real, int(self.emb_dim / 2), activation=self.act_fn)
        embedding_var = input = real2sim_embedding + real_embedding
        output = tf.layers.dense(embedding_var, self.emb_dim, activation=self.act_fn)
        if self.rnn_hidden_dims:
            assert self.emb_dim == self.rnn_hidden_dims[0]
            cells = []
            for h in self.rnn_hidden_dims:
                cells.append(self.rnn_cell(h))
            cell = CustomMultiCell(cells, state_is_tuple=False, cell_unit_size=self.rnn_hidden_dims)
            self.cell = cell
            self.var_length = self.get_variable_length(ac)
            output, last_state = tf.nn.dynamic_rnn(cell, output,
                                                   sequence_length=self.var_length,
                                                   dtype=tf.float32)
            output, all_rnn_state = output
        for dim in self.hidden_dims:
            output = tf.layers.dense(output, dim)
            if self.layer_norm:
                output = one_dim_layer_normalization(output)
            output = self.act_fn(output)
        output = tf.layers.dense(output, self.emb_dim)
        output = tf.reshape(output, shape(input)[:-1] + [int(output.shape[-1])])
        # TODO: hyper_param
        return output

