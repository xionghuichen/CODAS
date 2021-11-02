import tensorflow as tf
from codas.utils.tf_basic import TfBasicClass


class CustomMultiCell(tf.nn.rnn_cell.MultiRNNCell):
    def __init__(self, cells, cell_unit_size, mask_state=False, state_is_tuple=True, ):
        self.cell_unit_size = cell_unit_size
        self.mask_state = mask_state
        super(CustomMultiCell, self).__init__(cells, state_is_tuple=state_is_tuple)

    def call(self, input, pre_state):
        output, next_state = super(CustomMultiCell, self).call(input, pre_state)
        if self.mask_state:
            next_state = next_state * 0
        return (output, next_state), next_state

    @property
    def state_size(self):
        return sum(self.cell_unit_size)

    @property
    def output_size(self):
        return (self.cell_unit_size[-1], self.state_size)

    def zero_state(self, batch_size, dtype):
        return tf.zeros([batch_size, self.state_size], dtype)


class CustomDynCell(tf.nn.rnn_cell.MultiRNNCell):
    def __init__(self, cells, cell_unit_size, transition: TfBasicClass,
                 transition_decoder: TfBasicClass, ob_shape, state_is_tuple=True):
        self.cell_unit_size = cell_unit_size
        # self.encoding_hidden_dims = encoding_hidden_dims
        self.transition = transition
        self.transition_decoder = transition_decoder
        # self.decoding_hidden_dims = decoding_hidden_dims
        self.act_fn = tf.nn.tanh
        self.ob_shape = ob_shape
        super(CustomDynCell, self).__init__(cells, state_is_tuple=state_is_tuple)


    def call(self, inputs, pre_state):
        ob_real_emb, ac, adjust_allowed = inputs
        state = pre_state[..., :-self.ob_shape]
        ob_sim = pre_state[..., -self.ob_shape:]
        next_ob_sim = self.transition.obj_graph_construct([ob_sim, ac])
        full_ob = tf.concat([ob_real_emb, next_ob_sim, ac], axis=-1)

        pre_state = state
        output, next_rnn_state = super(CustomDynCell, self).call(full_ob, pre_state)
        print("output cell name {}".format(output.name))
        # for hidden_dim in self.decoding_hidden_dims:
        #     output = tf.layers.dense(output, hidden_dim, activation=self.act_fn)
        mu, std = self.transition_decoder.obj_graph_construct((output, next_ob_sim, ob_real_emb, ac, adjust_allowed))
        next_ob_sim_distribution = tf.contrib.distributions.Normal(loc=mu, scale=std)
        # next_ob_sim_distribution = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=std)
        next_state = tf.concat([next_rnn_state, next_ob_sim_distribution.sample()], axis=-1)
        return (mu, std, next_state), next_state

    @property
    def state_size(self):
        return sum(self.cell_unit_size) + self.ob_shape

    @property
    def output_size(self):
        return (self.ob_shape, self.ob_shape, self.state_size)

    def zero_state(self, batch_size, dtype):
        return tf.zeros([batch_size, self.state_size], dtype)
