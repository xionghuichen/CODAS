from codas.utils.tf_basic import TfBasicClass
import tensorflow as tf
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Policy(TfBasicClass):
    def __init__(self, policy_trainable, ac_shape, sess, policy_hidden_dims=[64, 64], act_fn=tf.nn.tanh,  scope='policy'):
        self.act_fn = act_fn
        self.policy_hidden_dims = policy_hidden_dims
        self.sess = sess
        self.policy_trainable = policy_trainable
        self.ac_shape = ac_shape
        self.policy_input = None
        TfBasicClass.__init__(self, scope)

    def _obj_construct(self, policy_input, *args, **kwargs):
        self.policy_input = policy_input
        next_ob_sim = tf.layers.dense(policy_input, self.policy_hidden_dims[0], activation=self.act_fn,
                                      trainable=self.policy_trainable)
        for hidden_dim in self.policy_hidden_dims[1:]:
            next_ob_sim = tf.layers.dense(next_ob_sim, hidden_dim, activation=self.act_fn,
                                          trainable=self.policy_trainable)
        mean = tf.layers.dense(next_ob_sim, self.ac_shape, trainable=self.policy_trainable)
        logstd = tf.get_variable(name='logstd', shape=[1, self.ac_shape], initializer=tf.zeros_initializer())
        self.mean = mean
        self.logstd = logstd
        return mean, logstd

    def pretrained_value_assignment(self):
        pretrained_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._global_scope.name)
        assignment_ops = []
        phs = []
        for var in pretrained_vars:
            ph = tf.placeholder(shape=var.shape, dtype=tf.float32,
                                name=var.name.replace('/', '-').replace(':', '_') + '-ph')
            op = var.assign(ph)
            phs.append(ph)
            assignment_ops.append(op)
        return assignment_ops, phs

    def predict_run(self, policy_input, sess, deterministic=False):
        assert deterministic
        return self.sess.run(self.mean, feed_dict={self.policy_input: policy_input})

#
# if __name__ == '__main__':
#     data, params = PPO2._load_from_file('../../data/saved_model/trpo_Hopper-v2_2000000.zip')
#     model = Policy(True, 3)
#     model.obj_graph_construct(tf.placeholder(tf.float32, shape=(1, 11)))
#     for key in params.keys():
#         if key.startswith('model/pi'):
#             print(key, params.get(key))
#     print()




