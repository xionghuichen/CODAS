import tensorflow as tf


class BasicConfig(object):
    def __init__(self):
        pass

    @classmethod
    def config_check(cls, var):
        assert var in vars(cls).values(), "var %s invalid in cls %s" % (var, type(cls))

class RandomSeqType(BasicConfig):
    BIAS = 'bias'
    SCALE = 'scale'
    RNN = 'rnn'
class BufferMode(BasicConfig):
    SINGLE_BUFFER = 'single_buffer'
    DOUBLE_BUFFER = 'double_buffer'


class PolicyMode(BasicConfig):
    RESIDUAL = 'residual'
    RESIDUAL_STO = 'residual_sto'
    FAKE = 'fake'
    DIRECT ='direct'


class GanLoss(BasicConfig):
    MINIMAX = 'minimax'
    LSGAN = 'lsgan'
    LSGAN_LOGIT = 'lsgan_logit'
    WGAN = 'wgan'

class Trajectory(BasicConfig):
    OB = 'ob'
    OB1 = 'ob1'
    AC = 'ac'
    REWARD = 'reward'
    RETURN = 'return'
    LENGTH = 'length'
    DONE = 'done'


class MappingDirecition(BasicConfig):
    SRS = 'srs'
    RSR = 'rsr'
    S2R = 's2r'
    R2S = 'r2s'

class LossType(BasicConfig):
    VAE = 'vae'
    GAN = 'gan'

class ResnetStructure(BasicConfig):
    EMBEDDING_RS = '{r}{s}'
    EMBEDDING_RAS = '{r}{as}'
    RAS_ADD_EMBEDDING = 'r{as}'
    RAS_EMBEDDING = '{ras}'
    RS_EMBEDDING = '{rs}'



class ActivateFn(BasicConfig):
    LEAKLYRELU = 'leakyrelu'
    TANH = 'tanh'
    RELU = 'relu'
    ID = 'id'

    @classmethod
    def obj_convect(cls, var):
        if var == cls.LEAKLYRELU:
            return tf.nn.leaky_relu
        elif var == cls.TANH:
            return tf.nn.tanh
        elif var == cls.RELU:
            return tf.nn.relu
        elif var == cls.ID:
            return tf.identity
        else:
            raise NotImplementedError


class LabelMaskType(BasicConfig):
    RANDOM = 'ran'
    FIRST_LAST = 'fir_las'


class AEType(BasicConfig):
    STABLE = 'sta'


class NoiseType(BasicConfig):
    NORMAL = 'normal'

class RewardMode(BasicConfig):
    TARGET ='target'
    CURRENT = 'current'

class SeqGanLossMode(BasicConfig):
    PGLOSS = 'pg'
    MESLOSS = 'mse'
    TRANSLOSS = 'trans'

class RNNCell(BasicConfig):
    RNN = 'rnn'
    GRU = 'gru'
    LSTM = 'lstm'
    LAYER_NORM_LSTM = 'norm_lstm'

    @classmethod
    def obj_convect(cls, var):
        if var == cls.RNN:
            return tf.contrib.rnn.BasicRNNCell
        elif var == cls.GRU:
            return tf.contrib.rnn.GRUBlockCell
        elif var == cls.LSTM:
            return tf.contrib.rnn.LSTMCell
        elif var == cls.LAYER_NORM_LSTM:
            # TODO how to do multi layer
            return tf.contrib.rnn.LayerNormBasicLSTMCell # lambda x: (x, state_is_tuple=False)
        else:
            raise NotImplementedError


class RolloutMode(BasicConfig):
    ORACLE_ROLLOUT_FAKE_GEN = 'oracle_rollout_fake_gen'
    ORACLE_ROLLOUT_ONE_STEP_GEN = 'oracle_rollout_one_gen'
    ORACLE_ROLLOUT_MULTI_STEP_GEN = 'oracle_rollout_multi_gen'
    REAL_ROLLOUT = 'real_rollout'
    REAL_ROLLOUT_MULTI_STEP = 'real_rollout_multi_step'


class DiscriminatorStructure(BasicConfig):
    OB_AC_CONCATE = '{oa}'
    OB = 'o'

class Transformation(BasicConfig):
    NOISE = 'noise'
    RANDNET = 'rand'
    IDENTITY = 'id'
    SHIFT = 'shift'
    LINEAR = 'linear'
    RANDSEQNET = 'randseq'
    IMG = 'img'

class AlgType(BasicConfig):
    CODAS = 'codas'
    MLP_TRAJ_DIS = 'mlp_traj_dis'
    MLP = 'mlp'
    NO_DYN = 'no_dyn'
    NO_DYN_NO_TRAJ_DIS = 'no_dyn_no_traj_dis'
    NO_TRAJ_DIS = 'no_traj_dis'
    VAN_GAN = 'van_gan'
    CYCLE_GAN = 'cycle_gan'
    VAN_GAN_RNN = 'van_gan_rnn'
    VAN_GAN_STACK = 'van_gan_stack'
    MULTISTEP_GAN = 'multistep_gan'

class ImgShape(object):
    WIDTH = 'width'
    HEIGHT = 'height'
    CHANNEL = 'channel'

if __name__ == '__main__':
    MappingDirecition.config_check('s2r')
