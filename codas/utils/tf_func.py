import tensorflow as tf

def term_weight_gen(ops, weight):
    assert len(ops.shape) == 3
    # onstant_term_weight = tf.pow(1.0, tf.cast(tf.range(0, self.sequence_length), tf.float32))
    cur_term_weight = weight[:tf.cast(ops.shape[1], tf.int32)]
    return ops * 0 + tf.expand_dims(tf.expand_dims(cur_term_weight, axis=0), axis=2)

def mask_filter(ops, mask):
    return tf.boolean_mask(ops, tf.not_equal(mask, 0.0))