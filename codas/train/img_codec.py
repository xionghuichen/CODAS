# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
from codas.utils.tf_basic import TfBasicClass
from codas.utils import tf_util


class Encoder(TfBasicClass):
    def __init__(self, scope='encoder', stack_imgs=0):
        TfBasicClass.__init__(self, scope, )
        self.stack_imgs = stack_imgs

    def _obj_construct(self, imgs, *args, **kwargs):
        """Extract deterministic features from an observation."""
        kwargs = dict(strides=2, activation=tf.nn.relu)
        ndims = imgs.get_shape().ndims
        if ndims == 5: # [batch, horizon, h, w, c]
            if imgs.shape[1] == 1 or self.stack_imgs == 1:
                stack_imgs = imgs
            else:
                # padding zeros
                def stack_idx(idx):
                    pre_pad_img = tf.zeros([tf.shape(imgs)[0], idx] + imgs.shape[2:].as_list(), dtype=imgs.dtype)
                    post_pad_img = tf.zeros([tf.shape(imgs)[0], self.stack_imgs - 1 - idx] + imgs.shape[2:].as_list(), dtype=imgs.dtype)
                    stacked_imgs = tf.concat([pre_pad_img, imgs, post_pad_img], axis=1)
                    return stacked_imgs
                idx_list = tuple(list(range(self.stack_imgs)))
                st_imgs = list(map(stack_idx, idx_list))
                stack_imgs = tf.concat(st_imgs, axis=-1)[:, :-1 * (self.stack_imgs - 1)]
            hidden = tf.reshape(stack_imgs, [-1] + stack_imgs.shape[2:].as_list())
        elif ndims == 4:
            stack_imgs = imgs
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
            hidden = tf.reshape(hidden, tf_util.shape(stack_imgs)[:2] + [
                np.prod(hidden.shape[1:].as_list())])
        return hidden


class LargeEncoder(TfBasicClass):
    def __init__(self, scope='encoder', stack_imgs=0):
        TfBasicClass.__init__(self, scope, )
        self.stack_imgs = stack_imgs

    def _obj_construct(self, imgs, *args, **kwargs):
        """Extract deterministic features from an observation."""
        kwargs = dict(strides=2, activation=tf.nn.relu)
        ndims = imgs.get_shape().ndims
        if ndims == 5: # [batch, horizon, h, w, c]
            if imgs.shape[1] == 1 or self.stack_imgs == 1:
                stack_imgs = imgs
            else:
                # padding zeros
                def stack_idx(idx):
                    pre_pad_img = tf.zeros([tf.shape(imgs)[0], idx] + imgs.shape[2:].as_list(), dtype=imgs.dtype)
                    post_pad_img = tf.zeros([tf.shape(imgs)[0], self.stack_imgs - 1 - idx] + imgs.shape[2:].as_list(), dtype=imgs.dtype)
                    stacked_imgs = tf.concat([pre_pad_img, imgs, post_pad_img], axis=1)
                    return stacked_imgs
                idx_list = tuple(list(range(self.stack_imgs)))
                st_imgs = list(map(stack_idx, idx_list))
                stack_imgs = tf.concat(st_imgs, axis=-1)[:, :-1 * (self.stack_imgs - 1)]
            hidden = tf.reshape(stack_imgs, [-1] + stack_imgs.shape[2:].as_list())
        elif ndims == 4:
            stack_imgs = imgs
            hidden = imgs
        else:
            raise NotImplemented
        hidden = tf.layers.conv2d(hidden, 64, 4, name='enc_conv1', **kwargs)
        hidden = tf.layers.conv2d(hidden, 128, 4, name='enc_conv2', **kwargs)
        hidden = tf.layers.conv2d(hidden, 256, 4, name='enc_conv3', **kwargs)
        hidden = tf.layers.conv2d(hidden, 512, 4, name='enc_conv4', **kwargs)
        hidden = tf.layers.conv2d(hidden, 512, 4, name='enc_conv5', **kwargs)
        hidden = tf.layers.flatten(hidden)
        hidden = tf.layers.dense(hidden, 1024, activation=tf.nn.relu)
        assert hidden.shape[1:].as_list() == [1024], hidden.shape.as_list()
        # hidden = tf.layers.dense(hidden, 128, None, name='enc_fc5')
        if ndims == 5:
            hidden = tf.reshape(hidden, tf_util.shape(stack_imgs)[:2] + [
                np.prod(hidden.shape[1:].as_list())])
        return hidden


class Decoder(TfBasicClass):
    def __init__(self, scope='decoder'):
        TfBasicClass.__init__(self, scope)

    """Compute the data distribution of an observation from its state."""
    def _obj_construct(self, source_input, *args, **kwargs):
        state, data_shape = source_input
        final_channel = data_shape[2]
        net_kwargs = dict(strides=2, activation=tf.nn.relu)
        hidden = tf.layers.dense(state, 1024, None, name='dec_fc1')
        hidden = tf.layers.dense(hidden, 2048, activation=tf.nn.relu, name='dec_fc2')
        hidden = tf.reshape(hidden, [-1, 1, 1, hidden.shape[-1].value])
        hidden = tf.layers.conv2d_transpose(hidden, 128, 5, name='dec_conv1', **net_kwargs)
        hidden = tf.layers.conv2d_transpose(hidden, 64, 5, name='dec_conv2', **net_kwargs)
        hidden = tf.layers.conv2d_transpose(hidden, 32, 6, name='dec_conv3', **net_kwargs)
        mean = tf.layers.conv2d_transpose(hidden, final_channel, 6, strides=2, name='dec_conv4')
        mean = tf.reshape(mean, tf_util.shape(state)[:-1] + data_shape)
        return mean


class LargeDecoder(TfBasicClass):
    def __init__(self, scope='decoder'):
        TfBasicClass.__init__(self, scope)

    """Compute the data distribution of an observation from its state."""
    def _obj_construct(self, source_input, *args, **kwargs):
        state, data_shape = source_input
        final_channel = data_shape[2]
        net_kwargs = dict(strides=2, activation=tf.nn.relu)
        hidden = tf.layers.dense(state, 1024, None, name='dec_fc1')
        hidden = tf.layers.dense(hidden, 2048, None, name='dec_fc2')
        hidden = tf.reshape(hidden, [-1, 1, 1, hidden.shape[-1].value])
        hidden = tf.layers.conv2d_transpose(hidden, 256, 5, name='dec_conv1', **net_kwargs)
        hidden = tf.layers.conv2d_transpose(hidden, 128, 5, name='dec_conv2', **net_kwargs)
        hidden = tf.layers.conv2d_transpose(hidden, 64, 5, name='dec_conv3', **net_kwargs)
        hidden = tf.layers.conv2d_transpose(hidden, 32, 6, name='dec_conv4', **net_kwargs)
        mean = tf.layers.conv2d_transpose(hidden, final_channel, 6, strides=2, name='dec_conv5')
        mean = tf.reshape(mean, tf_util.shape(state)[:-1] + data_shape)
        return mean


