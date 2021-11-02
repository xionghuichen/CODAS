# !/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017-11-12
# Modified    :   2017-12-01
# Version     :   1.0

"""
This file define some common functions and give simpliest implementation.
You can modify by yourself.
"""
import tensorflow as tf
import numpy as np
from RLA.easy_log import logger

def scope_vars(scope, graph_keys=[tf.GraphKeys.GLOBAL_VARIABLES]):
    """
    Get variables inside a scope
    The scope can be specified as a string

    Parameters
    ----------
    scope: str or VariableScope
        scope in which the variables reside.
    trainable_only: bool
        whether or not to return only the variables that were marked as trainable.

    Returns
    -------
    vars: [tf.Variable]
        list of variables in `scope`.
    """
    var_list = []
    for item in graph_keys:
        var_list.extend(tf.get_collection(
            item, scope=scope if isinstance(scope, str) else scope.name))
    return var_list

#
# def huber_loss(x, delta=1.0, np=False):
#     """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
#     if np == False:
#         return tf.where(
#             tf.abs(x) < delta,
#             tf.square(x) * 0.5,
#             delta * (tf.abs(x) - 0.5 * delta)
#         )
#     else:
#         n_copy = np.array(x, copy=True)
#         n_copy[np.where(np.abs(x) < delta)] = np.square(n_copy[np.where(np.abs(x) < delta)]) * 0.5
#         n_copy[np.where(np.abs(x) >= delta)] = delta * (np.abs(n_copy[np.where(np.abs(x) >= delta)]) - 0.5 * delta)
#         return n_copy

#
# def cal_avg(arr):
#     if len(arr) == 0:
#         return 0
#     else:
#         return sum(arr) * 1.0 / len(arr)
#
#
# def cal_improve_percent(pre, cur, max_return):
#     # method 1:
#     if pre < 0:
#         return 1 - (cur / pre)
#     elif pre == 0:
#         return cur - 1
#     else:
#         return (cur / pre) - 1
#     # method 2: distance to max return.
#     # if max_return == cur:
#     #     return 0.001
#     # else:
#     #     return (max_return - pre)/(max_return - cur)
# #
#
# def dynamic_precision_function(x, x_1, x_2, y_1, y_2):
#     # return 20
#     # return 100
#     min_y = min(y_1, y_2)
#     max_y = max(y_1, y_2)
#     return min(max(min_y, (y_2 - y_1) / (x_2 - x_1) * (x - x_1) + y_1), max_y)
#
#
# def optional_get(var, default):
#     return var if var is not None else default
#
#
def padding(dataset, padding_var=0, max_length=-1):
    lengths = [len(s) for s in dataset]
    current_max_length = max(lengths)
    if max_length < 0:
        max_length = current_max_length
    else:
        if max_length != current_max_length:
            logger.warn("padding length not match current:{0}, target:{1}".format(current_max_length, max_length))
    max_shape = [len(dataset), max_length]
    max_shape.extend(dataset[0].shape[1:])
    padding_dataset = np.ones(max_shape, dtype=dataset.dtype) * padding_var
    for idx, seq in enumerate(dataset):
        length = min(len(seq), max_length)
        if len(padding_dataset.shape) > 2:
            padding_dataset[idx, :length, :] = seq[:length, :]
        else:
            padding_dataset[idx, :length] = seq[:length, :]
    return padding_dataset

# def ceil_n_digit_tf(x, digit=64):
#     return tf.ceil(x * (10 ** digit)) / (10 ** digit)
#
# def ceil_n_digit(x, digit=64):
#     if digit < 1:
#         return x
#     return np.ceil(x * (10 ** digit)) / (10 ** digit)


def compute_adjusted_r2(real_data, predict_data):
    real_data = np.array(real_data)
    predict_data = np.array(predict_data)
    r2_error = (np.sum(np.abs(real_data - predict_data)) /
                np.sum(np.abs(real_data - np.mean(real_data, axis=(0, 1)))))
    data_shape = real_data.shape
    sample_amount = np.float(np.prod(data_shape[:-1]))
    feature_amount = np.float(data_shape[-1])
    if sample_amount - feature_amount - 1 > 5:
        ar2 = 1 - r2_error * (sample_amount - 1) / (sample_amount - feature_amount - 1)
        return ar2
    else:
        return None


def compute_rmse(real_data, predict_data):
    real_data = np.array(real_data)
    predict_data = np.array(predict_data)
    return np.sqrt(np.square(real_data - predict_data).mean(axis=(0, 1))).mean() # .mean(axis=(0, 1))


def compute_image_mse(real_data, predicted_data):
    real_data = np.array(real_data)
    predicted_data = np.array(predicted_data)
    return np.sqrt(np.square(real_data - predicted_data).mean(axis=(0, 1, 2, 3))).mean() # .mean(axis=(0, 1))


def compute_rmse_d_bias(real_data, predict_data):
    real_data = np.array(real_data)
    predict_data = np.array(predict_data)
    scale = np.abs(np.mean(real_data, axis=(0, 1))) # np.sqrt(np.square(real_data - ).mean(axis=(0,1)))
    rmse = np.sqrt(np.square(real_data - predict_data).mean(axis=(0, 1)))
    return (rmse / scale).mean()


def robust_append_dict(input_dict, k, v):
    if k in input_dict:
        input_dict[k].append(v)
    else:
        input_dict[k] = [v]
