# -*- coding: UTF-8 -*-
import tensorflow as tf
import collections
import random


def batch_norm_layer(_x, train_phase, scope_bn, batch_norm_decay):
    bn_train = tf.contrib.layers.batch_norm(_x, decay=batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=True, reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(_x, decay=batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)  # 需要判断是否在训练中
    return z


def dice(_x, axis=-1, epsilon=0.0000001, name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        input_shape = list(_x.get_shape())

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes)
    broadcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - broadcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    broadcast_std = tf.reshape(std, broadcast_shape)
    # x_normed = (_x - broadcast_mean) / (broadcast_std + epsilon)
    x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)
    x_p = tf.sigmoid(x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x


def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_" + scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def build_deep_layers(net, params):
    # Build the hidden layers, sized according to the 'hidden_units' param.

    for layer_id, num_hidden_units in enumerate(params['hidden_units']):
        net = tf.layers.dense(net, units=num_hidden_units, activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_uniform_initializer())
    return net


def _check_fm_columns(feature_columns):
    if isinstance(feature_columns, collections.Iterator):
        feature_columns = list(feature_columns)
    column_num = len(feature_columns)
    if column_num < 2:
        raise ValueError('feature_columns must have as least two elements.')
    dimension = -1
    for column in feature_columns:
        if dimension != -1 and column.dimension != dimension:
            raise ValueError('fm_feature_columns must have the same dimension.')
        dimension = column.dimension
    return column_num, dimension


def build_Bilinear_Interaction_layers(net, params):
    # Build Bilinear-Interaction Layer

    column_num, dimension = _check_fm_columns(params['feature_columns'])
    feature_embeddings = tf.reshape(net, (-1, column_num, dimension))  # (batch_size,column_num, embedding_size)(b,f,k)

    element_wise_product_list = []
    count = 0
    for i in range(0, column_num):
        for j in range(i + 1, column_num):
            with tf.variable_scope('weight_', reuse=tf.AUTO_REUSE):
                weight = tf.get_variable(name='weight_' + str(count), shape=[dimension, dimension],
                                         initializer=tf.glorot_normal_initializer(seed=random.randint(0, 1024)),
                                         dtype=tf.float32)
            element_wise_product_list.append(
                tf.multiply(tf.matmul(feature_embeddings[:, i, :], weight), feature_embeddings[:, j, :]))
            # tf.multiply(feature_embeddings[:, i, :], feature_embeddings[:, j, :]))
            count += 1
    element_wise_product = tf.stack(element_wise_product_list)  # (f*(f-1)/2,b,k)(把它们组合成一个tensor)
    element_wise_product = tf.transpose(element_wise_product, perm=[1, 0, 2],
                                        name="element_wise_product")  # (b, f*(f-1)/2, k)

    bilinear_output = tf.layers.flatten(element_wise_product)  # (b, f*(f-1)/2*k)
    return bilinear_output


def build_SENET_layers(net, params):
    # Build SENET Layer

    column_num, dimension = _check_fm_columns(params['feature_columns'])
    reduction_ratio = params['reduction_ratio']
    feature_embeddings = tf.reshape(net, (-1, column_num, dimension))  # (batch_size,column_num, embedding_size)(b,f,k)
    original_feature = feature_embeddings
    if params['pooling'] == "max":
        feature_embeddings = tf.reduce_max(feature_embeddings, axis=2)  # (b,f) max pooling
    else:
        feature_embeddings = tf.reduce_mean(feature_embeddings, axis=2)  # (b,f) mean pooling

    reduction_num = max(column_num / reduction_ratio, 1)  # f/r
    """
    weight1 = tf.get_variable(name='weight1', shape=[column_num, reduction_num],
                             initializer=tf.glorot_normal_initializer(seed=random.randint(0, 1024)),
                             dtype=tf.float32)
    weight2 = tf.get_variable(name='weight2', shape=[reduction_num, column_num],
                              initializer=tf.glorot_normal_initializer(seed=random.randint(0, 1024)),
                              dtype=tf.float32)
    """
    att_layer = tf.layers.dense(feature_embeddings, units=reduction_num, activation=tf.nn.relu,
                                kernel_initializer=tf.glorot_uniform_initializer())  # (b, f/r)
    att_layer = tf.layers.dense(att_layer, units=column_num, activation=tf.nn.relu,
                                kernel_initializer=tf.glorot_uniform_initializer())  # (b, f)
    senet_layer = original_feature * tf.expand_dims(att_layer, axis=-1)  # (b, f, k)
    senet_output = tf.layers.flatten(senet_layer)  # (b, f*k)

    return senet_output


# attention layer
def self_attention_layer(net, params):  # S:batch_size,time_step,dim/hidden_size
    column_num, dimension = _check_fm_columns(params['user_cols']['prop_baseinfo_interest'])
    time_step = column_num
    hidden_size = dimension
    h_t = tf.Variable(tf.truncated_normal(shape=[hidden_size, 1], stddev=0.5, dtype=tf.float32))
    W = tf.Variable(tf.truncated_normal(shape=[hidden_size, hidden_size], stddev=0.5, dtype=tf.float32))

    net = tf.reshape(net, shape=[-1, column_num, hidden_size])
    score = tf.matmul(tf.matmul(tf.reshape(net, shape=[-1, hidden_size]), W), h_t)  # score: [batch_size*time_step,1]
    score = tf.reshape(score, shape=[-1, time_step, 1])  # score: [batch_size,time_step,1]

    alpha_ts = tf.nn.softmax(score)  # alpha:  [batch_size,time_step,1]
    c_t = tf.matmul(tf.transpose(net, [0, 2, 1]), alpha_ts)  # [batch_size,dim,1]

    return tf.reshape(tf.tanh(c_t), shape=[-1, hidden_size])


# FM layer Factorization Machine models pairwise (order-2) feature interactions
def fm_layer(net, dim):
    field_size = int(net.shape.as_list()[1] / dim)
    concated_embeds_value = tf.reshape(net, shape=[-1, field_size, dim]) # (batch_size,field_size,embedding_size)

    square_of_sum = tf.square(tf.reduce_sum(
        concated_embeds_value, axis=1, keep_dims=True))
    sum_of_square = tf.reduce_sum(
        concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
    cross_term = square_of_sum - sum_of_square
    cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keep_dims=False)

    return cross_term

