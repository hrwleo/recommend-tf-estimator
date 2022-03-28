# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow import feature_column as fc

import numpy as np


def build_user_model(features, mode, params):
    user_feat = params["user_feat"]
    net = fc.input_layer(features, user_feat)
    # print("user-net------")
    # print(net)
    # 全连接
    for idx, units in enumerate(params["hidden_units"]):
        net = tf.layers.dense(net, units=units, activation=tf.nn.leaky_relu, name="user_fc_layer_%s" % idx)
        net = tf.layers.dropout(net, params["dropout_rate"], training=(mode == tf.estimator.ModeKeys.TRAIN))

    # 最后隐层
    net = tf.layers.dense(net, units=params["last_hidden_units"], activation=tf.nn.leaky_relu, name="user_output_layer")
    return net


def build_item_model(features, mode, params):
    item_feat = params["item_feat"]
    net = fc.input_layer(features, item_feat)
    # print("item-net------")
    # print(net)
    # 全连接
    for idx, units in enumerate(params["hidden_units"]):
        net = tf.layers.dense(net, units=units, activation=tf.nn.leaky_relu, name="item_fc_layer_%s" % idx)
        net = tf.layers.dropout(net, params["dropout_rate"], training=(mode == tf.estimator.ModeKeys.TRAIN))

    # 最后隐层
    net = tf.layers.dense(net, units=params["last_hidden_units"], activation=tf.nn.leaky_relu,
                          name="item_output_layer")
    return net


def two_tower_model_fn(features, labels, mode, params):
    user_net = build_user_model(features, mode, params)
    item_net = build_item_model(features, mode, params)
    dot = tf.reduce_sum(tf.multiply(user_net, item_net), axis=1, keepdims=True)
    pred = tf.sigmoid(dot)
    # labels = tf.reshape(labels, shape=[-1, 1])
    # print("pred shape:", pred.shape)
    # print("labels shape: ", labels.shape)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,
                                          predictions={"output": pred, "item_net": item_net, "user_net": user_net})

    if mode == tf.estimator.ModeKeys.EVAL:
        labels = tf.reshape(labels, shape=[-1, 1])
        loss = tf.losses.log_loss(labels, pred)
        metrics = {"auc": tf.compat.v1.metrics.auc(labels=labels, predictions=pred)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        labels = tf.reshape(labels, shape=[-1, 1])
        loss = tf.losses.log_loss(labels, pred)
        global_step = tf.compat.v1.train.get_global_step()
        learning_rate = tf.compat.v1.train.exponential_decay(params["learning_rate"], global_step, 100000, 0.9,
                                                             staircase=True)
        train_op = tf.compat.v1.train.AdagradOptimizer(learning_rate).minimize(loss,
                                                                               global_step=tf.compat.v1.train.get_global_step())
        tf.summary.scalar('loss', loss)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
