# -*- coding: UTF-8 -*-
# coding:utf-8
#############################################
# FileName: esmm.py
# Author: Stefan
# CreateTime: 2021-04-25
# Descreption: Entire-Space Multi-Task Model
#############################################
import tensorflow as tf
from tensorflow import feature_column as fc

import numpy as np

from src.common_utils import loss_fn


def build_dnn_model(features, mode, params):
    net = fc.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
            net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))   # 只在训练时加入dropout
    # Compute logits
    logits = tf.layers.dense(net, 1, activation=None)
    return logits


def build_share_dnn_model(features, mode, params):
    net = fc.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
            net = tf.layers.dropout(net, params['dropout_rate'],
                                    training=(mode == tf.estimator.ModeKeys.TRAIN))  # 只在训练时加入dropout
    # Compute ctr logits
    ctr_logits = tf.layers.dense(net, 1, activation=None)
    ctr_pred = tf.sigmoid(ctr_logits, name="CTR")
    # Compute cvr logits
    cvr_logits = tf.layers.dense(net, 1, activation=None)
    cvr_pred = tf.sigmoid(cvr_logits, name="CVR")
    return ctr_pred, cvr_pred


# 底层共享
def esmm_share_model_fn(features, labels, mode, params):
    ctr_predictions, cvr_predictions = build_share_dnn_model(features, mode, params)
    ctcvr_predictions = tf.multiply(ctr_predictions, cvr_predictions, name="CTCVR")
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'ctr_probabilities': ctr_predictions,
            'cvr_probabilities': cvr_predictions
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    ctcvr_loss = loss_fn.focal_loss(ctcvr_predictions, labels['ctcvr'])  # ctcvr 为显性标签，计算损失函数
    ctr_loss = loss_fn.focal_loss(ctr_predictions, labels['ctr'])
    loss = tf.add(ctr_loss, ctcvr_loss, name="total_loss")

    ctr_auc = tf.metrics.auc(labels['ctr'], ctr_predictions)
    ctcvr_auc = tf.metrics.auc(labels['ctcvr'], ctcvr_predictions)
    metrics = {'ctr_auc': ctr_auc, 'ctcvr_auc': ctcvr_auc}
    tf.summary.scalar('ctr_auc', ctr_auc[1])
    tf.summary.scalar('ctcvr_auc', ctcvr_auc[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)


# 底层不共享
def esmm_model_fn(features, labels, mode, params):
    with tf.variable_scope('ctr_model'):
        ctr_logits = build_dnn_model(features, mode, params)
    with tf.variable_scope('cvr_model'):
        cvr_logits = build_dnn_model(features, mode, params)

    ctr_predictions = tf.sigmoid(ctr_logits, name="CTR")
    cvr_predictions = tf.sigmoid(cvr_logits, name="CVR")
    ctcvr = tf.multiply(ctr_predictions, cvr_predictions, name="CTCVR")
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'ctr_probabilities': ctr_predictions,
            'cvr_probabilities': cvr_predictions
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    cvr_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(labels['ctcvr'], ctcvr), name="ctcvr_loss")
    ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['ctr'], logits=ctr_logits),
                             name="ctr_loss")
    loss = tf.add(ctr_loss, cvr_loss, name="total_loss")

    ctr_auc = tf.metrics.auc(labels['ctr'], ctr_predictions)
    ctcvr_auc = tf.metrics.auc(labels['ctcvr'], ctcvr)
    metrics = {'ctr_auc': ctr_auc, 'ctcvr_auc': ctcvr_auc}
    tf.summary.scalar('ctr_auc', ctr_auc[1])
    tf.summary.scalar('ctcvr_auc', ctcvr_auc[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)


def model_predict(model, eval_input_fn):
    prediction_result = model.predict(eval_input_fn)
    ctr_probabilities = []
    cvr_probabilities = []
    for predict_dict in prediction_result:
        ctr = predict_dict['ctr_probabilities'][0]
        cvr = predict_dict['cvr_probabilities'][0]
        ctr_probabilities.append(ctr)
        cvr_probabilities.append(cvr)

    res = np.column_stack((ctr_probabilities, cvr_probabilities))
    np.savetxt("res.csv", res, delimiter=',', fmt="%s,%s")
