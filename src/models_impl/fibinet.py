# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses

from src.common_utils.layers import build_SENET_layers, build_Bilinear_Interaction_layers, build_deep_layers

'''
A tensorflow implementation of Fibinet

'''


def fibinet_model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    senet_layer = build_SENET_layers(net, params)
    combination_layer = tf.concat([build_Bilinear_Interaction_layers(net, params),
                                   build_Bilinear_Interaction_layers(senet_layer, params)], axis=1)

    last_layer = build_deep_layers(combination_layer, params)
    # head = tf.contrib.estimator.binary_classification_head(loss_reduction=losses.Reduction.SUM)
    head = head_lib._binary_logistic_or_multi_class_head(  # pylint: disable=protected-access
        n_classes=2, weight_column=None, label_vocabulary=None, loss_reduction=losses.Reduction.SUM)
    logits = tf.layers.dense(last_layer, units=head.logits_dimension,
                             kernel_initializer=tf.glorot_uniform_initializer())
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    preds = tf.sigmoid(logits)
    label = features['label']

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': preds,
            'label': label
        }
        export_outputs = {
            'regression': tf.estimator.export.RegressionOutput(predictions['probabilities'])
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
    labels = tf.to_float(labels)
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        logits=logits,
        train_op_fn=lambda loss: optimizer.minimize(loss, global_step=tf.train.get_global_step())
    )
