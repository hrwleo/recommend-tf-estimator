# -*- coding: UTF-8 -*-
# coding:utf-8
#############################################
# FileName: mmoe.py
# Author: Stefan
# CreateTime: 2021-04-25
# Descreption: mmoe Model  and  mmoe +wide + esmm Model
#############################################
import tensorflow as tf
from tensorflow import feature_column as fc
from tensorflow import expand_dims
from tensorflow.keras import layers, initializers, regularizers, constraints
from tensorflow.keras.backend import expand_dims, repeat_elements, sum

import numpy as np
from src.common_utils import loss_fn
from src.common_utils.layers import fm_layer


class MMoE(layers.Layer):
    """
    Multi-gate Mixture-of-Experts model.
    """

    def __init__(self,
                 units,
                 num_experts,
                 num_tasks,
                 use_expert_bias=True,
                 use_gate_bias=True,
                 expert_activation='relu',
                 gate_activation='softmax',
                 expert_bias_initializer='zeros',
                 gate_bias_initializer='zeros',
                 expert_bias_regularizer=None,
                 gate_bias_regularizer=None,
                 expert_bias_constraint=None,
                 gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling',
                 gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None,
                 gate_kernel_regularizer=None,
                 expert_kernel_constraint=None,
                 gate_kernel_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        """
         Method for instantiating MMoE layer.
        :param units: Number of hidden units
        :param num_experts: Number of experts
        :param num_tasks: Number of tasks
        :param use_expert_bias: Boolean to indicate the usage of bias in the expert weights
        :param use_gate_bias: Boolean to indicate the usage of bias in the gate weights
        :param expert_activation: Activation function of the expert weights
        :param gate_activation: Activation function of the gate weights
        :param expert_bias_initializer: Initializer for the expert bias
        :param gate_bias_initializer: Initializer for the gate bias
        :param expert_bias_regularizer: Regularizer for the expert bias
        :param gate_bias_regularizer: Regularizer for the gate bias
        :param expert_bias_constraint: Constraint for the expert bias
        :param gate_bias_constraint: Constraint for the gate bias
        :param expert_kernel_initializer: Initializer for the expert weights
        :param gate_kernel_initializer: Initializer for the gate weights
        :param expert_kernel_regularizer: Regularizer for the expert weights
        :param gate_kernel_regularizer: Regularizer for the gate weights
        :param expert_kernel_constraint: Constraint for the expert weights
        :param gate_kernel_constraint: Constraint for the gate weights
        :param activity_regularizer: Regularizer for the activity
        :param kwargs: Additional keyword arguments for the Layer class
        """
        super(MMoE, self).__init__(**kwargs)

        # Hidden nodes parameter
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)

        # Activation parameter
        # self.expert_activation = activations.get(expert_activation)
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.expert_layers = []
        self.gate_layers = []
        for i in range(self.num_experts):
            self.expert_layers.append(tf.layers.Dense(self.units, activation=self.expert_activation,
                                                      use_bias=self.use_expert_bias,
                                                      kernel_initializer=self.expert_kernel_initializer,
                                                      kernel_regularizer=self.expert_kernel_regularizer,
                                                      bias_regularizer=self.expert_bias_regularizer,
                                                      activity_regularizer=None,
                                                      kernel_constraint=self.expert_kernel_constraint,
                                                      bias_constraint=self.expert_bias_constraint))
        for i in range(self.num_tasks):
            self.gate_layers.append(tf.layers.Dense(self.num_experts, activation=self.gate_activation,
                                                    use_bias=self.use_gate_bias,
                                                    kernel_initializer=self.gate_kernel_initializer,
                                                    kernel_regularizer=self.gate_kernel_regularizer,
                                                    bias_regularizer=self.gate_bias_regularizer,
                                                    activity_regularizer=None,
                                                    kernel_constraint=self.gate_kernel_constraint,
                                                    bias_constraint=self.gate_bias_constraint))

    def call(self, inputs):
        """
        Method for the forward function of the layer.
        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments for the base method
        :return: A tensor
        """
        # assert input_shape is not None and len(input_shape) >= 2

        expert_outputs, gate_outputs, final_outputs = [], [], []
        for expert_layer in self.expert_layers:
            expert_output = expand_dims(expert_layer.apply(inputs), axis=2)
            expert_outputs.append(expert_output)
        expert_outputs = tf.concat(expert_outputs, 2)

        for gate_layer in self.gate_layers:
            gate_outputs.append(gate_layer.apply(inputs))

        for gate_output in gate_outputs:
            expanded_gate_output = expand_dims(gate_output, axis=1)
            aa = repeat_elements(expanded_gate_output, self.units, axis=1)
            weighted_expert_output = expert_outputs * aa
            bb = sum(weighted_expert_output, axis=2)
            final_outputs.append(bb)
        # 返回的矩阵维度 num_tasks * batch * units
        return final_outputs


def build_dnn_model(features, mode, params):
    # Build the hidden layers, sized according to the 'hidden_units' param.
    net = tf.layers.dense(features, units=params['hidden_units'][0], activation=tf.nn.relu)
    len_layers = len(params['hidden_units'])
    for i in range(1, len_layers):
        net = tf.layers.dense(net, units=params['hidden_units'][i], activation=tf.nn.relu)
        if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
            net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN)) # 只在训练时加入dropout
    # Compute logits
    logits = tf.layers.dense(net, 1, activation=None)
    return logits


def mmoe_model_fn(features, labels, mode, params):
    input_layer = fc.input_layer(features, params['feature_columns'])

    # Set up MMoE layer  构建MMOE层
    mmoe_layers = MMoE(units=128, num_experts=8, num_tasks=2)(input_layer)

    # 构建各个任务塔层
    with tf.variable_scope('ctr_model'):
        ctr_logits = build_dnn_model(mmoe_layers[0], mode, params)

    with tf.variable_scope('ctcvr_model'):
        ctcvr_logits = build_dnn_model(mmoe_layers[1], mode, params)

    ctr_predictions = tf.sigmoid(ctr_logits, name="CTR")
    ctcvr_predictions = tf.sigmoid(ctcvr_logits, name="CTCVR")

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'ctr_probabilities': ctr_predictions,
            'ctcvr_probabilities': ctcvr_predictions
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    # ctcvr_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(y, ctcvr_predictions), name="ctcvr_loss")
    # ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['ctr'], logits=ctr_logits),
    #                          name="ctr_loss")
    ctcvr_loss = loss_fn.focal_loss(ctcvr_predictions, labels['ctcvr'])  # ctcvr 为显性标签，计算损失函数
    ctr_loss = loss_fn.focal_loss(ctr_predictions, labels['ctr'])
    loss = tf.add(ctr_loss, ctcvr_loss, name="total_loss")

    ctr_auc = tf.metrics.auc(labels['ctr'], ctr_predictions)
    ctcvr_auc = tf.metrics.auc(labels['ctcvr'], ctcvr_predictions)
    metrics = {'ctr_auc': ctr_auc, 'ctcvr_auc': ctcvr_auc}
    tf.summary.scalar('ctr_loss', ctr_loss)
    tf.summary.scalar('ctcvr_loss', ctcvr_loss)
    tf.summary.scalar('ctr_auc', ctr_auc[1])
    tf.summary.scalar('ctcvr_auc', ctcvr_auc[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    lr = tf.train.exponential_decay(
        learning_rate=params['learning_rate'], global_step=tf.train.get_global_step(), decay_steps=50000,
        decay_rate=0.9, staircase=False) #使用默认连续衰减
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def mmoe_fm_esmm_model_fn(features, labels, mode, params):
    cont_feature = fc.input_layer(features, params['cont_columns'])

    # user interet-map feature
    prop_interest = fc.input_layer(features, params['user_cols']['prop_baseinfo_interest'])
    comm_interest = fc.input_layer(features, params['user_cols']['comm_interest'])
    user_profile = fc.input_layer(features, params['user_cols']['user_profile'])

    # user click history seq feature
    user_hist_ft = fc.input_layer(features, params['user_cols']['user_hist_seq_ft'])

    # prop
    prop_feature = fc.input_layer(features, params['prop_cols'])

    # query
    prop_feature = tf.reshape(prop_feature, shape=[-1, int(prop_feature.shape.as_list()[1] / 4), 4])  # B, Tq, dim
    # Attention: prop -> prop_interest
    prop_interest = tf.reshape(prop_interest, shape=[-1, int(prop_interest.shape.as_list()[1] / 4), 4])  # B, Tv, dim
    prop_interest_atten = tf.keras.layers.Attention()([prop_feature, prop_interest])  # B, Tq, dim
    prop_interest_atten = tf.reshape(prop_interest_atten, shape=[-1, prop_interest_atten.shape.as_list()[1] * prop_interest_atten.shape.as_list()[2]])
    # Attention: prop -> comm_interest
    comm_interest = tf.reshape(comm_interest, shape=[-1, int(comm_interest.shape.as_list()[1] / 4), 4])  # B, Tv, dim
    comm_interest_atten = tf.keras.layers.Attention()([prop_feature, comm_interest])  # B, Tq, dim
    comm_interest_atten = tf.reshape(comm_interest_atten, shape=[-1, comm_interest_atten.shape.as_list()[1] * comm_interest_atten.shape.as_list()[2]])
    # Attention: prop -> user_hist_seq_ft
    user_hist_ft = tf.reshape(user_hist_ft, shape=[-1, int(user_hist_ft.shape.as_list()[1] / 4), 4])  # B, Tv, dim
    user_hist_ft_atten = tf.keras.layers.Attention()([prop_feature, user_hist_ft])  # B, Tq, dim
    user_hist_ft_atten = tf.reshape(user_hist_ft_atten, shape=[-1, user_hist_ft_atten.shape.as_list()[1] * user_hist_ft_atten.shape.as_list()[2]])

    # concat
    prop_feature = tf.reshape(prop_feature, shape=[-1, prop_feature.shape.as_list()[1] * prop_feature.shape.as_list()[2]])
    input_layer = tf.concat([user_profile, prop_interest_atten, comm_interest_atten, user_hist_ft_atten, prop_feature, cont_feature], -1)

    # Set up Wide layer  构建Wide层
    #wide = tf.feature_column.input_layer(features, params['wide_columns'])
    #wide_layer = tf.layers.dense(wide, units=2)

    # Set up FM layer   构建FM层
    fm_input = tf.feature_column.input_layer(features, params['cate_columns'])
    fm_output = fm_layer(fm_input, 4)

    # Set up MMoE layer  构建MMOE层
    mmoe_layers = MMoE(units=128, num_experts=8, num_tasks=2)(input_layer)

    # 构建各个任务塔层
    with tf.variable_scope('ctr_model'):
        mmoe_logit1 = build_dnn_model(mmoe_layers[0], mode, params)
        ctr_net = tf.concat([fm_output, mmoe_logit1], axis=1)
        ctr_logits = tf.layers.dense(ctr_net, 1, activation=None)
    with tf.variable_scope('cvr_model'):
        mmoe_logit2 = build_dnn_model(mmoe_layers[1], mode, params)
        cvr_net = tf.concat([fm_output, mmoe_logit2], axis=1)
        cvr_logits = tf.layers.dense(cvr_net, 1, activation=None)

    ctr_predictions = tf.sigmoid(ctr_logits, name="CTR")
    cvr_predictions = tf.sigmoid(cvr_logits, name="CVR")
    ctcvr_predictions = tf.multiply(ctr_predictions, cvr_predictions, name="CTCVR")

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'ctr_probabilities': ctr_predictions,
            'ctcvr_probabilities': ctcvr_predictions
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    ctcvr_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(labels['ctcvr'], ctcvr_predictions),
                               name="ctcvr_loss")
    ctr_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(labels['ctr'], ctr_predictions),
                             name="ctr_loss")
    # ctcvr_loss = loss_fn.focal_loss(ctcvr_predictions, labels['cvr'])  # ctcvr 为显性标签，计算损失函数
    # ctr_loss = loss_fn.focal_loss(ctr_predictions, labels['ctr'])
    loss = tf.add(ctr_loss, ctcvr_loss, name="total_loss")

    ctr_auc = tf.metrics.auc(labels['ctr'], ctr_predictions)
    ctcvr_auc = tf.metrics.auc(labels['ctcvr'], ctcvr_predictions)
    metrics = {'ctr_auc': ctr_auc, 'ctcvr_auc': ctcvr_auc}
    tf.summary.scalar('ctr_loss', ctr_loss)
    tf.summary.scalar('ctcvr_loss', ctcvr_loss)
    tf.summary.scalar('ctr_auc', ctr_auc[1])
    tf.summary.scalar('ctcvr_auc', ctcvr_auc[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    lr = tf.train.exponential_decay(
        learning_rate=params['learning_rate'], global_step=tf.train.get_global_step(), decay_steps=50000,
        decay_rate=0.9, staircase=False)  # 使用默认连续衰减
    tf.summary.scalar('lr', lr)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def model_predict(model, eval_input_fn):
    prediction_result = model.predict(eval_input_fn)
    ctr_probabilities = []
    ctcvr_probabilities = []
    for predict_dict in prediction_result:
        ctr = predict_dict['ctr_probabilities'][0]
        ctcvr = predict_dict['ctcvr_probabilities'][0]
        ctr_probabilities.append(ctr)
        ctcvr_probabilities.append(ctcvr)

    res = np.column_stack((ctr_probabilities, ctcvr_probabilities))
    np.savetxt("res.csv", res, delimiter=',', fmt="%s,%s")
