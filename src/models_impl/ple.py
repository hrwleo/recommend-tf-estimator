# coding:utf-8
#############################################
# FileName: PLE.py
# Author: Stefan
# CreateTime: 2021-04-13
# Descreption: Progressive Layered Extraction (PLE) Model
#############################################
import tensorflow as tf
from tensorflow import expand_dims
from tensorflow.keras import layers, initializers, regularizers, constraints
from tensorflow import feature_column as fc

from tensorflow.keras.backend import expand_dims,repeat_elements,sum

from src.common_utils import loss_fn


class PLE(layers.Layer):
    """
    Progressive Layered Extraction (PLE) Model.
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
        super(PLE, self).__init__(**kwargs)

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
        #self.expert_activation = activations.get(expert_activation)
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

        self.expert_share_layers = []
        self.expert_task1_layers = []
        self.expert_task2_layers = []
        self.gate_layers = []

        # shared experts
        for i in range(self.num_experts):
            self.expert_share_layers.append(tf.layers.Dense(self.units, activation=self.expert_activation,
                                                   use_bias=self.use_expert_bias,
                                                   kernel_initializer=self.expert_kernel_initializer,
                                                   kernel_regularizer=self.expert_kernel_regularizer,
                                                   bias_regularizer=self.expert_bias_regularizer,
                                                   activity_regularizer=None,
                                                   kernel_constraint=self.expert_kernel_constraint,
                                                   bias_constraint=self.expert_bias_constraint))
        # task1 experts
        for i in range(self.num_experts):
            self.expert_task1_layers.append(tf.layers.Dense(self.units, activation=self.expert_activation,
                                                   use_bias=self.use_expert_bias,
                                                   kernel_initializer=self.expert_kernel_initializer,
                                                   kernel_regularizer=self.expert_kernel_regularizer,
                                                   bias_regularizer=self.expert_bias_regularizer,
                                                   activity_regularizer=None,
                                                   kernel_constraint=self.expert_kernel_constraint,
                                                   bias_constraint=self.expert_bias_constraint))
        # task2 experts
        for i in range(self.num_experts):
            self.expert_task2_layers.append(tf.layers.Dense(self.units, activation=self.expert_activation,
                                                   use_bias=self.use_expert_bias,
                                                   kernel_initializer=self.expert_kernel_initializer,
                                                   kernel_regularizer=self.expert_kernel_regularizer,
                                                   bias_regularizer=self.expert_bias_regularizer,
                                                   activity_regularizer=None,
                                                   kernel_constraint=self.expert_kernel_constraint,
                                                   bias_constraint=self.expert_bias_constraint))

        # gates
        for i in range(self.num_tasks):
            self.gate_layers.append(tf.layers.Dense(self.num_experts * 2, activation=self.gate_activation,
                                                 use_bias=self.use_gate_bias,
                                                 kernel_initializer=self.gate_kernel_initializer,
                                                 kernel_regularizer=self.gate_kernel_regularizer,
                                                 bias_regularizer=self.gate_bias_regularizer, activity_regularizer=None,
                                                 kernel_constraint=self.gate_kernel_constraint,
                                                 bias_constraint=self.gate_bias_constraint))

    def call(self, inputs):
        """
        Method for the forward function of the layer.
        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments for the base method
        :return: A tensor
        """
        #assert input_shape is not None and len(input_shape) >= 2

        expert_outputs1,expert_share_outputs, expert_outputs2, gate_outputs, final_outputs = [], [], [], [], []

        # shared expert outputs
        for expert_layer in self.expert_share_layers:
            expert_output = expand_dims(expert_layer.apply(inputs), axis=2)
            expert_share_outputs.append(expert_output)
        expert_share_outputs = tf.concat(expert_share_outputs, 2)

        # task1 expert outputs
        for expert1_layer in self.expert_task1_layers:
            expert_output = expand_dims(expert1_layer.apply(inputs), axis=2)
            expert_outputs1.append(expert_output)
        expert_outputs1.append(expert_share_outputs)
        expert_outputs1 = tf.concat(expert_outputs1,2)  # flatten  [[], [], ...] => [...]

        # task2 expert outputs
        for expert2_layer in self.expert_task2_layers:
            expert_output = expand_dims(expert2_layer.apply(inputs), axis=2)
            expert_outputs2.append(expert_output)
        expert_outputs2.append(expert_share_outputs)
        expert_outputs2 = tf.concat(expert_outputs2, 2)

        for gate_layer in self.gate_layers:
            gate_outputs.append(gate_layer.apply(inputs))

        # tower1 input
        expanded_gate_output1 = expand_dims(gate_outputs[0], axis=1)
        weighted_expert_output1 = expert_outputs1 * repeat_elements(expanded_gate_output1, self.units, axis=1)
        final_outputs.append(sum(weighted_expert_output1, axis=2))

        # tower2 input
        expanded_gate_output2 = expand_dims(gate_outputs[1], axis=1)
        weighted_expert_output2 = expert_outputs2 * repeat_elements(expanded_gate_output2, self.units, axis=1)
        final_outputs.append(sum(weighted_expert_output2, axis=2))

        # 返回的矩阵维度 num_tasks * batch * units
        return final_outputs


def build_dnn_model(features, mode, params):
    for units in params['hidden_units']:
        net = tf.layers.dense(features, units=units, activation=tf.nn.relu)
        if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
            net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))  # 只在训练时加入dropout
    # Compute logits
    logits = tf.layers.dense(net, 1, activation=None)
    return logits


def ple_model_fn(features, labels, mode, params):
    input_layer = fc.input_layer(features, params['feature_columns'])

    # Set up MMoE layer  构建MMOE层
    PLE_layers = PLE(units=4, num_experts=4, num_tasks=2)(input_layer)

    # 构建各个任务塔层
    with tf.variable_scope('ctr_model'):
        ctr_logits = build_dnn_model(PLE_layers[0], mode, params)

    with tf.variable_scope('ctcvr_model'):
        ctcvr_logits = build_dnn_model(PLE_layers[1], mode, params)

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

    ctcvr_loss = loss_fn.focal_loss(ctcvr_predictions, labels['ctcvr'])  # ctcvr 为显性标签，计算损失函数
    ctr_loss = loss_fn.focal_loss(ctr_logits, labels['ctr'])
    loss = tf.reduce_mean(tf.add(ctr_loss, ctcvr_loss, name="total_loss"))

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
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)