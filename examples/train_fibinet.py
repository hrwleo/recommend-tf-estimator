# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
from sklearn.metrics import roc_auc_score

import numpy as np
import tensorflow as tf

from src.common_utils.loss_fn import cross_entropy_loss
from src.input_fn.esf_ctr_input_fn import build_model_columns, feature_input_fn
from src.models_impl.fibinet import fibinet_model_fn

from src.config import config_esf

FLAGS = config_esf.FLAGS


def export_model(model, export_dir, model_column_fn):
    """Export to SavedModel format.

    Args:
      model: Estimator object
      export_dir: directory to export the model.
      model_column_fn: Function to generate model feature columns.
    """
    columns = model_column_fn
    # columns.append(tf.feature_column.numeric_column("user_id", default_value=123456, dtype=tf.int64))
    columns.append(tf.feature_column.numeric_column("label", default_value=0, dtype=tf.int64))
    feature_spec = tf.feature_column.make_parse_example_spec(columns)
    example_input_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
    model.export_savedmodel(export_dir, example_input_fn)


def model_predict(model, eval_input_fn, epoch):
    """Display evaluate result."""
    prediction_result = model.predict(eval_input_fn)

    click_sum = 0.0
    predictions = []
    # user_id_list = []
    labels = []
    num_samples = FLAGS.batch_size * FLAGS.predict_steps
    num_pre_samples = 0
    print(num_samples)
    for pred_dict in prediction_result:
        # print(pred_dict)
        # user_id = pred_dict['user_id'][0]
        p = pred_dict['probabilities'][0]
        label = float(pred_dict['label'][0])
        click_sum += p
        predictions.append(p)
        # user_id_list.append(user_id)
        labels.append(label)
        if (p >= 0.5):
            num_pre_samples += 1

        if len(predictions) % (num_samples / 10) == 0:
            tf.logging.info('predict at step %d/%d', int(float(len(predictions)) / num_samples * FLAGS.predict_steps),
                            FLAGS.predict_steps)
        if len(predictions) >= num_samples:
            break

    # tf.metrics.precision
    # print(len(predictions))
    num_samples = len(predictions)
    print('the predicted positive samples is: ' + str(num_pre_samples))
    # Display evaluation metrics

    label_mean = sum(labels) / num_samples
    prediction_mean = sum(predictions) / num_samples
    loss = sum(cross_entropy_loss(labels, predictions)) / num_samples * FLAGS.batch_size
    auc = roc_auc_score(labels, predictions)
    # group_auc = cal_group_auc(labels, predictions)

    predict_diff = np.array(predictions) - prediction_mean
    predict_diff_square_sum = sum(np.square(predict_diff))
    s_deviation = np.sqrt(predict_diff_square_sum / num_samples)
    c_deviation = s_deviation / prediction_mean

    true_positive_samples = (np.array(predictions) * np.array(labels) >= 0.5).tolist().count(True)
    false_positive_samples = (np.array(predictions) * (1 - np.array(labels)) >= 0.5).tolist().count(True)
    print(true_positive_samples)
    print(false_positive_samples)
    # precision = float(true_positive_samples)/(true_positive_samples+false_positive_samples)
    precision = 0
    false_negative_samples = (np.array(predictions) * np.array(labels) < 0.5).tolist().count(True)
    recall = float(true_positive_samples) / (true_positive_samples + false_negative_samples)
    print(false_negative_samples)
    tf.logging.info('Results at epoch %d/%d', (epoch + 1), FLAGS.num_epochs)
    tf.logging.info('-' * 60)
    tf.logging.info('label/mean: %s' % label_mean)
    tf.logging.info('predictions/mean: %s' % prediction_mean)
    tf.logging.info('total loss average batchsize: %s' % loss)
    tf.logging.info('standard deviation: %s' % s_deviation)
    tf.logging.info('coefficient of variation: %s' % c_deviation)
    tf.logging.info('precision: %s' % precision)
    tf.logging.info('recall: %s' % recall)
    tf.logging.info('auc: %s' % auc)


def main(unused_argv):
    train_files = []
    eval_files = []
    if isinstance(FLAGS.train_data, str):
        train_files = [FLAGS.train_data]

    if isinstance(FLAGS.eval_data, str):
        eval_files = [FLAGS.eval_data]

    random.shuffle(train_files)
    feature_columns, _ = build_model_columns()

    run_config = tf.estimator.RunConfig().replace(
        model_dir=FLAGS.model_dir, log_step_count_steps=1000, save_summary_steps=20000)

    model = tf.estimator.Estimator(
        model_fn=fibinet_model_fn,
        params={
            'feature_columns': feature_columns,
            'hidden_units': FLAGS.hidden_units.split(','),
            'learning_rate': FLAGS.learning_rate,
            'hidden_factor': FLAGS.hidden_factor.split(','),
            'reduction_ratio': FLAGS.reduction_ratio,
            'pooling': FLAGS.pooling
        },
        config=run_config
    )
    train_input_fn = lambda: feature_input_fn(train_files, 1, True, FLAGS.batch_size)
    eval_input_fn = lambda: feature_input_fn(eval_files, 1, False, FLAGS.batch_size)  # not shuffle for evaluate

    # model_predict(model,eval_input_fn,0)
    for epoch in range(1):
        if FLAGS.evaluate_only == False:
            model.train(train_input_fn)
        print("*" * 100)
        # results = model.evaluate(input_fn=eval_input_fn, steps=6000)
        model_predict(model, eval_input_fn, epoch)

    # Export the model
    if FLAGS.export_dir is not None:
        export_model(model, FLAGS.export_dir, feature_columns)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
