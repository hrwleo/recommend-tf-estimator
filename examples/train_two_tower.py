# -*- coding:utf-8 -*-
import shutil

import tensorflow as tf

from src.config import config_esf
from src.input_fn.esf_mt_input_fn import get_file_list
from src.input_fn.esf_twotower_input_fn import create_features_columns, input_fn
from src.models_impl.twoTower import two_tower_model_fn

FLAGS = config_esf.FLAGS

def main(unused_argv):
    # 1.构建feature columns
    user_features_columns, item_features_columns = create_features_columns()

    # 2.初始化Estimator对象
    classifier = tf.estimator.Estimator(
        model_fn=two_tower_model_fn,
        params={
            'user_feat': user_features_columns,
            'item_feat': item_features_columns,
            'hidden_units': FLAGS.hidden_units.split(','),
            'last_hidden_units': FLAGS.last_hidden_units,
            'learning_rate': FLAGS.learning_rate,
            'dropout_rate': FLAGS.dropout_rate,
        },
        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )

    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    shutil.rmtree(FLAGS.model_output, ignore_errors=True)

    # 3.生成训练数据，测试数据
    train_files, eval_files = get_file_list(FLAGS.train_data, FLAGS.eval_data)

    # 训练模型输入
    def train_input_fn():
        return input_fn(train_files, FLAGS.epochs_between_evals, True, FLAGS.batch_size, FLAGS.shuffle_buffer_size)

    # 评估模型输入
    def test_input_fn():
        return input_fn(eval_files, 1, False, FLAGS.batch_size, FLAGS.shuffle_buffer_size)

    for n in range(FLAGS.epochs // FLAGS.epochs_between_evals):
        classifier.train(input_fn=train_input_fn)
        results = classifier.evaluate(input_fn=test_input_fn)
        tf.logging.info('Results at epoch %d / %d' %
                        ((n + 1) * FLAGS.epochs_between_evals,
                        FLAGS.epochs))
        tf.logging.info('-' * 60)

        print('Results at epoch {} / {}'.format(
              (n + 1) * FLAGS.epochs_between_evals,
              FLAGS.epochs))
        print('-' * 60)

        for key in sorted(results):
            tf.logging.info('%s: %s' % (key, results[key]))
            print('{}: {}'.format(key, results[key]))


    # 导出模型
    print("exporting model...")
    feature_spec = tf.feature_column.make_parse_example_spec(user_features_columns + item_features_columns)
    example_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    classifier.export_saved_model(FLAGS.output_model, example_input_fn)
    print("quit main")

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)