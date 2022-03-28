# -*- coding: UTF-8 -*-
import shutil

import tensorflow as tf

from src.input_fn.esf_mt_input_fn import get_file_list
from src.input_fn.libsvm_input_fn import generate_batch_data, input_fn
from src.models_impl.deepFM import deepFM_model_fn
from src.config import config_esf

FLAGS = config_esf.FLAGS


def main(unused_argv):
    # 1.初始化Estimator对象
    DeepFM = tf.estimator.Estimator(
        model_fn=deepFM_model_fn,
        params={
            "field_size": FLAGS.field_size,
            "feature_size": FLAGS.feature_size,
            "embedding_size": FLAGS.embedding_size,
            "learning_rate": FLAGS.learning_rate,
            "batch_norm": FLAGS.batch_norm,
            "batch_norm_decay": FLAGS.batch_norm_decay,
            "l2_reg": FLAGS.l2_reg,
            "deep_layers": FLAGS.deep_layers,
            "dropout": FLAGS.dropout,
            "optimizer": FLAGS.optimizer
        },
        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )

    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    shutil.rmtree(FLAGS.model_output, ignore_errors=True)

    # 2.生成训练数据，测试数据
    train_files, eval_files = get_file_list(FLAGS.train_data, FLAGS.eval_data)

    train_data_spec, eval_data_spec = generate_batch_data(train_files, eval_files, FLAGS.batch_size,
                                                          FLAGS.train_steps, FLAGS.shuffle_buffer_size,
                                                          FLAGS.num_parallel_readers)

    # 3.Train and Eval
    print("before train and evaluate")
    tf.estimator.train_and_evaluate(DeepFM, train_data_spec, eval_data_spec)
    print("after train and evaluate")

    eval_dataset = lambda: input_fn(eval_files, FLAGS.batch_size, 0, 1)
    results = DeepFM.evaluate(input_fn=eval_dataset)
    for key in sorted(results): print('%s: %s' % (key, results[key]))
    print("after evaluate")

    # 4.Export model
    feature_spec = {
        'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_ids'),
        'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
    }
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
    DeepFM.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)
    print("quit main")


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
