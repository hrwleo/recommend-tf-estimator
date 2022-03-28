# -*- coding:utf-8 -*-
import tensorflow as tf

from src.input_fn.esf_mt_input_fn import generate_batch_data, eval_input_fn, build_std_model_columns, get_file_list
from src.config import config_esf
from src.models_impl.ple import ple_model_fn

FLAGS = config_esf.FLAGS


def main(unused_argv):
    # 1.构建feature columns
    numeric_cols, category_cols, _ = build_std_model_columns()

    # 2.初始化Estimator对象
    classifier = tf.estimator.Estimator(
        model_fn=ple_model_fn,
        params={
            'feature_columns': numeric_cols + category_cols,  # 特征列变量
            'hidden_units': FLAGS.hidden_units.split(','),
            'learning_rate': FLAGS.learning_rate,
            'dropout_rate': FLAGS.dropout_rate
        },
        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )

    # 3.生成训练数据，测试数据
    train_files, eval_files = get_file_list(FLAGS.train_data, FLAGS.eval_data)

    train_data_spec, eval_data_spec = generate_batch_data(train_files, eval_files, FLAGS.batch_size,
                                                          FLAGS.train_steps, FLAGS.shuffle_buffer_size,
                                                          FLAGS.num_parallel_readers)

    # 4.Train and Eval
    print("before train and evaluate")
    tf.estimator.train_and_evaluate(classifier, train_data_spec, eval_data_spec)
    print("after train and evaluate")

    print("start evaluate")
    eval_dataset = lambda: eval_input_fn(eval_files, FLAGS.batch_size)
    results = classifier.evaluate(input_fn=eval_dataset)
    for key in sorted(results): print('%s: %s' % (key, results[key]))
    print("after evaluate")

    # 5.Export model
    print("exporting model ...")
    feature_spec = tf.feature_column.make_parse_example_spec(numeric_cols + category_cols)
    print(feature_spec)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    classifier.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)
    print("quit main")


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
