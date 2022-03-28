import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string("model_dir", "./model_dir", "Base directory for the model.")
flags.DEFINE_string("output_model", "./model_output", "Path to the pb model.")
flags.DEFINE_string("train_data", "../data/data_train.tfrecord", "Directory for storing data")
flags.DEFINE_string("eval_data", "../data/data_train.tfrecord", "Path to the evaluation data.")
flags.DEFINE_integer("train_steps", 200000,
                     "Number of (global) training steps to perform")
flags.DEFINE_string("hidden_units", "512,256,128",
                    "Comma-separated list of number of units in each hidden layer of the NN")
flags.DEFINE_float("dropout_rate", 0.5, "Drop out rate")
flags.DEFINE_integer("batch_size", 1024, "Training batch size")
flags.DEFINE_integer("shuffle_buffer_size", 50000, "dataset shuffle buffer size")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_integer("num_parallel_readers", 8, "number of parallel readers for training data")
flags.DEFINE_integer("save_checkpoints_steps", 50000, "Save checkpoints every this many steps")

# fibinet config
flags.DEFINE_string("hidden_factor", "16,16", "Number of hidden factors")
flags.DEFINE_integer("reduction_ratio", 2, "reduction_ratio")
flags.DEFINE_string("pooling", "max", "pooling type for senet layer")
flags.DEFINE_boolean(name="evaluate_only", short_name="eo", default=False, help="evaluate only flag")

# tow tower
flags.DEFINE_string("last_hidden_units", "5", "last hidden layer of the NN")
flags.DEFINE_string("data_dir", "./two_tower_recommendation_system/tmp", "")
flags.DEFINE_integer("epochs", 6, "Training epochs")
flags.DEFINE_integer("epochs_between_evals", 1, "epochs between evals")
flags.DEFINE_boolean("predict", False, "Whether to predict")

# deepFM
tf.app.flags.DEFINE_integer("feature_size", 0, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 0, "Number of fields")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

FLAGS = flags.FLAGS