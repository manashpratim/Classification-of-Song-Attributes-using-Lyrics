import tensorflow as tf
import numpy as np
import os,sys
import datetime
import time
from rnn import RNN
import data_helpers

# Parameters
# ==================================================
did = sys.argv[1]
f1 = open("log_{}".format(did),"w")
# Data loading params
tf.flags.DEFINE_string("pos_dir", "data/rt-polaritydata/rt-polarity.pos", "Path of positive data")
tf.flags.DEFINE_string("neg_dir", "data/rt-polaritydata/rt-polarity.neg", "Path of negative data")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
#tf.flags.DEFINE_integer("max_sentence_length", 100, "Max sentence length in train/test data (Default: 100)")

# Model Hyperparameters
tf.flags.DEFINE_string("cell_type", "lstm", "Type of rnn cell. Choose 'vanilla' or 'lstm' or 'gru' (Default: vanilla)")
tf.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (Default: 300)")
tf.flags.DEFINE_integer("hidden_size", 64, "Dimensionality of character embedding (Default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (Default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularization lambda (Default: 3.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 60, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (Default: 100)")
tf.flags.DEFINE_integer("display_every", 50, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 11, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
f1.write("\nParameters:\n")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
    f1.write("{} = {}\n".format(attr.upper(), value))
print("")


def train(x_train, y_train, vocabulary, x_dev, y_dev,cell):
    #with tf.device('/cpu:0'):
    #    x_text, y = data_helpers.load_data_and_labels(FLAGS.pos_dir, FLAGS.neg_dir)

    #text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    #x = np.array(list(text_vocab_processor.fit_transform(x_text)))
    #print("Text Vocabulary Size: {:d}".format(len(text_vocab_processor.vocabulary_)))

    #print("x = {0}".format(x.shape))
    #print("y = {0}".format(y.shape))
    #print("")

    # Randomly shuffle data
    #np.random.seed(10)
    #shuffle_indices = np.random.permutation(np.arange(len(y)))
    #x_shuffled = x[shuffle_indices]
    #y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    #dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    #x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    #y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    #print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))
    print("*************************************************"+cell+"****************************************************")
    with tf.Graph().as_default():
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            rnn = RNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocabulary),
                embedding_size=FLAGS.embedding_dim,
                cell_type=cell,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(rnn.loss, global_step=global_step)

            # Output directory for models and summaries
            #timestamp = str(int(time.time()))
            #out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            #print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

            # Train Summaries
            #train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            #train_summary_dir = os.path.join(out_dir, "summaries", "train")
            #train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            #dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            #dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            #dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = "models/"
            checkpoint_prefix = os.path.join(checkpoint_dir, "model_{}_{}".format(cell,did))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            #text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            '''
            if FLAGS.word2vec:
                # initial matrix with random uniform
                initW = np.random.uniform(-0.25, 0.25, (len(text_vocab_processor.vocabulary_), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec file {0}".format(FLAGS.word2vec))
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in range(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1).decode('latin-1')
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx = text_vocab_processor.vocabulary_.get(word)
                        if idx != 0:
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)
                sess.run(rnn.W_text.assign(initW))
                print("Success to load pre-trained word2vec model!\n")
            '''
            initW = data_helpers.initialize_word2vec(did,FLAGS.embedding_dim)
            print("Initialized with word2vec")
            f1.write("Initialized with word2vec\n")
            sess.run(rnn.W_text.assign(initW))
            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                # Train
                feed_dict = {
                    rnn.input_text: x_batch,
                    rnn.input_y: y_batch,
                    rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, rnn.loss, rnn.accuracy], feed_dict)
                #train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    f1.write("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, accuracy))
                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    f1.write("\nEvaluation:\n")
                    print("\nEvaluation:")
                    feed_dict_dev = {
                        rnn.input_text: x_dev,
                        rnn.input_y: y_dev,
                        rnn.dropout_keep_prob: 1.0
                    }
                    step, loss, accuracy = sess.run(
                        [global_step, rnn.loss, rnn.accuracy], feed_dict_dev)
                    #dev_summary_writer.add_summary(summaries_dev, step)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, accuracy))
                    f1.write("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, accuracy))
                # Model checkpoint
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))
                    f1.write("Saved model checkpoint to {}\n".format(path))
            path = saver.save(sess, "models/model-final_{}_{}".format(cell,did))
            print("Saved model checkpoint to {}\n".format(path))
            f1.write("Saved model checkpoint to {}\n".format(path))

def main(_):
    x_train, y_train, vocabulary, x_dev, y_dev = data_helpers.preprocess(did)
    cell_types =["gru"]
    for cell_type in cell_types:
        train(x_train, y_train, vocabulary, x_dev, y_dev,cell_type)


if __name__ == "__main__":
    tf.app.run()
