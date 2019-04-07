import tensorflow as tf
import numpy as np
import os,sys,pickle
import data_helpers
from data_helpers import preprocess_test
from data_helpers import preprocess

did = sys.argv[1]
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("pos_dir", "data/rt-polaritydata/rt-polarity.pos", "Path of positive data")
tf.flags.DEFINE_string("neg_dir", "data/rt-polaritydata/rt-polarity.neg", "Path of negative data")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")


def eval():
    #with tf.device('/cpu:0'):
    #x_text, y = data_helpers.load_data_and_labels(FLAGS.pos_dir, FLAGS.neg_dir)
    x_test,y_test = preprocess_test(did)
    print("\nEvaluating...\n")
    # Map data into vocabulary
    #text_path = os.path.join(FLAGS.checkpoint_dir, "..", "text_vocab")
    #text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)

    #x_eval = np.array(list(text_vocab_processor.transform(x_text)))
    #y_eval = np.argmax(y, axis=1)

    #checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    checkpoint_file = "models/model-final_gru_{}".format(did)
    #checkpoint_file = "models/model_gru_001-3000"
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            for x_batch in batches:
                batch_predictions = sess.run(predictions, {input_text: x_batch,
                                                           dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
            genres_dict ={ 0 : 'Metal', 1 : 'Country', 2 : 'Rap', 3 : 'Religious', 4 : 'R&B', 5 : 'Reggae', 6 : 'Blues', 7 : 'Folk'}
            genres =['Metal','Country' ,'Rap' ,'Religious' ,'R&B' ,'Reggae' ,'Blues' ,'Folk' ]
            conf_mat ={}
            for genre in genres:
                conf_mat[genre] = {}
                for genre1 in genres:
                    conf_mat[genre][genre1] = 0
            for i in range(len(y_test)):
                conf_mat[genres_dict[y_test[i]]][genres_dict[all_predictions[i]]] += 1
            pickle.dump(conf_mat,open("conf_mat_{}".format(did),"wb"))
            
            correct_predictions = float(sum(all_predictions == y_test))
            print("Total number of test examples: {}".format(len(y_test)))
            print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))


def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()
