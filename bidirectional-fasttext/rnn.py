import tensorflow as tf


class RNN:
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 cell_type, hidden_size, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        l2_loss = tf.constant(0.0)
        text_length = self._length(self.input_text)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("text-embedding"):
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_text")
            self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)

        # Recurrent Neural Network
        with tf.name_scope("rnn"):
            cell,num_layers = self._get_cell(hidden_size, cell_type,self.dropout_keep_prob)
            #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            #all_outputs, _ = tf.nn.dynamic_rnn(cell=cell,
            #                                   inputs=self.embedded_chars,
            #                                   sequence_length=text_length,
            #                                   dtype=tf.float32)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell[0],cell[1],inputs=self.embedded_chars,sequence_length=text_length,dtype=tf.float32)
            #print(all_outputs)
            #state_fw = state[0]
            #state_bw = state[1]
            all_outputs = tf.concat([outputs[0],outputs[1]],1)
            #print(state_fw)
            #output = tf.concat([state_fw[num_layers - 1].h, state_bw[num_layers - 1].h], 1)
            self.h_outputs = self.last_relevant(all_outputs, text_length)

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[hidden_size, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_outputs, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    @staticmethod
    def _get_cell(hidden_size, cell_type,dropout_keep_prob):
        if cell_type == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        elif cell_type == "lstm":
            num_units = [256,128,64]
            #cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(n) for n in num_units])
            #print(cell)
            #return cell
            #return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            fw_cell_1 = tf.contrib.rnn.LSTMCell(256)
            bw_cell_1 = tf.contrib.rnn.LSTMCell(256)
            fw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(fw_cell_1, output_keep_prob=dropout_keep_prob)
            bw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(bw_cell_1, output_keep_prob=dropout_keep_prob)
            fw_cell_2 = tf.contrib.rnn.LSTMCell(128)
            bw_cell_2 = tf.contrib.rnn.LSTMCell(128)
            fw_cell_2 = tf.nn.rnn_cell.DropoutWrapper(fw_cell_2, output_keep_prob=dropout_keep_prob)
            bw_cell_2 = tf.nn.rnn_cell.DropoutWrapper(bw_cell_2, output_keep_prob=dropout_keep_prob)
            fw_cell_3 = tf.contrib.rnn.LSTMCell(64)
            bw_cell_3 = tf.contrib.rnn.LSTMCell(64)
            fw_cell_3 = tf.nn.rnn_cell.DropoutWrapper(fw_cell_3, output_keep_prob=dropout_keep_prob)
            bw_cell_3 = tf.nn.rnn_cell.DropoutWrapper(bw_cell_3, output_keep_prob=dropout_keep_prob)
            fw_cell = tf.contrib.rnn.MultiRNNCell([fw_cell_1,fw_cell_2,fw_cell_3],state_is_tuple=True)
            bw_cell = tf.contrib.rnn.MultiRNNCell([bw_cell_1,bw_cell_2,bw_cell_3],state_is_tuple=True)
            #initial_fw_cell = cell_fw.zero_state(self.batch_size, dtype=tf.float32)
            #initial_bw_cell = 
            #return tf.nn.rnn_cell.GRUCell(hidden_size)
            return [fw_cell,bw_cell],len(num_units)

        elif cell_type == "gru":
            num_units = [256,128,64]
            #return tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(n) for n in num_units])
            fw_cell_1 = tf.contrib.rnn.GRUCell(256)
            bw_cell_1 = tf.contrib.rnn.GRUCell(256)
            fw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(fw_cell_1, output_keep_prob=dropout_keep_prob)
            bw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(bw_cell_1, output_keep_prob=dropout_keep_prob)
            fw_cell_2 = tf.contrib.rnn.GRUCell(128)
            bw_cell_2 = tf.contrib.rnn.GRUCell(128)
            fw_cell_2 = tf.nn.rnn_cell.DropoutWrapper(fw_cell_2, output_keep_prob=dropout_keep_prob)
            bw_cell_2 = tf.nn.rnn_cell.DropoutWrapper(bw_cell_2, output_keep_prob=dropout_keep_prob)
            fw_cell_3 = tf.contrib.rnn.GRUCell(64)
            bw_cell_3 = tf.contrib.rnn.GRUCell(64)
            fw_cell_3 = tf.nn.rnn_cell.DropoutWrapper(fw_cell_3, output_keep_prob=dropout_keep_prob)
            bw_cell_3 = tf.nn.rnn_cell.DropoutWrapper(bw_cell_3, output_keep_prob=dropout_keep_prob)
            fw_cell = tf.contrib.rnn.MultiRNNCell([fw_cell_1,fw_cell_2,fw_cell_3],state_is_tuple=True)
            bw_cell = tf.contrib.rnn.MultiRNNCell([bw_cell_1,bw_cell_2,bw_cell_3],state_is_tuple=True)
            #initial_fw_cell = cell_fw.zero_state(self.batch_size, dtype=tf.float32)
            #initial_bw_cell = 
            #return tf.nn.rnn_cell.GRUCell(hidden_size)
            return [fw_cell,bw_cell],len(num_units)
        else:
            print("ERROR: '" + cell_type + "' is a wrong cell type !!!")
            return None

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        #print(length)
        return length

    # Extract the output of last cell of each sequence
    # Ex) The movie is good -> length = 4
    #     output = [ [1.314, -3.32, ..., 0.98]
    #                [0.287, -0.50, ..., 1.55]
    #                [2.194, -2.12, ..., 0.63]
    #                [1.938, -1.88, ..., 1.31]
    #                [  0.0,   0.0, ...,  0.0]
    #                ...
    #                [  0.0,   0.0, ...,  0.0] ]
    #     The output we need is 4th output of cell, so extract it.
    @staticmethod
    def last_relevant(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)
