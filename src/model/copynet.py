import tensorflow as tf

from .module.learning_schedule import CustomSchedule
from .module.copynet_wrapper import CopyNetWrapper


class CopyNet:
    def __init__(self, config, embedding_matrix):
        self.sos_id = config.sos_id
        self.eos_id = config.eos_id
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.sequence_len
        self.beam_size = config.top_k

        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size

        self.lr = config.lr
        self.dropout = config.dropout

        self.src_inp = tf.placeholder(tf.int32, [None, None], name='src_inp')
        self.tgt_inp = tf.placeholder(tf.int32, [None, None], name='tgt_inp')
        self.tgt_out = tf.placeholder(tf.int32, [None, None], name='tgt_out')
        self.src_len = tf.placeholder(tf.int32, [None], name='src_len')
        self.tgt_len = tf.placeholder(tf.int32, [None], name='tgt_len')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        if embedding_matrix is not None:
            self.word_embedding = tf.keras.layers.Embedding(
                self.vocab_size,
                self.embedding_size,
                embeddings_initializer=tf.constant_initializer(embedding_matrix),
                trainable=config.embedding_trainable,
                name='word_embedding'
            )
        else:
            self.word_embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, name='word_embedding')
        self.embedding_dropout = tf.keras.layers.Dropout(self.dropout)
        self.encoder_cell_fw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        self.encoder_cell_bw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)

        if config.optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif config.optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif config.optimizer == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.lr)
        elif config.optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif config.optimizer == 'custom':
            self.lr = CustomSchedule(self.hidden_size, self.global_step)
            self.optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.98, epsilon=1e-9)
        else:
            assert False

        self.train_op, self.loss, self.accu = self.get_train_op()

        tf.summary.scalar('learning_rate', self.lr() if callable(self.lr) else self.lr)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accu)
        self.train_summary = tf.summary.merge_all()

        self.greedy_pred_id = self.greedy_inference()
        self.beam_search_pred_id = self.beam_search_inference()

        # check if new trainable variables created after get_train_op()
        # print('==========  Checking Trainable Variables  ==========')
        # for v in tf.trainable_variables():
        #     print(v)

    def get_train_op(self):
        def get_loss(labels, logits):
            mask = tf.cast(tf.math.not_equal(labels, 0), tf.float32)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_mean(tf.reduce_sum(mask * loss, axis=-1) / tf.reduce_sum(mask, axis=-1))  # loss by token

            return loss

        def get_accuracy(labels, logits):
            mask = tf.cast(tf.math.not_equal(labels, 0), tf.float32)
            pred_ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
            accuracy = tf.cast(tf.equal(labels, pred_ids), tf.float32)
            accuracy = tf.reduce_mean(tf.reduce_sum(mask * accuracy, axis=-1) / tf.reduce_sum(mask, axis=-1))

            return accuracy

        # embedding
        src_em = self._word_embedding_layer(self.src_inp, training=True)
        tgt_em = self._word_embedding_layer(self.tgt_inp, training=True)

        # encoding
        enc_output, enc_state = self._encoding_layer(src_em, reuse=False)

        # decoding
        logits = self._training_decoding_layer(enc_output, enc_state, tgt_em)

        loss = get_loss(self.tgt_out, logits)
        accu = get_accuracy(self.tgt_out, logits)

        gradients = tf.gradients(loss, tf.trainable_variables())
        gradients, _ = tf.clip_by_global_norm(gradients, 5)
        train_op = self.optimizer.apply_gradients(zip(gradients, tf.trainable_variables()), self.global_step)

        print('==========  Trainable Variables  ==========')
        for v in tf.trainable_variables():
            print(v)

        print('==========  Gradients  ==========')
        for g in gradients:
            print(g)

        return train_op, loss, accu

    # greedy decoding
    def greedy_inference(self):
        # embedding
        src_em = self._word_embedding_layer(self.src_inp, training=False)

        # encoding
        enc_output, enc_state = self._encoding_layer(src_em, reuse=True)

        # decoding
        pred_id = self._inference_decoding_layer(enc_output, enc_state, self.src_inp, beam_search=False)

        return pred_id

    # beam search decoding
    def beam_search_inference(self):
        # embedding
        src_em = self._word_embedding_layer(self.src_inp, training=False)

        # encoding
        enc_output, enc_state = self._encoding_layer(src_em, reuse=True)

        # tiled to beam size
        tiled_src_inp = tf.contrib.seq2seq.tile_batch(self.src_inp, multiplier=self.beam_size)
        tiled_enc_output = tf.contrib.seq2seq.tile_batch(enc_output, multiplier=self.beam_size)
        tiled_enc_state = tf.contrib.seq2seq.tile_batch(enc_state, multiplier=self.beam_size)

        # decoding
        pred_id = self._inference_decoding_layer(tiled_enc_output, tiled_enc_state, tiled_src_inp, beam_search=True)

        return pred_id

    def _word_embedding_layer(self, inputs, training):
        with tf.device('/cpu:0'):
            inputs_em = self.word_embedding(inputs)
        inputs_em = self.embedding_dropout(inputs_em, training=training)

        return inputs_em

    def _encoding_layer(self, src_em, reuse):
        with tf.variable_scope('encoder', reuse=reuse):
            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.encoder_cell_fw,
                self.encoder_cell_bw,
                src_em,
                sequence_length=self.src_len,
                dtype=tf.float32
            )
        enc_output = tf.concat(enc_output, axis=-1)
        enc_state = tf.maximum(enc_state[0], enc_state[1])

        return enc_output, enc_state

    def _training_decoding_layer(self, enc_output, enc_state, tgt_em):
        # add copy mechanism to decoder cell
        with tf.variable_scope('copy_mechanism', reuse=False):
            decoder_cell = CopyNetWrapper(
                self.decoder_cell,
                enc_output,
                self.src_inp,
                vocab_size=self.vocab_size,
                gen_vocab_size=self.vocab_size
            )

        dec_initial_state = decoder_cell.zero_state(batch_size=tf.shape(self.src_inp)[0], dtype=tf.float32)
        dec_initial_state = dec_initial_state.clone(cell_state=enc_state)

        # build teacher forcing decoder
        helper = tf.contrib.seq2seq.TrainingHelper(tgt_em, self.tgt_len)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, dec_initial_state)

        with tf.variable_scope('decoder', reuse=False):
            dec_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)

        logits = dec_output.rnn_output

        return logits

    def _inference_decoding_layer(self, enc_output, enc_state, src_inp, beam_search):
        # add copy mechanism to decoder cell
        with tf.variable_scope('copy_mechanism', reuse=True):
            decoder_cell = CopyNetWrapper(
                self.decoder_cell,
                enc_output,
                src_inp,
                vocab_size=self.vocab_size,
                gen_vocab_size=self.vocab_size
            )

        dec_initial_state = decoder_cell.zero_state(batch_size=tf.shape(src_inp)[0], dtype=tf.float32)
        dec_initial_state = dec_initial_state.clone(cell_state=enc_state)

        if not beam_search:
            # build greedy decoder
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.word_embedding,
                tf.fill([tf.shape(self.src_inp)[0]], self.sos_id),
                self.eos_id
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, dec_initial_state)
        else:
            # build beam search decoder
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=self.word_embedding,
                start_tokens=tf.fill([tf.shape(self.src_inp)[0]], self.sos_id),
                end_token=self.eos_id,
                initial_state=dec_initial_state,
                beam_width=self.beam_size,
                length_penalty_weight=0.0,
                coverage_penalty_weight=0.0
            )

        # decoding
        with tf.variable_scope('decoder', reuse=True):
            dec_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_seq_len)

        if not beam_search:
            pred_id = dec_output.sample_id
        else:
            pred_id = dec_output.predicted_ids  # (batch_size, seq_len, beam_size)
            pred_id = tf.transpose(pred_id, perm=[0, 2, 1])  # (batch_size, beam_size, seq_len)

        return pred_id
