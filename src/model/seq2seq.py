# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/2/20 20:42
@Desc       :
"""

import tensorflow as tf

from .module.metrics import get_sparse_softmax_cross_entropy_loss, get_accuracy


class Seq2Seq:
    def __init__(self, config, word_embedding_matrix):
        self.sos_id = config.sos_id
        self.eos_id = config.eos_id
        self.vocab_size = config.vocab_size
        self.oov_vocab_size = config.oov_vocab_size
        self.max_seq_len = config.sequence_len
        self.beam_size = config.top_k
        self.beam_search = config.beam_search

        self.word_em_size = config.word_em_size
        self.hidden_size = config.hidden_size
        self.attention_size = config.attention_size
        self.lr = config.lr
        self.dropout = config.dropout

        self.src_inp = tf.placeholder(tf.int32, [None, None], name='src_inp')
        self.tgt_inp = tf.placeholder(tf.int32, [None, None], name='tgt_inp')
        self.tgt_out = tf.placeholder(tf.int32, [None, None], name='tgt_out')
        self.src_len = tf.placeholder(tf.int32, [None], name='src_len')
        self.tgt_len = tf.placeholder(tf.int32, [None], name='tgt_len')
        self.training = tf.placeholder(tf.bool, [], name='training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        if word_embedding_matrix is not None:
            self.word_embedding = tf.keras.layers.Embedding(
                self.vocab_size + self.oov_vocab_size,
                self.word_em_size,
                embeddings_initializer=tf.constant_initializer(word_embedding_matrix),
                name='word_embedding'
            )
        else:
            self.word_embedding = tf.keras.layers.Embedding(
                self.vocab_size + self.oov_vocab_size,
                self.word_em_size,
                name='word_embedding'
            )
        self.embedding_dropout = tf.keras.layers.Dropout(self.dropout)
        self.encoder_cell_fw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        self.encoder_cell_bw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        self.final_dense = tf.layers.Dense(self.vocab_size, name='final_dense')

        if config.optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif config.optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif config.optimizer == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.lr)
        elif config.optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        else:
            assert False

        # clip oov words
        clipped_tgt_out = tf.where(
            tf.greater_equal(self.tgt_out, self.vocab_size),
            tf.ones_like(self.tgt_out) * config.unk_id,
            self.tgt_out
        )

        logits, self.predicted_ids, self.alignment_history = self.forward()
        self.loss = get_sparse_softmax_cross_entropy_loss(clipped_tgt_out, logits, mask_sequence_length=self.tgt_len)
        self.accu = get_accuracy(clipped_tgt_out, logits, mask_sequence_length=self.tgt_len)
        self.gradients, self.train_op = self.get_train_op()

        tf.summary.scalar('learning_rate', self.lr() if callable(self.lr) else self.lr)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accu)
        self.summary = tf.summary.merge_all()

    def forward(self):
        # embedding
        src_em = self.src_embedding_layer()
        tgt_em = self.tgt_embedding_layer()

        # encoding
        enc_output, enc_state = self.encoding_layer(src_em)

        # decoding in training
        logits = self.training_decoding_layer(enc_output, enc_state, self.src_len, tgt_em)

        # decoding in testing
        if not self.beam_search:
            predicted_ids, alignment_history = self.inference_decoding_layer(
                enc_output, enc_state, self.src_len, beam_search=self.beam_search
            )
        else:
            # tiled to beam size
            tiled_enc_output = tf.contrib.seq2seq.tile_batch(enc_output, multiplier=self.beam_size)
            tiled_enc_state = tf.contrib.seq2seq.tile_batch(enc_state, multiplier=self.beam_size)
            tiled_src_len = tf.contrib.seq2seq.tile_batch(self.src_len, multiplier=self.beam_size)
            predicted_ids, alignment_history = self.inference_decoding_layer(
                tiled_enc_output, tiled_enc_state, tiled_src_len, beam_search=self.beam_search
            )

        return logits, predicted_ids, alignment_history

    def get_train_op(self):
        gradients = tf.gradients(self.loss, tf.trainable_variables())
        gradients, _ = tf.clip_by_global_norm(gradients, 5)
        train_op = self.optimizer.apply_gradients(zip(gradients, tf.trainable_variables()), self.global_step)

        return gradients, train_op

    def src_embedding_layer(self):
        with tf.device('/cpu:0'):
            src_em = self.word_embedding(self.src_inp)
        src_em = self.embedding_dropout(src_em, training=self.training)

        return src_em

    def tgt_embedding_layer(self):
        with tf.device('/cpu:0'):
            tgt_em = self.word_embedding(self.tgt_inp)

        return tgt_em

    def encoding_layer(self, src_em):
        with tf.variable_scope('encoder'):
            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.encoder_cell_fw,
                self.encoder_cell_bw,
                src_em,
                self.src_len,
                dtype=tf.float32
            )

            enc_output = tf.concat(enc_output, axis=-1)
            if isinstance(enc_state[0], tf.nn.rnn_cell.LSTMStateTuple) \
                    and isinstance(enc_state[1], tf.nn.rnn_cell.LSTMStateTuple):
                enc_state = tf.nn.rnn_cell.LSTMStateTuple(
                    c=enc_state[0].c + enc_state[1].c,
                    h=enc_state[0].h + enc_state[1].h
                )
            else:
                enc_state = enc_state[0] + enc_state[1]

        return enc_output, enc_state

    def training_decoding_layer(self, enc_output, enc_state, src_len, tgt_em):
        with tf.variable_scope('decoder', reuse=False):
            # add attention mechanism to decoder cell
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.attention_size,
                enc_output,
                memory_sequence_length=src_len
            )
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.decoder_cell,
                attention_mechanism
            )

            dec_initial_state = decoder_cell.zero_state(batch_size=tf.shape(enc_output)[0], dtype=tf.float32)
            dec_initial_state = dec_initial_state.clone(cell_state=enc_state)

            # build teacher forcing decoder
            helper = tf.contrib.seq2seq.TrainingHelper(tgt_em, self.tgt_len)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, dec_initial_state, self.final_dense)

            # decoding
            final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True)

            logits = final_outputs.rnn_output

        return logits

    def inference_decoding_layer(self, enc_output, enc_state, src_len, beam_search):
        with tf.variable_scope('decoder', reuse=True):
            # add attention mechanism to decoder cell
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.attention_size,
                enc_output,
                memory_sequence_length=src_len
            )
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.decoder_cell,
                attention_mechanism,
                alignment_history=not self.beam_search
            )

            dec_initial_state = decoder_cell.zero_state(batch_size=tf.shape(enc_output)[0], dtype=tf.float32)
            dec_initial_state = dec_initial_state.clone(cell_state=enc_state)

            if not beam_search:
                # build greedy decoder
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.word_embedding,
                    tf.fill([tf.shape(self.src_len)[0]], self.sos_id),
                    self.eos_id
                )
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, dec_initial_state, self.final_dense)
            else:
                # build beam search decoder
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=self.word_embedding,
                    start_tokens=tf.fill([tf.shape(self.src_len)[0]], self.sos_id),
                    end_token=self.eos_id,
                    output_layer=self.final_dense,
                    initial_state=dec_initial_state,
                    beam_width=self.beam_size,
                    length_penalty_weight=0.0,
                    coverage_penalty_weight=0.0
                )

            # decoding
            final_outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_seq_len)

            if not beam_search:
                predicted_ids = final_outputs.sample_id
                alignment_history = tf.transpose(final_state.alignment_history.stack(), perm=[1, 0, 2])
            else:
                predicted_ids = final_outputs.predicted_ids  # (batch_size, seq_len, beam_size)
                predicted_ids = tf.transpose(predicted_ids, perm=[0, 2, 1])  # (batch_size, beam_size, seq_len)
                predicted_ids = predicted_ids[:, 0, :]  # keep top one
                alignment_history = tf.no_op()

        return predicted_ids, alignment_history
