# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/2/23 14:56
@Desc       :
"""

import collections

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util


class CopyNetWrapperState(collections.namedtuple('CopyNetWrapperState',
                                                 ('cell_state', 'time',
                                                  'predicted_ids', 'alignments',
                                                  'coverage', 'alignment_history'))):
    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
            with_same_shape,
            self,
            super(CopyNetWrapperState, self)._replace(**kwargs)
        )


class CopyNetWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 cell,
                 encoder_outputs,
                 encoder_input_ids,
                 encoder_input_length,
                 gen_vocab_size,
                 whole_vocab_size=None,
                 encoder_state_size=None,
                 initial_cell_state=None,
                 alignment_history=False,
                 name=None):
        """
        Args:
            cell: RNN cell
            encoder_outputs: encoder output with shape (batch_size, seq_len, encoder_state_size)
            encoder_input_ids: encoder input ids with shape (batch_size, seq_len)
            encoder_input_length: length of encoder inputs with shape (batch_size)
            gen_vocab_size: generative vocabulary size
            whole_vocab_size: whole vocabulary size
            encoder_state_size:
            initial_cell_state:
            alignment_history:
        """
        super(CopyNetWrapper, self).__init__(name=name)
        self._cell = cell
        self._encoder_outputs = encoder_outputs
        self._encoder_input_ids = encoder_input_ids
        self._encoder_input_length = encoder_input_length
        self._gen_vocab_size = gen_vocab_size
        self._whole_vocab_size = whole_vocab_size or gen_vocab_size
        if self._whole_vocab_size < self._gen_vocab_size:
            raise ValueError(
                'whole_vocab_size must greater or equal to gen_vocab_size, but '
                '{} vs. {}'.format(self._whole_vocab_size, self._gen_vocab_size)
            )
        if encoder_state_size is None:
            encoder_state_size = self._encoder_outputs.shape[-1].value
            if encoder_state_size is None:
                raise ValueError('encoder_state_size must be set if we can not infer encoder_outputs last dimension size.')
        self._encoder_state_size = encoder_state_size
        self._initial_cell_state = initial_cell_state
        self._alignment_history = alignment_history

        self._copy_weight = tf.get_variable('copy_weight', [self._encoder_state_size, self._cell.output_size])
        self._projection = tf.layers.Dense(self._gen_vocab_size, use_bias=False, name='output_projection')

    def __call__(self, inputs, state, scope=None):
        if not isinstance(state, CopyNetWrapperState):
            raise TypeError(
                'Expected state to be instance of CopyNetWrapperState. Received type {} instead.'.format(type(state))
            )
        prev_cell_state = state.cell_state
        prev_time = state.time
        prev_predicted_ids = state.predicted_ids
        prev_alignments = state.alignments
        prev_coverage = state.coverage
        prev_alignment_history = state.alignment_history

        mask = tf.cast(tf.equal(prev_predicted_ids, self._encoder_input_ids), tf.float32)
        mask = tf.math.divide_no_nan(mask, tf.reduce_sum(mask, axis=-1, keepdims=True))
        rou = mask * prev_alignments
        selective_read = tf.einsum('ijk,ij->ik', self._encoder_outputs, rou)
        inputs = tf.concat([inputs, selective_read], axis=-1)  # (batch_size, embedding_size + encoder_state_size)

        cell_outputs, cell_state = self._cell(inputs, prev_cell_state, scope)
        generate_score = self._projection(cell_outputs)  # (batch_size, gen_vocab_size)

        copy_score = tf.einsum('ijk,km->ijm', self._encoder_outputs, self._copy_weight)
        copy_score = tf.nn.tanh(copy_score)
        copy_score = tf.einsum('ijm,im->ij', copy_score, cell_outputs)  # (batch_size, seq_len)

        if self._encoder_input_length is not None:
            mask = tf.sequence_mask(self._encoder_input_length)
            mask = tf.cast(tf.logical_not(mask), dtype=tf.float32)
            copy_score += -1e9 * mask
        mixed_score = tf.concat([generate_score, copy_score], axis=-1)
        mixed_prob = tf.math.softmax(mixed_score, axis=-1)
        generate_prob = mixed_prob[:, :self._gen_vocab_size]
        copy_prob = mixed_prob[:, self._gen_vocab_size:]

        # expand probability to [batch_size, whole_vocab_size]
        expanded_generate_prob = tf.pad(generate_prob, [[0, 0], [0, self._whole_vocab_size - self._gen_vocab_size]])
        expanded_copy_prob = self._expand_copy_prob(copy_prob)
        outputs = expanded_generate_prob + expanded_copy_prob  # the output is probability not logits

        predicted_ids = tf.expand_dims(tf.argmax(outputs, axis=-1, output_type=tf.int32), axis=-1)
        alignments = copy_prob
        coverage =  prev_coverage + copy_prob
        if self._alignment_history:
            alignment_history = prev_alignment_history.write(prev_time, copy_prob)
        else:
            alignment_history = prev_alignment_history
        state = CopyNetWrapperState(cell_state=cell_state, time=prev_time + 1,
                                    predicted_ids=predicted_ids, alignments=alignments,
                                    coverage=coverage, alignment_history=alignment_history)
        return outputs, state

    def _expand_copy_prob(self, copy_prob):
        batch_size = tf.shape(self._encoder_input_ids)[0]
        sequence_length = tf.shape(self._encoder_input_ids)[1]

        row_indices = tf.tile(tf.reshape(tf.range(batch_size), [-1, 1, 1]), [1, sequence_length, 1])
        col_indices = tf.expand_dims(self._encoder_input_ids, axis=-1)
        indices = tf.reshape(tf.concat([row_indices, col_indices], axis=-1), [-1, 2])  # (batch_size * seq_len, 2)
        values = tf.reshape(copy_prob, [-1])  # (batch_size * seq_len)

        expanded_copy_prob = tf.scatter_nd(indices, values, [batch_size, self._whole_vocab_size])

        return expanded_copy_prob

    @property
    def state_size(self):
        """
            size(s) of state(s) used by this cell.

            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return CopyNetWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]),
            predicted_ids=tf.TensorShape([1]),
            alignments=self._encoder_input_ids.shape[1].value,
            coverage=self._encoder_input_ids.shape[1].value,
            alignment_history=self._encoder_input_ids.shape[1].value if self._alignment_history else ()
        )

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._whole_vocab_size

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            time = tf.zeros([], tf.int32)
            predicted_ids = tf.zeros([batch_size, 1], tf.int32) - 1
            alignments = tf.zeros_like(self._encoder_input_ids, tf.float32)
            coverage = tf.zeros_like(self._encoder_input_ids, tf.float32)
            alignment_history = tf.TensorArray(tf.float32, size=0, dynamic_size=True) if self._alignment_history else ()
            return CopyNetWrapperState(cell_state=cell_state, time=time,
                                       predicted_ids=predicted_ids, alignments=alignments,
                                       coverage=coverage, alignment_history=alignment_history)
