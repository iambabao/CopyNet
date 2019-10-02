import collections

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util


class CopyNetWrapperState(collections.namedtuple('CopyNetWrapperState', ('cell_state', 'last_ids', 'prob_c'))):
    def clone(self, **kwargs):
        def with_same_shape(old, new):
            '''Check and set new tensor's shape.'''
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
                 vocab_size,
                 gen_vocab_size=None,
                 encoder_state_size=None,
                 initial_cell_state=None,
                 name=None):
        '''
        Args:
            cell: RNN cell
            encoder_outputs: encoder output with shape (batch_size, seq_len, hidden_size)
            encoder_input_ids: encoder input ids with shape (batch_size, seq_len)
            vocab_size: whole vocabulary size
            gen_vocab_size: generative vocabulary size
            encoder_state_size:
            initial_cell_state:
        '''
        super(CopyNetWrapper, self).__init__(name=name)
        self._cell = cell
        self._vocab_size = vocab_size
        self._gen_vocab_size = gen_vocab_size or vocab_size

        self._encoder_input_ids = encoder_input_ids
        self._encoder_outputs = encoder_outputs
        if encoder_state_size is None:
            encoder_state_size = self._encoder_outputs.shape[-1].value
            if encoder_state_size is None:
                raise ValueError('encoder_state_size must be set if we can not infer encoder_outputs last dimension size.')
        self._encoder_state_size = encoder_state_size

        self._initial_cell_state = initial_cell_state
        self._copy_weight = tf.get_variable('copy_weight', [self._encoder_state_size, self._cell.output_size])
        self._projection = tf.layers.Dense(self._gen_vocab_size, use_bias=False, name='output_projection')

    def __call__(self, inputs, state, scope=None):
        if not isinstance(state, CopyNetWrapperState):
            raise TypeError(
                'Expected state to be instance of CopyNetWrapperState. Received type {} instead.'.format(type(state))
            )
        last_ids = state.last_ids
        prob_c = state.prob_c
        cell_state = state.cell_state

        mask = tf.cast(tf.equal(tf.expand_dims(last_ids, axis=1), self._encoder_input_ids), tf.float32)
        mask_sum = tf.reduce_sum(mask, axis=1)
        mask = tf.where(tf.less(mask_sum, 1e-7), mask, mask / tf.expand_dims(mask_sum, 1))
        rou = mask * prob_c  # (batch_size, seq_len)
        selective_read = tf.einsum('ijk,ij->ik', self._encoder_outputs, rou)  # (batch_size, hidden_size)
        inputs = tf.concat([inputs, selective_read], 1)  # (batch_size, embedding_size + hidden_size)

        outputs, cell_state = self._cell(inputs, cell_state, scope)
        generate_score = self._projection(outputs)  # (batch_size, vocab_size)

        expanded_generate_score = tf.pad(generate_score, [[0, 0], [0, self._vocab_size - self._gen_vocab_size]])

        copy_score = tf.einsum('ijk,km->ijm', self._encoder_outputs, self._copy_weight)
        copy_score = tf.nn.tanh(copy_score)
        copy_score = tf.einsum('ijm,im->ij', copy_score, outputs)  # (batch_size, seq_len)

        # sum up the copy scores of the same word
        expanded_copy_score = self._expand_copy_score(copy_score)

        outputs = expanded_generate_score + expanded_copy_score
        last_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        state = CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=prob_c)

        return outputs, state

    def _expand_copy_score(self, copy_score):
        # version 1: may cause OOM problem
        # encoder_input_mask = tf.one_hot(self._encoder_input_ids, self._vocab_size)  # (batch_size, seq_len, vocab_size)
        # expanded_copy_score = tf.einsum('ijn,ij->in', encoder_input_mask, copy_score)  # (batch_size, vocab_size)
        # return expanded_copy_score

        # version 2: memory efficient
        batch_size = tf.shape(self._encoder_input_ids)[0]
        seq_len = tf.shape(self._encoder_input_ids)[1]

        # generate coordinates to satisfy tf.scatter_nd
        row_indices = tf.tile(tf.reshape(tf.range(batch_size), [-1, 1, 1]), [1, seq_len, 1])  # index of row
        col_indices = tf.expand_dims(self._encoder_input_ids, axis=-1)  # index of column
        expand_indices = tf.concat([row_indices, col_indices], axis=-1)  # (batch_size, seq_len, 2)

        expand_indices = tf.reshape(expand_indices, [-1, 2])  # (batch_size * seq_len, 2)
        expand_values = tf.reshape(copy_score, [-1])  # (batch_size * seq_len)
        expanded_copy_score = tf.scatter_nd(expand_indices, expand_values, [batch_size, self._vocab_size])
        return expanded_copy_score

    @property
    def state_size(self):
        '''
            size(s) of state(s) used by this cell.

            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        '''
        return CopyNetWrapperState(
            cell_state=self._cell.state_size,
            last_ids=tf.TensorShape([]),
            prob_c=self._encoder_state_size
        )

    @property
    def output_size(self):
        '''Integer or TensorShape: size of outputs produced by this cell.'''
        return self._vocab_size

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + 'zero_state', values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            last_ids = tf.zeros([batch_size], tf.int32) - 1
            prob_c = tf.zeros([batch_size, tf.shape(self._encoder_outputs)[1]], tf.float32)
            return CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=prob_c)
