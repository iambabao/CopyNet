# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/2/20 20:42
@Desc       :
"""

import tensorflow as tf


def get_sparse_softmax_cross_entropy_loss(labels, logits, mask_sequence_length=None):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    if mask_sequence_length is not None:
        mask = tf.sequence_mask(mask_sequence_length, dtype=tf.float32)
        loss = tf.reduce_mean(tf.reduce_sum(mask * loss, axis=-1) / tf.reduce_sum(mask, axis=-1))
    else:
        loss = tf.reduce_mean(loss)

    return loss


def get_sparse_cross_entropy_loss(y_true, y_pred, mask_sequence_length=None):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    if mask_sequence_length is not None:
        mask = tf.sequence_mask(mask_sequence_length, dtype=tf.float32)
        loss = tf.reduce_mean(tf.reduce_sum(mask * loss, axis=-1) / tf.reduce_sum(mask, axis=-1))
    else:
        loss = tf.reduce_mean(loss)

    return loss


def get_accuracy(y_true, y_pred, mask_sequence_length=None):
    pred_ids = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
    accuracy = tf.cast(tf.equal(y_true, pred_ids), tf.float32)
    if mask_sequence_length is not None:
        mask = tf.sequence_mask(mask_sequence_length, dtype=tf.float32)
        accuracy = tf.reduce_mean(tf.reduce_sum(mask * accuracy, axis=-1) / tf.reduce_sum(mask, axis=-1))
    else:
        accuracy = tf.reduce_mean(accuracy)

    return accuracy
