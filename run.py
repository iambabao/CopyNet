# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/2/25 20:20
@Desc       :
"""

import os
import time
import json
import argparse
import numpy as np
import tensorflow as tf

from src.config import Config
from src.data_reader import DataReader
from src.evaluator import Evaluator
from src.model import get_model
from src.utils import makedirs, read_json_dict, read_json_lines, save_json, load_glove_embedding,\
    make_batch_iter, pad_batch, convert_list, print_title

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, required=True)
parser.add_argument('--do_train', action='store_true', default=False)
parser.add_argument('--do_eval', action='store_true', default=False)
parser.add_argument('--do_test', action='store_true', default=False)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--model_file', type=str)
parser.add_argument('--log_steps', type=int, default=100)
parser.add_argument('--save_steps', type=int, default=1000)
parser.add_argument('--pre_train_epochs', type=int, default=0)
parser.add_argument('--early_stop', type=int, default=0)
parser.add_argument('--early_stop_delta', type=float, default=0.00)
parser.add_argument('--beam_search', action='store_true', default=False)
args = parser.parse_args()

config = Config('.', args.model,
                num_epoch=args.epoch, batch_size=args.batch,
                optimizer=args.optimizer, lr=args.lr, dropout=args.dropout,
                beam_search=args.beam_search)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True


def save_result_v1(predicted_ids, alignment_history, id_2_label, input_file, output_file):
    src_inputs = []
    for line in read_json_lines(input_file):
        src_inputs.append(line['src'])

    tgt_outputs = []
    for tgt in predicted_ids:
        tgt[-1] = config.eos_id
        tgt_outputs.append(convert_list(tgt[:tgt.index(config.eos_id)], id_2_label, config.pad, config.unk))

    assert len(src_inputs) == len(tgt_outputs)

    with open(output_file, 'w', encoding='utf-8') as fout:
        for src, tgt, alignment in zip(src_inputs, tgt_outputs, alignment_history):
            for i, (word, index) in enumerate(zip(tgt, alignment)):
                if word == config.unk:
                    tgt[i] = src[index]
            print(json.dumps({'tgt': tgt}, ensure_ascii=False), file=fout)


def save_result_v2(predicted_ids, id_2_label, output_file):
    with open(output_file, 'w', encoding='utf-8') as fout:
        for tgt in predicted_ids:
            tgt[-1] = config.eos_id
            tgt = convert_list(tgt[:tgt.index(config.eos_id)], id_2_label, config.pad, config.unk)
            print(json.dumps({'tgt': tgt}, ensure_ascii=False), file=fout)


def run_test(sess, model, test_data, verbose=True):
    predicted_ids = []
    alignment_history = []
    batch_iter = make_batch_iter(list(zip(*test_data)), config.batch_size, shuffle=False, verbose=verbose)
    for step, batch in enumerate(batch_iter):
        src_seq, _ = list(zip(*batch))
        src_len_seq = np.array([len(src) for src in src_seq])

        src_seq = np.array(pad_batch(src_seq, config.pad_id))

        _predicted_ids, _alignment_history = sess.run(
            [model.predicted_ids, model.alignment_history],
            feed_dict={
                model.src_inp: src_seq,
                model.src_len: src_len_seq,
                model.training: False
            }
        )
        predicted_ids.extend(_predicted_ids.tolist())
        if not config.beam_search:
            alignment_history.extend(np.argmax(_alignment_history, axis=-1).tolist())

        if verbose:
            print('\rprocessing batch: {:>6d}'.format(step + 1), end='')
    print()

    return predicted_ids, alignment_history


def run_evaluate(sess, model, valid_data, valid_summary_writer=None, verbose=True):
    steps = 0
    predicted_ids = []
    alignment_history = []
    total_loss = 0.0
    total_accu = 0.0
    batch_iter = make_batch_iter(list(zip(*valid_data)), config.batch_size, shuffle=False, verbose=verbose)
    for batch in batch_iter:
        src_seq, tgt_seq = list(zip(*batch))
        src_len_seq = np.array([len(src) for src in src_seq])
        tgt_len_seq = np.array([len(tgt) for tgt in tgt_seq])

        src_seq = np.array(pad_batch(src_seq, config.pad_id))
        tgt_seq = np.array(pad_batch(tgt_seq, config.pad_id))

        _predicted_ids, _alignment_history, loss, accu, global_step, summary = sess.run(
            [model.predicted_ids, model.alignment_history, model.loss, model.accu, model.global_step, model.summary],
            feed_dict={
                model.src_inp: src_seq,
                model.tgt_inp: tgt_seq[:, :-1],  # 1 for eos
                model.tgt_out: tgt_seq[:, 1:],  # 1 for sos
                model.src_len: src_len_seq,
                model.tgt_len: tgt_len_seq - 1,  # 1 for eos
                model.training: False
            }
        )
        predicted_ids.extend(_predicted_ids.tolist())
        if not config.beam_search:
            alignment_history.extend(np.argmax(_alignment_history, axis=-1).tolist())

        steps += 1
        total_loss += loss
        total_accu += accu
        if verbose:
            print('\rprocessing batch: {:>6d}'.format(steps + 1), end='')
        if steps % args.log_steps == 0 and valid_summary_writer is not None:
            valid_summary_writer.add_summary(summary, global_step)
    print()

    return predicted_ids, alignment_history, total_loss / steps, total_accu / steps


def run_train(sess, model, train_data, valid_data, saver, evaluator,
              train_summary_writer=None, valid_summary_writer=None, verbose=True):
    flag = 0
    valid_log = 0.0
    best_valid_log = 0.0
    valid_log_history = {'loss': [], 'accuracy': [], 'global_step': []}
    global_step = 0
    for i in range(config.num_epoch):
        print_title('Train Epoch: {}'.format(i + 1))
        steps = 0
        total_loss = 0.0
        total_accu = 0.0
        batch_iter = make_batch_iter(list(zip(*train_data)), config.batch_size, shuffle=True, verbose=verbose)
        for batch in batch_iter:
            start_time = time.time()
            src_seq, tgt_seq = list(zip(*batch))
            src_len_seq = np.array([len(src) for src in src_seq])
            tgt_len_seq = np.array([len(tgt) for tgt in tgt_seq])

            src_seq = np.array(pad_batch(src_seq, config.pad_id))
            tgt_seq = np.array(pad_batch(tgt_seq, config.pad_id))

            _, loss, accu, global_step, summary = sess.run(
                [model.train_op, model.loss, model.accu, model.global_step, model.summary],
                feed_dict={
                    model.src_inp: src_seq,
                    model.tgt_inp: tgt_seq[:, :-1],  # 1 for eos
                    model.tgt_out: tgt_seq[:, 1:],  # 1 for sos
                    model.src_len: src_len_seq,
                    model.tgt_len: tgt_len_seq - 1,  # 1 for eos
                    model.training: True
                }
            )

            steps += 1
            total_loss += loss
            total_accu += accu
            if verbose:
                print('\rafter {:>6d} batch(s), train loss is {:>.4f}, train accuracy is {:>.4f}, {:>.4f}s/batch'
                      .format(steps, loss, accu, time.time() - start_time), end='')
            if steps % args.log_steps == 0 and train_summary_writer is not None:
                train_summary_writer.add_summary(summary, global_step)
            if global_step % args.save_steps == 0:
                # evaluate saved models after pre-train epochs
                if i < args.pre_train_epochs:
                    saver.save(sess, config.model_file, global_step=global_step)
                else:
                    predicted_ids, alignment_history, valid_loss, valid_accu = run_evaluate(
                        sess, model, valid_data, valid_summary_writer, verbose=False
                    )
                    print_title('Valid Result', sep='*')
                    print('average valid loss: {:>.4f}, average valid accuracy: {:>.4f}'.format(valid_loss, valid_accu))

                    print_title('Saving Result')
                    if not config.beam_search:
                        save_result_v1(predicted_ids, alignment_history, config.id_2_word, config.valid_data, config.valid_result)
                    else:
                        save_result_v2(predicted_ids, config.id_2_word, config.valid_result)
                    valid_results = evaluator.evaluate(config.valid_data, config.valid_result, config.to_lower)

                    if valid_results['Bleu_4'] >= best_valid_log:
                        best_valid_log = valid_results['Bleu_4']
                        saver.save(sess, config.model_file, global_step=global_step)

                    # early stop
                    if valid_results['Bleu_4'] - args.early_stop_delta >= valid_log:
                        flag = 0
                    elif flag < args.early_stop:
                        flag += 1
                    elif args.early_stop:
                        return valid_log_history

                    valid_log = valid_results['Bleu_4']
                    valid_log_history['loss'].append(valid_loss)
                    valid_log_history['accuracy'].append(valid_accu)
                    valid_log_history['global_step'].append(int(global_step))
        print()
        print_title('Train Result')
        print('average train loss: {:>.4f}, average train accuracy: {:>.4f}'.format(
            total_loss / steps, total_accu / steps))
    saver.save(sess, config.model_file, global_step=global_step)

    return valid_log_history


def main():
    makedirs(config.temp_dir)
    makedirs(config.result_dir)
    makedirs(config.train_log_dir)
    makedirs(config.valid_log_dir)

    print('preparing data...')
    config.word_2_id, config.id_2_word = read_json_dict(config.vocab_dict)
    config.vocab_size = min(config.vocab_size, len(config.word_2_id))
    config.oov_vocab_size = min(config.oov_vocab_size, len(config.word_2_id) - config.vocab_size)

    embedding_matrix = None
    if args.do_train:
        if os.path.exists(config.glove_file):
            print('loading embedding matrix from file: {}'.format(config.glove_file))
            embedding_matrix, config.word_em_size = load_glove_embedding(config.glove_file, list(config.word_2_id.keys()))
            print('shape of embedding matrix: {}'.format(embedding_matrix.shape))
    else:
        if os.path.exists(config.glove_file):
            with open(config.glove_file, 'r', encoding='utf-8') as fin:
                line = fin.readline()
                config.word_em_size = len(line.strip().split()) - 1

    data_reader = DataReader(config)
    evaluator = Evaluator('tgt')

    print('building model...')
    model = get_model(config, embedding_matrix)
    saver = tf.train.Saver(max_to_keep=10)

    if args.do_train:
        print('loading data...')
        train_data = data_reader.read_train_data()
        valid_data = data_reader.read_valid_data()

        print_title('Trainable Variables')
        for v in tf.trainable_variables():
            print(v)

        print_title('Gradients')
        for g in model.gradients:
            print(g)

        with tf.Session(config=sess_config) as sess:
            model_file = args.model_file
            if model_file is None:
                model_file = tf.train.latest_checkpoint(config.result_dir)
            if model_file is not None:
                print('loading model from {}...'.format(model_file))
                saver.restore(sess, model_file)
            else:
                print('initializing from scratch...')
                tf.global_variables_initializer().run()

            train_writer = tf.summary.FileWriter(config.train_log_dir, sess.graph)
            valid_writer = tf.summary.FileWriter(config.valid_log_dir, sess.graph)

            valid_log_history = run_train(sess, model, train_data, valid_data, saver, evaluator,
                                          train_writer, valid_writer, verbose=True)
            save_json(valid_log_history, os.path.join(config.result_dir, 'valid_log_history.json'))

    if args.do_eval:
        print('loading data...')
        valid_data = data_reader.read_valid_data()

        with tf.Session(config=sess_config) as sess:
            model_file = args.model_file
            if model_file is None:
                model_file = tf.train.latest_checkpoint(config.result_dir)
            if model_file is not None:
                print('loading model from {}...'.format(model_file))
                saver.restore(sess, model_file)

                predicted_ids, alignment_history, valid_loss, valid_accu = run_evaluate(sess, model, valid_data, verbose=True)
                print('average valid loss: {:>.4f}, average valid accuracy: {:>.4f}'.format(valid_loss, valid_accu))

                print_title('Saving Result')
                if not config.beam_search:
                    save_result_v1(predicted_ids, alignment_history, config.id_2_word, config.valid_data, config.valid_result)
                else:
                    save_result_v2(predicted_ids, config.id_2_word, config.valid_result)
                evaluator.evaluate(config.valid_data, config.valid_result,config.to_lower)
            else:
                print('model not found!')

    if args.do_test:
        print('loading data...')
        test_data = data_reader.read_test_data()

        with tf.Session(config=sess_config) as sess:
            model_file = args.model_file
            if model_file is None:
                model_file = tf.train.latest_checkpoint(config.result_dir)
            if model_file is not None:
                print('loading model from {}...'.format(model_file))
                saver.restore(sess, model_file)

                predicted_ids, alignment_history = run_test(sess, model, test_data, verbose=True)

                print_title('Saving Result')
                if not config.beam_search:
                    save_result_v1(predicted_ids, alignment_history, config.id_2_word, config.test_data, config.test_result)
                else:
                    save_result_v2(predicted_ids, config.id_2_word, config.test_result)
                evaluator.evaluate(config.test_data, config.test_result, config.to_lower)
            else:
                print('model not found!')


if __name__ == '__main__':
    main()
    print('done')
