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
from src.utils import read_dict, load_glove_embedding, make_batch_iter, pad_batch, convert_list

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, required=True)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--em', action='store_true', default=False)
args = parser.parse_args()

config = Config('./', args.model, num_epoch=args.epoch, batch_size=args.batch,
                optimizer=args.optimizer, lr=args.lr,
                embedding_trainable=args.em)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True


def save_result(outputs, result_file, id_2_label):
    with open(result_file, 'w', encoding='utf-8') as fout:
        for line in outputs:
            res = {
                'question': convert_list(line, id_2_label, config.pad, config.unk)
            }
            print(json.dumps(res, ensure_ascii=False), file=fout)


def refine_outputs(outputs):
    ret = []
    for output in outputs:
        output[-1] = config.eos_id
        ret.append(output[:output.index(config.eos_id)])

    return ret


def evaluate(sess, model, batch_iter, verbose=True):
    outputs = []
    for step, batch in enumerate(batch_iter):
        src_seq, _ = list(zip(*batch))
        src_len_seq = np.array([len(src) for src in src_seq])

        src_seq = np.array(pad_batch(src_seq, config.pad_id))

        pred_id = sess.run(
            model.greedy_pred_id,
            feed_dict={
                model.src_inp: src_seq,
                model.src_len: src_len_seq
            }
        )

        outputs.extend(refine_outputs(pred_id.tolist()))

        if verbose:
            print('\rprocessing batch: {:>6d}'.format(step + 1), end='')
    print()

    return outputs


def run_epoch(sess, model, batch_iter, summary_writer, verbose=True):
    steps = 0
    total_loss = 0.0
    total_accu = 0.0
    start_time = time.time()
    for batch in batch_iter:
        src_seq, tgt_seq = list(zip(*batch))
        src_len_seq = np.array([len(src) for src in src_seq])
        tgt_len_seq = np.array([len(tgt) for tgt in tgt_seq])

        src_seq = np.array(pad_batch(src_seq, config.pad_id))
        tgt_seq = np.array(pad_batch(tgt_seq, config.pad_id))

        _, loss, accu, global_step, summary = sess.run(
            [model.train_op, model.loss, model.accu, model.global_step, model.train_summary],
            feed_dict={
                model.src_inp: src_seq,
                model.tgt_inp: tgt_seq[:, :-1],  # 1 for eos
                model.tgt_out: tgt_seq[:, 1:],  # 1 for sos
                model.src_len: src_len_seq,
                model.tgt_len: tgt_len_seq - 1  # 1 for eos
            }
        )

        steps += 1
        total_loss += loss
        total_accu += accu
        if verbose and steps % 10 == 0:
            summary_writer.add_summary(summary, global_step)

            current_time = time.time()
            time_per_batch = (current_time - start_time) / 10.0
            start_time = current_time
            print('\rAfter {:>6d} batch(es), loss is {:>.4f}, accuracy is {:>.4f}, {:>.4f}s/batch'.format(
                steps, loss, accu, time_per_batch), end='')
    print()

    return total_loss / steps, total_accu / steps


def train():
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    if not os.path.exists(config.train_log_dir):
        os.mkdir(config.train_log_dir)
    if not os.path.exists(config.valid_log_dir):
        os.mkdir(config.valid_log_dir)

    print('loading data...')
    word_2_id, id_2_word = read_dict(config.vocab_dict)
    config.vocab_size = len(word_2_id)

    if os.path.exists(config.glove_file) and False:
        print('loading embedding matrix from file: {}'.format(config.glove_file))
        embedding_matrix, config.embedding_size = load_glove_embedding(config.glove_file, list(word_2_id.keys()))
        print('shape of embedding matrix: {}'.format(embedding_matrix.shape))
    else:
        embedding_matrix = None

    data_reader = DataReader(config, word_2_id)
    train_data = data_reader.read_train_data()
    valid_data = data_reader.read_valid_data()

    evaluator = Evaluator()

    print('building model...')
    model = get_model(config, embedding_matrix)

    best_score = 0.0
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session(config=sess_config) as sess:
        if tf.train.latest_checkpoint(config.result_dir):
            saver.restore(sess, tf.train.latest_checkpoint(config.result_dir))
            print('loading model from {}'.format(tf.train.latest_checkpoint(config.result_dir)))
        else:
            tf.global_variables_initializer().run()
            print('initializing from scratch.')

        train_writer = tf.summary.FileWriter(config.train_log_dir, sess.graph)

        for i in range(config.num_epoch):
            print('==========  Epoch {} Train  =========='.format(i + 1))
            train_batch_iter = make_batch_iter(list(zip(*train_data)), config.batch_size, shuffle=True)
            train_loss, train_accu = run_epoch(sess, model, train_batch_iter, train_writer, verbose=True)
            print('The average train loss is {:>.4f}, average accuracy is {:>.4f}'.format(train_loss, train_accu))

            print('==========  Epoch {} Valid  =========='.format(i + 1))
            valid_batch_iter = make_batch_iter(list(zip(*valid_data)), config.batch_size, shuffle=False)
            outputs = evaluate(sess, model, valid_batch_iter, verbose=True)

            print('==========  Saving Result  ==========')
            save_result(outputs, config.valid_result, id_2_word)
            score = evaluator.get_bleu_score(config.valid_data, config.valid_result, config.to_lower)
            print('score: {:>.4f}'.format(score))

            if score > best_score:
                best_score = score
                saver.save(sess, config.model_file, global_step=1)

    print('best score: {:>.4f}'.format(best_score))
    print('done')


if __name__ == '__main__':
    train()
