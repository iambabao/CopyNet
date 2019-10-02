import os
import json
import argparse
import numpy as np
import tensorflow as tf

from src.config import Config
from src.data_reader import DataReader
from src.evaluator import Evaluator
from src.model import get_model
from src.utils import read_dict, make_batch_iter, pad_batch, convert_list

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


def beam_search_inference(sess, model, batch_iter, verbose=True):
    outputs = []
    for step, batch in enumerate(batch_iter):
        src_seq, _ = list(zip(*batch))
        src_len_seq = np.array([len(src) for src in src_seq])

        src_seq = np.array(pad_batch(src_seq, config.pad_id))

        pred_id = sess.run(
            model.beam_search_pred_id,
            feed_dict={
                model.src_inp: src_seq,
                model.src_len: src_len_seq
            }
        )

        # keep top one
        pred_id = pred_id[:, 0, :]
        outputs.extend(refine_outputs(pred_id.tolist()))

        if verbose:
            print('\rprocessing batch: {:>6d}'.format(step + 1), end='')
    print()

    return outputs


def greedy_inference(sess, model, batch_iter, verbose=True):
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


def test():
    assert os.path.exists(config.result_dir)

    print('loading data...')
    word_2_id, id_2_word = read_dict(config.vocab_dict)
    config.vocab_size = len(word_2_id)

    data_reader = DataReader(config, word_2_id)
    test_data = data_reader.read_test_data()

    evaluator = Evaluator()

    print('building model...')
    model = get_model(config)

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session(config=sess_config) as sess:
        if tf.train.latest_checkpoint(config.result_dir):
            saver.restore(sess, tf.train.latest_checkpoint(config.result_dir))
            print('loading model from {}'.format(tf.train.latest_checkpoint(config.result_dir)))

            test_batch_iter = make_batch_iter(list(zip(*test_data)), config.batch_size, shuffle=False)
            outputs = greedy_inference(sess, model, test_batch_iter, verbose=True)
            # outputs = beam_search_inference(sess, model, test_batch_iter, verbose=True)

            print('==========  Saving Result  ==========')
            save_result(outputs, config.test_result, id_2_word)
            score = evaluator.get_bleu_score(config.test_data, config.test_result, config.to_lower)
            print('score: {:>.4f}'.format(score))
        else:
            print('model not found')

    print('done')


if __name__ == '__main__':
    test()
