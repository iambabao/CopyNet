# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/2/20 20:42
@Desc       :
"""

import os
import collections

from src.config import Config
from src.utils import read_file, read_json_lines, save_json_lines,  save_json_dict


def generate_data(src_file, tgt_file, output_file):
    data = []
    for src, tgt in zip(read_file(src_file), read_file(tgt_file)):
        src = src.strip().split()
        tgt = tgt.strip().split()
        if len(src) == 0 or len(tgt) == 0:
            continue
        data.append({'src': src, 'tgt': tgt})

    save_json_lines(data, output_file)


def build_dict(config):
    counter = collections.Counter()

    for line in read_json_lines(config.train_data):
        src_seq = line.get('src', [])
        if config.to_lower:
            src_seq = list(map(str.lower, src_seq))
        for word in src_seq:
            counter[word] += 1

        tgt_seq = line.get('tgt', [])
        if config.to_lower:
            tgt_seq = list(map(str.lower, tgt_seq))
        for word in tgt_seq:
            counter[word] += 1

    counter[config.pad] = 1e9 - config.pad_id
    counter[config.unk] = 1e9 - config.unk_id
    counter[config.sos] = 1e9 - config.sos_id
    counter[config.eos] = 1e9 - config.eos_id
    counter[config.sep] = 1e9 - config.sep_id
    counter[config.num] = 1e9 - config.num_id
    counter[config.time] = 1e9 - config.time_id
    print('number of words: {}'.format(len(counter)))

    word_dict = {}
    for word, _ in counter.most_common(config.vocab_size + config.oov_vocab_size):
        word_dict[word] = len(word_dict)

    save_json_dict(word_dict, config.vocab_dict)


def preprocess():
    config = Config('.', 'temp')

    print('generating data...')
    data_root = os.path.join(config.data_dir, 'redistribute', 'QG')
    generate_data(
        os.path.join(data_root, 'train', 'train.txt.source.txt'),
        os.path.join(data_root, 'train', 'train.txt.target.txt'),
        config.train_data
    )
    generate_data(
        os.path.join(data_root, 'dev', 'dev.txt.shuffle.dev.source.txt'),
        os.path.join(data_root, 'dev', 'dev.txt.shuffle.dev.target.txt'),
        config.valid_data
    )
    generate_data(
        os.path.join(data_root, 'test', 'dev.txt.shuffle.test.source.txt'),
        os.path.join(data_root, 'test', 'dev.txt.shuffle.test.target.txt'),
        config.test_data
    )

    print('building dict...')
    build_dict(config)


if __name__ == '__main__':
    preprocess()
