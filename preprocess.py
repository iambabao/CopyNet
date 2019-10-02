import json
import collections

from src.config import Config
from src.utils import cut_text, find_context_span


def generate_data(src_file, dst_file):
    with open(src_file, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    fout = open(dst_file, 'w', encoding='utf-8')

    counter = 0
    for line in data['data']:
        for paragraph in line['paragraphs']:
            context = cut_text(paragraph['context'])
            for qa in paragraph['qas']:
                if qa['is_impossible']:
                    continue
                question = cut_text(qa['question'])
                answer = cut_text(qa['answers'][0]['text'])
                answer_start = len(cut_text(paragraph['context'][:qa['answers'][0]['answer_start']]))
                context_span = find_context_span(context, answer_start)

                doc = {
                    'paragraph': context,
                    'question': question,
                    'answer': answer,
                    'context_span': context_span
                }
                doc['references'] = [doc['question']]
                print(json.dumps(doc, ensure_ascii=False), file=fout)

                counter += 1
                print('\rprocessing file {}: {:>6d}'.format(src_file, counter), end='')
    print()


def build_word_dict(data_file, config):
    counter = collections.Counter()

    with open(data_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = json.loads(line)

            paragraph = line['paragraph']
            if config.to_lower:
                paragraph = list(map(str.lower, paragraph))
            for word in paragraph:
                counter[word] += 1

            question = line['question']
            if config.to_lower:
                question = list(map(str.lower, question))
            for word in question:
                counter[word] += 1

    counter[config.pad] = 1e9 - config.pad_id
    counter[config.unk] = 1e9 - config.unk_id
    counter[config.sos] = 1e9 - config.sos_id
    counter[config.eos] = 1e9 - config.eos_id
    counter[config.num] = 1e9 - config.num_id
    counter[config.time] = 1e9 - config.time_id
    print('number of words in data: {}'.format(len(counter)))

    word_dict = {}
    for word, _ in counter.most_common(config.vocab_size):
        word_dict[word] = len(word_dict)

    with open(config.vocab_dict, 'w', encoding='utf-8') as fout:
        json.dump(word_dict, fout, ensure_ascii=False, indent=4)


def preprocess():
    config = Config('./', 'temp')

    print('generating data...')
    generate_data('./data/train-v2.0.json', config.train_data)
    generate_data('./data/dev-v2.0.json', config.valid_data)
    generate_data('./data/dev-v2.0.json', config.test_data)

    print('building dict...')
    build_word_dict(config.train_data, config)


if __name__ == '__main__':
    preprocess()
