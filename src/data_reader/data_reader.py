import json

from src.utils import convert_list


class DataReader:
    def __init__(self, config, word_2_id):
        self.config = config
        self.word_2_id = word_2_id

    def _read_data(self, data_file):
        src_seq = []
        tgt_seq = []

        counter = 0
        with open(data_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = json.loads(line)
                paragraph = line.get('paragraph', [''])
                context_span = line.get('context_span', [0, len(paragraph)])
                paragraph = paragraph[context_span[0]:context_span[1]]
                answer = line.get('answer', [''])
                question = line.get('question', [''])

                if self.config.to_lower:
                    paragraph = list(map(str.lower, paragraph))
                paragraph = [self.config.sos] + paragraph[:self.config.sequence_len-2] + [self.config.eos]

                if self.config.to_lower:
                    answer = list(map(str.lower, answer))
                answer = [self.config.sos] + answer[:self.config.sequence_len-2] + [self.config.eos]

                if self.config.to_lower:
                    question = list(map(str.lower, question))
                question = [self.config.sos] + question[:self.config.sequence_len-2] + [self.config.eos]

                src_seq.append(convert_list(paragraph + answer, self.word_2_id, self.config.pad_id, self.config.unk_id))
                tgt_seq.append(convert_list(question, self.word_2_id, self.config.pad_id, self.config.unk_id))

                counter += 1
                if counter % 10000 == 0:
                    print('\rprocessing file {}: {:>6d}'.format(data_file, counter), end='')
            print()

        return src_seq, tgt_seq

    def read_train_data(self):
        return self._read_data(self.config.train_data)

    def read_valid_data(self):
        return self._read_data(self.config.valid_data)

    def read_test_data(self):
        return self._read_data(self.config.test_data)
