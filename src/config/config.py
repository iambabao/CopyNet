import os


class Config:
    def __init__(self, root_dir, current_model, num_epoch=30, batch_size=32,
                 sequence_len=128, sentence_len=128, document_len=16,
                 top_k=5, threshold=0.4, tfidf_size=5000, embedding_size=200,
                 hidden_size=256, att_size=256,
                 kernel_size=(2, 3, 4, 5), filter_dim=64,
                 num_layer=4, num_head=8, model_dim=256,
                 fc_size_s=128, fc_size_m=512, fc_size_l=1024,
                 optimizer='Adam', lr=0.001, dropout=0.1, l2_rate=0.0,
                 embedding_trainable=False):
        self.root_dir = root_dir

        self.temp_dir = os.path.join(self.root_dir, 'temp')

        self.data_dir = os.path.join(self.root_dir, 'data')
        self.train_data = os.path.join(self.data_dir, 'data_train.json')
        self.valid_data = os.path.join(self.data_dir, 'data_valid.json')
        self.test_data = os.path.join(self.data_dir, 'data_test.json')
        self.stop_word = os.path.join(self.data_dir, 'stop_word.txt')
        self.vocab_dict = os.path.join(self.data_dir, 'vocab_dict.json')
        self.src_vocab_dict = os.path.join(self.data_dir, 'src_vocab_dict.json')
        self.tgt_vocab_dict = os.path.join(self.data_dir, 'tgt_vocab_dict.json')
        self.label_dict = os.path.join(self.data_dir, 'label_dict.json')

        self.embedding_dir = os.path.join(self.data_dir, 'embedding')
        self.plain_text = os.path.join(self.embedding_dir, 'plain_text.txt')
        self.word2vec_model = os.path.join(self.embedding_dir, 'word2vec.model')
        self.tfidf_model = os.path.join(self.embedding_dir, 'tfidf.model')
        self.glove_file = os.path.join(self.embedding_dir, 'glove.6B.300d.txt')

        self.current_model = current_model
        self.result_dir = os.path.join(self.root_dir, 'result', self.current_model)
        self.model_file = os.path.join(self.result_dir, 'model')
        self.valid_result = os.path.join(self.result_dir, 'valid_result.json')
        self.test_result = os.path.join(self.result_dir, 'test_result.json')
        self.train_log_dir = os.path.join(self.result_dir, 'train_log')
        self.valid_log_dir = os.path.join(self.result_dir, 'valid_log')

        # BERT
        self.bert_dir = os.path.join(self.root_dir, 'chinese_L-12_H-768_A-12')
        self.bert_vocab = os.path.join(self.bert_dir, 'vocab.txt')
        self.bert_config = os.path.join(self.bert_dir, 'bert_config.json')
        self.bert_ckpt = os.path.join(self.bert_dir, 'bert_model.ckpt')

        self.pad = 'PAD'
        self.pad_id = 0
        self.unk = 'UNK'
        self.unk_id = 1
        self.sos = 'SOS'
        self.sos_id = 2
        self.eos = 'EOS'
        self.eos_id = 3
        self.num = 'NUM'
        self.num_id = 4
        self.time = 'TIME'
        self.time_id = 5
        self.vocab_size = 80000
        self.src_vocab_size = 40000
        self.tgt_vocab_size = 40000
        self.to_lower = True

        self.top_k = top_k
        self.threshold = threshold
        self.tfidf_size = tfidf_size
        self.embedding_size = embedding_size
        self.sequence_len = sequence_len
        self.sentence_len = sentence_len
        self.document_len = document_len

        # RNN
        self.hidden_size = hidden_size
        self.att_size = att_size

        # CNN
        self.kernel_size = kernel_size
        self.filter_dim = filter_dim

        # Transformer
        self.num_layer = num_layer
        self.num_head = num_head
        self.model_dim = model_dim

        # FC
        self.fc_size_s = fc_size_s
        self.fc_size_m = fc_size_m
        self.fc_size_l = fc_size_l

        # Train
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.optimizer = optimizer
        self.l2_rate = l2_rate
        self.embedding_trainable = embedding_trainable
