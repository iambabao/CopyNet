from .seq2seq import Seq2Seq
from .copynet import CopyNet

model_list = {
    'seq2seq': Seq2Seq,
    'copynet': CopyNet
}


def get_model(config, embedding_matrix=None):
    assert config.current_model in model_list

    return model_list[config.current_model](config, embedding_matrix)
