from torch import tensor
from torch.nn.functional import pad

def word_filename(idx):
    return str(idx) + '.word'


def subgroup_by_label(label):
    result = []
    for idx in range(label.bit_length()):
        if label & (1 << (idx)):
            result.append(idx + 1)
    return result


def convert_to_tensor(word, length = None):
    if length is None:
        length = len(word)
    return pad(input=tensor(word), pad=(0, length - len(word)), mode='constant', value=0)


class TorchConvertor:
    def __init__(self, length=100, word=None, label=None):
        self.__word__ = lambda w: convert_to_tensor(w, length)
        self.__labels__ = lambda l: tensor(l)


    def word(self, word):
        return self.__word__(word)


    def labels(self, labels):
        return self.__labels__(labels)


def labels_with_y(labels):
    return labels + ['y']


def default_labels(generators_number):
    return list(range(2 ** generators_number))
