from torch.utils.data import Dataset

class RandomFreeGroupDataset(Dataset):
    def __init__(self, generator, count, preprocess_word, evaluate_label):
        self.generator = generator
        self.count = count
        self.preprocess_word = preprocess_word
        self.evaluate_label = evaluate_label

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        word = next(self.generator)
        return self.preprocess_word(word), self.evaluate_label(word)


from torch.nn.functional import pad
from torch import tensor, float as tfloat

def center_preprocess_word(length = 50):
    def helper(word):
        zeros = length - len(word)
        return pad(tensor(word, dtype=tfloat), (zeros // 2, zeros - zeros//2))
    return helper


from core.free_group import is_from_normal_closure

def subgroup_by_label(label):
    result = []
    for idx in range(label.bit_length()):
        if label & (1 << (idx)):
            result.append(idx + 1)
    return result

def is_from_normal_closure_label(label):
    def helper(word):
        return tensor(is_from_normal_closure(subgroup_by_label(label), word), dtype=tfloat)
    return helper


from numpy import argmin, arange

def round_postprocess_word(generators_number = 2):
    generators = arange(-generators_number, generators_number + 1)
    def helper(word):
        return list(map(
            lambda v: generators[(generators - v).abs.argmin()], word.squeeze().detach()
        ))
    return helper


from core.free_group import normalize
from random import random
from math import floor, ceil

def random_postprocess_word(generators_number = 2):
    generators_number = arange(-generators_number, generators_number + 1)

    def handle_symbol(v):
        if ceil(v) > generators[-1] or floor(v) < generators[0]:
            return generators[0] if v < 0 else generators[1]
        return ceil(v) if random() > v - floor(v) else floor(v)


    def helper(word):
        return normalize(list(map(
            handle_symbol, word.squeeze().detach()
        )))

    return helper
