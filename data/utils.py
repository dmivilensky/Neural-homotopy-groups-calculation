from torch import tensor, float as torchfloat
from torch.nn.functional import pad


########################
##
#  These methods are used in datasets to convert generated data to torch tensors
##

def convert_word_pad_one_side(length=100, side='left'):
    def convert(word):
        shape = (0, length - len(word)) if side == 'left' else (length - len(word), 0)
        return pad(input = tensor(word, dtype=torchfloat), pad=shape)
    return convert

def convert_word_pad_two_side(length=100):
    def convert(word):
        space = length - len(word)
        shape = (space // 2, space - space // 2)
        return pad(input = tensor(word, dtype=torchfloat), pad=shape)
    return convert

########################
##
#  These methods are used in datasets to convert labels to torch tensors
##

def convert_labels():
    def convert(labels):
        return tensor(labels, dtype=torchfloat)
    return convert

########################
##
#   columns = columns_of_subgroups(0b01, 0b10) + columns_of_intersection(0b01, 0b10)
##
#   This code creates columuns for future dataset
##

# 0b101 --> <x_1 * x_3>^F     0b110 --> <x_2 * x_3>^F
# --^-^                       --^^_      
#   3 1                         32

def subgroup_by_label(label):
    result = []
    for idx in range(label.bit_length()):
        if label & (1 << (idx)):
            result.append(idx + 1)
    return result

from core.free_group import is_in_intersection, is_in_subgroup

def columns_of_subgroups(*args):
    labels = list(args)
    return [(str(label), lambda w: is_in_subgroup(subgroup_by_label(label), w)) for label in labels]

def columns_of_intersection(*args):
    labels = list(args)
    return [(str(labels), lambda w: is_in_intersection([subgroup_by_label(label) for label in labels], w))]


########################
##
#   generator = ClosureGenerator(generators_number=2).take(100)
#   columns = columns_of_subgroups(0b01)
#   create_dataset(generator, columns, path_or_name = '2-generators;random')
##
#   This code generates folder data/datasets/2-generators;random with 100 files.
#   Each file has a name {i}.word for i from 1 to 100, and content: {'word' : <generated word>, '1': <True / False>}
##

def word_filename(idx):
    return str(idx) + '.word'

from pathlib import Path
from random import sample
from string import ascii_letters
from json import dump
from tqdm import tqdm

def create_dataset(generator, columns, path_or_name = None, verbose = True):
    if path_or_name is None:
        path = Path('data', 'datasets', ''.join(sample(ascii_letters, 6)))
    elif isinstance(path_or_name, str):
        path = Path('data', 'datasets', path_or_name)
    elif isinstance(path_or_name, Path):
        path = path_or_name
    else:
        raise ValueError

    path.mkdir(parents=True, exist_ok=True)
    
    columns.append(('word', lambda w: w))
    for idx, word in enumerate(generator) if not verbose else tqdm(enumerate(generator)):
        with open(path / word_filename(idx), 'w') as file:
            to_dump = {str(c_name) : c_value(word) for (c_name, c_value) in columns}
            dump(to_dump, file)
    return path
