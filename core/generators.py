from random import sample, random
from math import asinh, cosh
from .free_group import reciprocal, conjugation


class LengthConfig:
    LENGTH = 'length'
    MAX_LENGTH = 'max-length'
    DIST = 'dist'

    @classmethod
    def read_length_config(cls, length_config):
        if cls.LENGTH in length_config:
            return lambda: length_config[cls.LENGTH]
        elif cls.MAX_LENGTH in length_config:
            return lambda: max(1, int(asinh(random() * cosh(length_config[cls.MAX_LENGTH] - 1))))
        elif cls.DIST in length_config:
            return length_config[cls.DIST]
        else:
            raise ValueError('incorrect `length_config`')


def generator_from_free_group_bounded(generators_number=2, length_config = {'length': 5}):
    generators = set(range(-generators_number, generators_number + 1)) - set(0)
    length_generator = LengthConfig.read_length_config()
    while True:
        word, length = [], length_generator()

        for _ in range(length):
            factor = sample(generators - set(reciprocal(word[-1:])), 1)[0]
            word.append(factor)
        
        yield word


def generator_from_normal_closure(subgroup, generators_number=2, length_config = {'length': 5}):
    length_generator = LenConfig.read_length_config(length_config)
    while True:
        word, length = [], length_generator()

        while len(word) < length:
            factor = subgroup if random() > 0.5 else reciprocal(subgroup)
            
            conjugator = next(generator_from_free_group_bounded(
                generators_number=generators_number, 
                {LengthConfig.MAX_LENGTH : (length - len(word) - len(factor)) // 2}
            ))
            word += conjugation(factor, conjugator)

        yield word


def generator_from_intersection(word_sampler, group_index):
    subgroups = wu_numerator_subgroups(group_index)
    while True:
        word = []
        while is_trivial(word) or not is_in_intersection(subgroups, word):
            word = next(word_sampler)
        yield word


def default_generator_from_subgroup(generators_number = 2, subgroup = [], length_config = {'length': 5}):
    if not subgroup:
        return generator_from_free_group_bounded(generators_number, length_config)
    else:
        return generator_from_normal_closure(subgroup, generators_number, length_config)
