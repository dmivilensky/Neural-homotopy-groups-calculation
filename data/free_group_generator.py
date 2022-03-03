from random import sample, random
from math import asinh, cosh
from core.free_group import reciprocal


def generator_from_free_group_bounded(generators_number=2, max_length=5):
    generators = set(range(1, generators_number + 1)) | set(range(-generators_number, 0))

    while True:
        length = max(1, int(asinh(random() * cosh(max_length - 1))))
        word = []

        for _ in range(length):
            factor = sample(generators - set(reciprocal(word[-1:])), 1)[0]
            word.append(factor)
        
        yield word
