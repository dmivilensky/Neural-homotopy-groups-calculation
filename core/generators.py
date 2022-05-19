import math
import random
import numpy.random as r
import core.free_group as group


def uniform_hyperbolic_length(radius=5):
    if radius <= 0:
        while True:
            yield 0
    # https://arxiv.org/pdf/1805.08207.pdf 6.3 Uniform sampling in hyperbolic space
    while True:
        yield max(1, int(round(math.acosh(1 + random.random() * (math.cosh(radius) - 1)))))


def constant_length(length=5):
    while True:
        yield length


def uniform_length(max_length=5):
    while True:
        yield random.sample(range(1, max_length + 1), 1)[0]


def from_free_group(generators_number=2, length_distribution=uniform_hyperbolic_length(5)):
    generators = set(range(1, generators_number + 1)) | set(range(-generators_number, 0))

    while True:
        length = next(length_distribution)
        word = [random.sample(generators, 1)[0]]

        for _ in range(length-1):
            factor = random.sample(generators - set([-word[-1]]), 1)[0]
            word.append(factor)
        
        yield word


def from_normal_closure(subgroup, generators_number=2, length_distribution=uniform_hyperbolic_length(5)):
    while True:
        length = next(length_distribution)
        word = []

        while len(word) < length:
            factor = subgroup[::]
            if random.random() > 0.5:
                factor = group.reciprocal(factor)

            if (length - len(word) - len(factor)) // 2 <= 0:
                break

            conjugator = next(from_free_group(
                generators_number=generators_number, 
                length_distribution=uniform_hyperbolic_length(radius=(length - len(word) - len(factor)) // 2)
            ))
            if random.random() > 0.5:
                conjugator = group.reciprocal(conjugator)
            word += group.conjugation(factor, conjugator)

        yield group.normalize(word)


class RandomChoiceGenerator:
    def __init__(self, generators, p = None):
        if p is None:
            self.p = [1 / len(generators)] * len(generators)
        self.generators = generators

    def __next__(self):
        coin = random.random()
        accumulated = 0
        for p, gen in zip(self.p, self.generators):
            accumulated += p
            if accumulated >= coin:
                return next(gen)
        return next(self.generators[-1])


def sample_word(generators_number, probas):
    choice_set = list(range(-generators_number, generators_number + 1))
    return list(map(
        lambda prob: r.choice(choice_set, size = 1, p=prob)[0], probas
    ))
