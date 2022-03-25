import math
import random
from .free_group import *


def uniform_hyperbolic_length(radius=5):
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
            factor = random.sample(subgroup, 1)[0]
            if random.random() > 0.5:
                factor = reciprocal(factor)

            conjugator = next(from_free_group(
                generators_number=generators_number, 
                length_distribution=uniform_hyperbolic_length(radius=(length - len(word) - len(factor)) // 2)
            ))
            word += conjugation(factor, conjugator)

        yield normalize(word)


def from_choice(*probabilities_with_generators):
    probabilities_with_generators = list(probabilities_with_generators)    
    assert sum(map(lambda p: p[0], probabilities_with_generators)) >= 1.

    while True:
        accumulated, coin = 0, random.random()
        for probability, generator in probabilities_with_generators:
            accumulated += probability
            if coin <= accumulated:
                yield next(generator)
                break


def from_uniform_choice(*generators):
    generators = list(generators)
    return from_choice(*[(1/len(generators), gen) for gen in generators])
