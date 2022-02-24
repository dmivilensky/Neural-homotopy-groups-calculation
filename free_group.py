import math
import random


def free_group_bounded(generators_number=2, max_length=5):
    generators = set(range(1, generators_number + 1)) | set(range(-generators_number, -1))

    while True:
        length = max(1, int(math.asinh(random.random() * math.cosh(max_length - 1))))
        word = []

        for _ in range(length):
            factor = random.sample(generators - set(word[-1:]), 1)[0]
            word.append(factor)
        
        yield word


def reciprocal(word):
    return [-factor for factor in word[::-1]]


def conjugation(word, conjugator):
    inverted_conjugator = reciprocal(conjugator)

    i = 0
    while i < min(len(inverted_conjugator), len(word)) and inverted_conjugator[-(i+1)] + word[i] == 0:
        i += 1

    j = 0
    while j < min(len(word), len(conjugator)) and word[-(j+1)] + conjugator[j] == 0:
        j += 1
    
    return inverted_conjugator[:(-i if i != 0 else len(inverted_conjugator)+1)] + word[i:(-j if j != 0 else len(word)+1)] + conjugator[j:]


def normal_closure(subgroup, generators_number=2, max_length=5):
    while True:
        length = max(1, int(math.asinh(random.random() * math.cosh(max_length - 1))))
        word = []

        while len(word) < length:
            factor = random.sample(subgroup, 1)[0]
            if random.random() > 0.5:
                factor = reciprocal(factor)

            conjugator = next(free_group_bounded(
                generators_number=generators_number, 
                max_length=(length - len(word) - len(factor)) // 2
            ))
            print(factor, conjugator)
            word += conjugation(factor, conjugator)

        yield word