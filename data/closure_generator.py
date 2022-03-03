from random import randint, choice, sample, random
from numpy.random import poisson
from math import asinh, cosh
from core.free_group import normalize, reciprocal, conjugation
from .free_group_generator import generator_from_free_group_bounded as free_group_bounded


'''
    Generates random balanced bracket sequence
'''
def random_reducibale_sequence(n):
    seq = sample([-1, 1] * n, 2 * n)

    # This now corresponds to a balanced bracket sequence (same number of
    # opening and closing brackets), but it might not be well-formed
    # (brackets closed before they open). Fix this up using the bijective
    # map in the paper (step 3).
    prefix, suffix, word = [], [], []
    partial_sum = 0
    for s in seq:
        word.append(s)
        partial_sum += s
        if partial_sum == 0: # at the end of an irreducible balanced word
            if s == -1: # it was well-formed! append it.
                prefix += word
            else:
                # it was not well-formed! fix it.
                prefix.append(1)
                suffix = [-1] + [-x for x in word[1:-1]] + suffix
            word.clear()
    return prefix + suffix
    
'''
    From random balanced bracket sequence makes random 
    trivial element of free group with index = `group_index` 
'''
def random_mutate_reducible_pairs(group_index, reducible_word):
    word, stack = reducible_word[::], []
    for idx, w in enumerate(word):
        if len(stack) == 0 or stack[-1][-1] != -w:
            stack.append((idx, w))
        else:
            match = stack.pop()
            element = choice(range(1, group_index + 1))
            indecies = sample([match[0], idx], 2)
            word[indecies[0]], word[indecies[1]] = -element, element
    return word

'''
    From random trivial element makes an element from normal closure
    by inserting subgroup element
'''
def random_insert_generators(subgroup, reducible_word):
    i_subgroup = reciprocal(subgroup)
    n, m = len(subgroup), len(reducible_word)
    places_to_insert = sample([0, 1] * (m + 1), k = m + 1)
    lengths_to_insert = (poisson(lam = 0.1, size = m + 1) + 1).tolist()
    new_word = []
    for idx, w in enumerate(reducible_word + [None]):
        length = places_to_insert[idx] * lengths_to_insert[idx]
        for _ in range(length):
            new_word.extend(subgroup if randint(1, 2) == 1 else i_subgroup )
        new_word.append(w)
    new_word.pop()
    return normalize(new_word)

'''
    Generates random element from normal closure.
    To check whether an element is in subgroup sufficient to remove rotations of subgroup
    So to generate random element from normal closure one can generate 
    balanced bracket sequence, representing remainder after removing rotations of subgroup, 
    and insert subgroup element in some places of this sequence.   
'''
def generator_from_normal_closure_v2(subgroup, group_index = 2, lam = None):
    while True:
        reducible_length = poisson(lam = lam if lam is not None else group_index ** 2)
        word = random_reducibale_sequence(reducible_length)
        word = random_mutate_reducible_pairs(group_index, word)
        word = random_insert_generators(subgroup, word)
        yield word


def generator_from_normal_closure(subgroup, generators_number=2, max_length=5):
    while True:
        length = max(1, int(asinh(random() * cosh(max_length - 1))))
        word = []

        while len(word) < length:
            factor = sample(subgroup, 1)[0]
            if random() > 0.5:
                factor = reciprocal(factor)

            conjugator = next(free_group_bounded(
                generators_number=generators_number, 
                max_length=(length - len(word) - len(factor)) // 2
            ))
            print(factor, conjugator)
            word += conjugation(factor, conjugator)

        yield word
