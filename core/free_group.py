import math
import random


def reciprocal(word):
    return [-factor for factor in word[::-1]]


def normalize(word):
    _word = []
    for el in word:
        if el != 0 and (len(_word) == 0 or _word[-1] != -el):
            _word.append(el)
        else:
            _word.pop()
    return _word


def is_in_subgroup(subgroup, word):
    doubled_s, doubled_rs = subgroup * 2, reciprocal(subgroup) * 2

    def sublist(needle, source):
        m = len(needle)
        for idx, el in enumerate(source):
            # el == needle[0] speeds up a little (https://stackoverflow.com/a/12576755)
            if el == needle[0] and source[idx:idx+m] == needle:
                return idx
        return -1

    def remove_subgroup_rotations(subgroup, word):
        n, m, _word, pointer, flag = len(word), len(subgroup), [], 0, False
        while pointer < n:
            # s is a rotation of t <=> |s| = |t| && s is a substirng of t_1t_2...t_nt_1t_2...t_n 
            if pointer + m <= n and (sublist(word[pointer:pointer + m], doubled_s) != -1 or sublist(word[pointer:pointer + m], doubled_rs) != -1):
                pointer, flag = pointer + m, True
            else:
                _word.append(word[pointer])
                pointer += 1
        return (flag, normalize(_word))
    
    flag, _word = True, word[::]
    while flag:
        flag, _word = remove_subgroup_rotations(subgroup, _word)
    
    return is_trivial(_word)


def is_in_intersection(subgroups, word):
    return all(map(lambda s: is_in_subgroup(s, word), subgroups))


def wu_numerator_subgroups(group_index):
    return [[t] for t in range(1, group_index + 1)] + [[t for t in range(1, group_index + 1)]]


def is_trivial(word):
    return len(word) == 0 or len(normalize(word)) == 0


def conjugation(word, conjugator):
    return normalize(reciprocal(conjugator) + word + conjugator)


def commutator(x, y):
    return normalize(reciprocal(x) + reciprocal(y) + x + y)
