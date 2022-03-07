import math
import random


def free_group_bounded(generators_number=2, max_length=5):
    generators = set(range(1, generators_number + 1)) | set(range(-generators_number, 0))

    while True:
        length = max(1, int(math.asinh(random.random() * math.cosh(max_length - 1))))
        word = []

        for _ in range(length):
            factor = random.sample(generators - set(reciprocal(word[-1:])), 1)[0]
            word.append(factor)
        
        yield word


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


'''

'''
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


def is_trivial(word):
    return len(word) == 0 or len(normalize(word)) == 0


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