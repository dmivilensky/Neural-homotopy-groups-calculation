from core.free_group import free_group_bounded, is_in_subgroup, normalize, is_trivial
from core.show_word import word_to_string
from tqdm import tqdm


def naive_generate_from_intersection(generators_number, max_length, subgroups):
    word = None
    while word is None or not all(map(lambda s: is_in_subgroup(s, word), subgroups)):
        word = next(free_group_bounded(generators_number, max_length))
    return word


def naive_generator(generators_number = 2, max_length = 15):
    subgroups = [[t] for t in range(1, generators_number + 1)] + [[t for t in range(1, generators_number + 1)]]
    while True:
        yield naive_generate_from_intersection(generators_number, max_length, subgroups)

'''
Function samples words from intersection of <x_1>, <x_2>, ..., <x_1x_2x_3...>
Method is naive - check whether the random word is in the given intersection
'''
def naive_sample(generators_number = 2, max_length = 6, to_sample = 20, distinct = True, verbose = True):
    '''
    :param generators_number:   number of generators of embracing group
    :param max_length:          maximum length of generated words
    :param to_sample:           number of words to sample
    :param distinct:            is sampled words should be distintive
    :param verbose:             use tqdm? 
    :returns: list of distinct words from the given intersection
    '''
    used, generator, result = set(), naive_generator(generators_number, max_length), []
    for _ in tqdm(range(to_sample)) if verbose else range(to_sample):
        word = None
        while word is None or is_trivial(word) or (distinct and tuple(word) in used):
            word = normalize(next(generator))
        used.add(tuple(word))
        result.append(word)
    return result


def write_results(generators_number = 2, max_length = 6, to_sample = 10):
    results = naive_sample(generators_number=generators_number, max_length=max_length, to_sample=to_sample, distinct=True, verbose=True)
    from pathlib import Path
    Path('results', 'naive-sampling').mkdir(parents=True, exist_ok=True)
    with open(f'results/naive-sampling/{generators_number}-generators-{max_length}-length.txt', 'w') as file:
        for r in results:
            file.write(word_to_string(r, inverse_to_string="minus") + '\n')
