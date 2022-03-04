from data.intersection_generator import generator_from_intersection
from data.commutator_generator import generator_from_commutator
from core.free_group import is_trivial
from core.show_word import word_to_string
from pathlib import Path
from tqdm import tqdm


def distinct_from_sampler(word_sampler, verbose = True, count = 5):
    result, used = list(), set()
    for _ in tqdm(range(count)) if verbose else range(count):
        word = []
        while is_trivial(word) or word_to_string(word) in used:
            word = next(word_sampler)
        result.append(word)
        used.add(word_to_string(word))
    return result


def write_to_file(words, generators_count=2):
    path = Path('results', 'naive-sampling')
    new_index = max(
        map(lambda s: int(str(s)[len(str(path)) + len(f'{generators_count}-generators-') + 1:-len('.txt')]), path.glob(f'{generators_count}-generators-*')), 
        default = 0
    ) + 1
    with open(str(path.joinpath(f'{generators_count}-generators-{new_index}.txt')), 'w') as f:
        for w in words:
            f.write(word_to_string(w) + '\n')
