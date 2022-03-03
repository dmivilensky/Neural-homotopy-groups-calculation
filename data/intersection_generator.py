from core.free_group import is_in_intersection, normalize, is_trivial, wu_numerator_subgroups


def generator_from_intersection(word_sampler, group_index):
    subgroups = wu_numerator_subgroups(group_index)
    while True:
        word = []
        while is_trivial(word) or not is_in_intersection(subgroups, word):
            word = normalize(next(word_sampler))
        yield word
