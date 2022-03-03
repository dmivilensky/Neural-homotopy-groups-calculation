from data.free_group_generator import generator_from_free_group_bounded
from data.closure_generator import generator_from_normal_closure_v2 as generator_from_closure
from core.free_group import normalize, is_in_subgroup, is_in_intersection, wu_numerator_subgroups
from itertools import combinations
from pandas import DataFrame


def subgroup_by_class(label):
    result = []
    for idx in range(label.bit_length()):
        if label & (1 << (idx)):
            result.append(idx + 1)
    return result


def create_dataset_by_class_distribution(group_index = 2,
                                        approximate_length = 20,
                                        approximate_count = 10 ** 6, 
                                        class_distribution = None):
    if class_distribution is None:
        class_distribution = {}
    undistributed_proba_unit = (1 - sum(class_distribution.values())) / (group_index + 1 - len(class_distribution))
    for label in range(0, 2 ** group_index):
        if label not in class_distribution:
            class_distribution[label] = undistributed_proba_unit
    result, subgroups = [], wu_numerator_subgroups(group_index)
    for label in range(0, 2 ** group_index):
        if label == 0:
            generator = generator_from_free_group_bounded(group_index, max_length=(1 + 0.2) * approximate_length)
        else:
            generator = generator_from_closure(subgroup_by_class(label), group_index, lam=approximate_length)
        for _ in range(int(approximate_count * class_distribution[label])):
            word = normalize(next(generator))
            result.append([word] + [is_in_subgroup(subgroup_by_class(label), word) for label in range(1, 2 ** group_index)] + [is_in_intersection(subgroups, word)])

    columns = ['word'] + [f'is-in-<{",".join(map(str, subgroup_by_class(label)))}>' for label in range(1, 2 ** group_index)] + ['is-in-intersection']
    return DataFrame(result, columns = columns)
