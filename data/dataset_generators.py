from random import sample
from string import ascii_letters
from core.free_group import wu_numerator_subgroups, is_in_subgroup, is_in_intersection
from core.generators import default_generator_from_subgroup
from .utils import word_filename, subgroup_by_label
from pathlib import Path
from json import dump


def generate_dataset_by_class_distribution(generators_number = 2,
                                           approximate_count = 10 ** 6,
                                           label_distribution = {},
                                           generator_from_subgroup = default_generator_from_subgroup,
                                           length_config = {'length': 5},
                                           name = None):
    path = Path(
        'data', 'datasets',
        ''.join(sample(ascii_letters, 6)) if name is None or len(name) == 0 else name
    )
    path.mkdir(parents=True, exist_ok=True)

    undistributed_proba_unit = (1 - sum(label_distribution.values())) / (2 ** generators_number - len(label_distribution))
    for label in range(2 ** generators_number):
        if label not in label_distribution:
            label_distribution[label] = undistributed_proba_unit
    count_by_label = { label : int(proba * approximate_count) for (label, proba) in label_distribution.items() }

    idx = 0
    wu_subgroups = wu_numerator_subgroups(generators_number)
    for label in range(2 ** generators_number):
        generator = generator_from_subgroup(generators_number=generators_number, subgroup=subgroup_by_label(label), length_config=length_config)
        for _ in range(count_by_label[label]):
            word = next(generator)
            to_dump = { str(label) : is_in_subgroup(subgroup_by_label(label=label), word) for label in range(2 ** generators_number) }
            to_dump['word'], to_dump['y'] = word, is_in_intersection(wu_subgroups, word)
            with open(path / word_filename(idx) , 'w') as file:
                dump(to_dump, file)
            idx += 1
    return path
