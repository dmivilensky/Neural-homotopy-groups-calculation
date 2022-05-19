from json import dumps
from itertools import islice
from tqdm import tqdm
import torch


def generate_dataset(generator, count, transform_word, transform_label, path):
    dataset = []
    for word in tqdm([next(generator) for _ in range(count)]):
        data = transform_word(word)
        label = transform_label(word)
        dataset.append({'data' : data, 'label' : label})
    path.write_text(dumps(dataset))


from pathlib import Path
import utils.transforms as tf
import core.generators as gens


LENGTH = 51
GENERATORS_NUMBER = 3
label = 3

length_distribution = gens.uniform_length(LENGTH)

other_generators = []
for other_label in [2 ** i for i in range(GENERATORS_NUMBER)] + [2 ** GENERATORS_NUMBER - 1]:
    if other_label != label:
        other_generators.append(gens.from_normal_closure(
            subgroup = tf.subgroup_by_label(other_label),
            generators_number = GENERATORS_NUMBER,
            length_distribution = length_distribution
        ))

free = gens.from_free_group(
    generators_number = GENERATORS_NUMBER,
    length_distribution = length_distribution
)

from_label = gens.from_normal_closure(
    subgroup = tf.subgroup_by_label(label),
    generators_number = GENERATORS_NUMBER,
    length_distribution = length_distribution
)

generator =  gens.RandomChoiceGenerator([
    from_label,
    gens.RandomChoiceGenerator(
        other_generators + [free]
    )
])

transform_word = tf.Compose([
    tf.ToTensor(torch.long),
    tf.Pad(length = LENGTH, mode = 'random'),
    lambda v: v.flatten().detach().tolist()
])

transform_label = tf.Compose([
    tf.FromSubgroupLabel(label),
    tf.ToTensor(torch.long),
    lambda v: v.flatten().detach().tolist()
])

path = Path('datasets', 'random-padding', f'label-{label}')
path.mkdir(parents=True, exist_ok=True)


generate_dataset(generator, 10 ** 5, transform_word, transform_label, path / 'data.json')
