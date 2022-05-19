import torch
import numpy as np

from utils.datasets import RandomFreeGroupDataset
import utils.transforms as trans

import core.free_group as group
import core.generators as generators


LENGTH = 51
GENERATORS_NUMBER = 3
label = 0b001

length_distribution = generators.uniform_length(LENGTH)

other_generators = []
for other_label in [2 ** i for i in range(GENERATORS_NUMBER)] + [2 ** GENERATORS_NUMBER - 1]:
    if other_label != label:
        other_generators.append(generators.from_normal_closure(
            subgroup = trans.subgroup_by_label(other_label),
            generators_number = GENERATORS_NUMBER,
            length_distribution = length_distribution
        ))

free = generators.from_free_group(
    generators_number = GENERATORS_NUMBER,
    length_distribution = length_distribution
)

from_label = generators.from_normal_closure(
    subgroup = trans.subgroup_by_label(label),
    generators_number = GENERATORS_NUMBER,
    length_distribution = length_distribution
)

generator = generators.RandomChoiceGenerator([
    from_label,
    generators.RandomChoiceGenerator(
        other_generators + [free]
    )
])


dataset = RandomFreeGroupDataset(
    generator = generator,
    count     = 10 ** 5,
    transform_word = trans.Compose([
        trans.ToTensor(torch.long),
        trans.Pad(LENGTH, mode = 'center'),
        trans.CrossEncoder(GENERATORS_NUMBER),
        trans.ToTensor(torch.float64)
    ]),
    transform_label = trans.Compose([
        trans.ToTensor(torch.long),
        trans.FromSubgroupLabel(label)
    ]),
)

# ....
