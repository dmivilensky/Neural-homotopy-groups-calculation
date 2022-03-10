from pathlib import Path
from core.generators import ClosureGenerator, ChoiceGenerator
from data.utils import create_dataset, convert_word_pad_one_side, convert_labels, columns_of_subgroups
from data.free_group_dataset import FreeGroupDataset

name = '2-generators;balanced'

def dataset():
    path = Path('data', 'datasets', name)
    if not path.is_dir():
        gen1 = ChoiceGenerator(
            (ClosureGenerator(generators_number=2, length_config={'max-length': 50}), 0.7),
            (ClosureGenerator(generators_number=2, length_config={'max-length': 20}), 0.3)
        )
        gen2 = ChoiceGenerator(
            (ClosureGenerator(generators_number=2, length_config={'max-length': 50}, subgroup=[1]), 0.7),
            (ClosureGenerator(generators_number=2, length_config={'max-length': 20}, subgroup=[1]), 0.3)
        )
        create_dataset(
            generator=ChoiceGenerator((gen1, 0.5), (gen2, 0.5)).take(500_000),
            columns=columns_of_subgroups(0b01),
            path_or_name=path
        )
    return FreeGroupDataset(source=path, word_convert=convert_word_pad_one_side(50), labels_convert=convert_labels())

ds = dataset()

from torch.utils.data import DataLoader

for X_batch, y_batch in DataLoader(ds, shuffle=True, batch_size=10):
    print(X_batch, y_batch)
    break
