from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable
from json import load
from .utils import convert_to_tensor, word_filename


class FreeGroupDataset(Dataset):
    def __init_source__(self, source):
        if isinstance(source, Path):
            self.path = source
        else:
            raise ValueError('parameter `source` must be `Path`')


    def __check_source_structure__(self):
        if not self.path.is_dir():
            raise ValueError('given `path` is not a directory')
        def all_names():
            return map(lambda p: int(p.stem), self.path.glob('*.word'))
        s, m = sum(all_names()), max(all_names())
        if s != (m * (m + 1)) // 2:
            raise ValueError('in the given directory some files are missing')
        self.length = m + 1

        
    def __init__(self, source, data_formatter, desired_labels):
        super(FreeGroupDataset).__init__()
        self.data_formatter = data_formatter
        self.desired_labels = desired_labels
        self.__init_source__(source)
        self.__check_source_structure__()

    
    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        with open(self.path / word_filename(idx), 'r') as file:
            to_return = load(file)
        return self.data_formatter.word(to_return['word']), self.data_formatter.labels([to_return[str(t)] for t in self.desired_labels])