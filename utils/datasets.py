from torch.utils.data import Dataset
from json import loads


class RandomFreeGroupDataset(Dataset):
    def __init__(self, generator, count, transform_word, transform_label):
        self.generator = generator
        self.count = count
        self.transform_word = transform_word
        self.transform_label = transform_label

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        word = next(self.generator)
        return self.transform_word(word), self.transform_label(word)


class FromFileFreeGroupDataset(Dataset):
    def __init__(self, path, transform, label):
        self.data = loads(path.read_text())
        self.transform_word = transform
        self.transform_label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return self.transform_word(row['data']), self.transform_label(row['label'])
