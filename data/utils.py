from torch.utils.data import Dataset


class RandomFreeGroupDataset(Dataset):
    def __init__(self, generator, count, preprocess_word, evaluate_label):
        self.generator = generator
        self.count = count
        self.preprocess_word = preprocess_word
        self.evaluate_label = evaluate_label

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        word = next(self.generator)
        return self.preprocess_word(word), self.evaluate_label(word)
