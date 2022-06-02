import torch

LENGTH = 51
GENERATORS_NUMBER = 3

global label_fn
def label_fn():
    return 0b010


global base_path_fn
from pathlib import Path
def base_path_fn():
    return Path('results', 'classificators', 'lstm', 'random-padding', 'hidden-size-128', 'train', f'label-{label_fn()}')


import torch.nn as nn
global model_fn
from models.classificators import NormalClosureClassificator, LSTMFeatureExtractor

def model_fn():
    return NormalClosureClassificator(
        input_shape = (LENGTH, GENERATORS_NUMBER),
        feautre_extractors = [
            LSTMFeatureExtractor(GENERATORS_NUMBER, 128, num_layers = 1)
        ],
        hidden_layers=[16]
    ).to(torch.float64)


import core.generators as generators
import utils.transforms as trans
def generator_fn(label):
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

    return generators.RandomChoiceGenerator([
        from_label,
        generators.RandomChoiceGenerator(
            other_generators + [free]
        )
    ])


def count_fn():
    return 10 ** 5


import utils.transforms as trans
def transform_word_fn():
    return trans.Compose([
        trans.ToTensor(torch.float64),
        trans.Pad(length = LENGTH, mode = 'center'),
        trans.CrossEncoder(GENERATORS_NUMBER),
        trans.ToTensor(torch.float64),
    ])


def transform_label_fn(label):
    return trans.Compose([
        trans.FromSubgroupLabel(label),
        trans.ToTensor(torch.long)
    ])


global dataset_fn
from utils.datasets import RandomFreeGroupDataset, FromFileFreeGroupDataset
def dataset_fn():
    '''
    return RandomFreeGroupDataset(
        generator = generator_fn(label_fn()),
        count     = count_fn(),
        transform_word = transform_word_fn(),
        transform_label = transform_label_fn(label_fn()),
    )
    '''
    
    return FromFileFreeGroupDataset(
        path = Path('datasets', 'random-padding', f'label-{label_fn()}') / 'data.json',
        transform = trans.Compose([
            trans.ToTensor(torch.long),
            trans.CrossEncoder(GENERATORS_NUMBER),
            lambda v: v.permute(1, 0).to(torch.float64)
        ]),
        label = trans.ToTensor(torch.long)
    )
    
    

global criterion_fn
from torch.nn import BCELoss
def criterion_fn():
    criterion = BCELoss()
    def func(model, batch):
        return criterion(model(batch[0]).squeeze(), batch[1].to(torch.float64).squeeze())
    return func


global optimizer_fn
from torch.optim import Adam
def optimizer_fn(oracle):
    return Adam(oracle._model.parameters(), lr=0.001)


global scheduler_fn
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR, LinearLR
def scheduler_fn(optimizer):
    class Stub:
        def __init__(self):
            pass
        def step(self):
            pass

    return LinearLR(optimizer, start_factor=0.1, total_iters = 50)


global metrics_fn
from utils.train_metrics import WeightsNormTrainMetric, GradientNormTrainMetric, PerformanceMetric, LossTrainMetric
from torchmetrics import Accuracy
def metrics_fn(oracle):
    return [
        ('weights norm', 'log', WeightsNormTrainMetric(oracle, offset = 5000)),
        ('gradient norm', 'log', GradientNormTrainMetric(oracle)),
        ('accuracy', 'linear', PerformanceMetric(oracle, Accuracy(multiclass = False))),
        ('loss', 'log', LossTrainMetric(oracle))
    ]


global epochs_fn
def epochs_fn():
    return 150


global batch_size_fn
def batch_size_fn():
    return 256


global device_fn
def device_fn():
    return torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
