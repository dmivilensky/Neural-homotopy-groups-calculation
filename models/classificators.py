import torch.nn as nn
import torch


def Conv2dNormReLU(
    input_channels, output_channels, kernel_size = 3, padding = 0, stride = 1, need_norm = True, alpha = 0.1
):
    result = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size, padding = padding, stride = stride)
    )
    if need_norm:
        result.append(nn.BatchNorm2d(output_channels))
    if not alpha is None:
        result.append(nn.LeakyReLU(alpha))
    return result


def Conv1dNormActivation(
    input_channels,
    output_channels,
    kernel_size = 3,
    padding = 0,
    need_norm=True,
    activation = nn.LeakyReLU(0.1)
):
    result = nn.Sequential(
        nn.Conv1d(input_channels, output_channels, kernel_size = kernel_size, padding = padding)
    )
    if need_norm:
        result.append(nn.BatchNorm1d(output_channels))
    if not activation is None:
        result.append(activation)
    return result
 

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, bias = False):
        super().__init__()
        self.layer = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bias = bias,
        )

    def forward(self, batch):
        ret, _ = self.layer(batch)
        return ret


class LinearClassificator(nn.Module):
    def __sub_module__(self, input_length, output_length):
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_length, output_length),
            nn.ReLU()
        )

    def __init__(self, input_length, hidden_layers = [64, 32]):
        super().__init__()
        hidden_layers = [input_length] + hidden_layers
        self.layer = nn.Sequential(
            *[
                self.__sub_module__(input_length, output_length) for input_length, output_length in zip(hidden_layers, hidden_layers[1:])
            ],
            nn.Linear(hidden_layers[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, batch):
        return self.layer(batch)


class NormalClosureClassificator(nn.Module):
    def __init__(self, input_shape, feautre_extractors, hidden_layers = [64, 32]):
        super().__init__()
        stub_input = torch.zeros(input_shape).unsqueeze(0)
        self.feautre_extractors = feautre_extractors
        self.classificator      = LinearClassificator(sum(map(lambda ex: ex.forward(stub_input).flatten().size()[0], feautre_extractors)), hidden_layers)
        self.flatten            = nn.Flatten()

    def __for_each_extractor__(self, func):
        for extractor in self.feautre_extractors:
            func(extractor)

    def to(self, arg):
        for i, e in enumerate(self.feautre_extractors):
            self.feautre_extractors[i] = e.to(arg)
        return super().to(arg)
        
    def forward(self, batch):
        features = torch.hstack(list(map(
            lambda ex: self.flatten(ex(batch)), self.feautre_extractors
        )))
        return self.classificator(features)
        
