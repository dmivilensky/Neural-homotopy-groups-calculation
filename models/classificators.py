from torch.nn import Module, Conv1d, Linear, ReLU, Sigmoid, Sequential, Flatten, Unflatten, BatchNorm1d


class ConvolutionFeatureExtractor(Module):
    def __sub_module__(self, input_channels):
        return Sequential(
            Conv1d(in_channels = input_channels, out_channels = 2 * input_channels, kernel_size = 3),
            BatchNorm1d(2 * input_channels),
            ReLU()
        )

    def __init__(self, initial_length = 50, up_samplings = 5):
        super().__init__()
        self.layer = Sequential(
            *[self.__sub_module__(2 ** idx) for idx in range(up_samplings)]
        )
        self.flatten_size = 2 ** (up_samplings) * (initial_length - 2 * up_samplings)
    

    def forward(self, batch):
        return self.layer(batch)


class LinearClassificator(Module):
    def __sub_module__(self, input_length, output_length):
        return Sequential(
            Linear(input_length, output_length),
            ReLU()
        )

    def __init__(self, input_length, linear_layers = [64, 32], classes = 1):
        super().__init__()
        linear_layers = [input_length] + linear_layers
        self.layer = Sequential(
            *[
                self.__sub_module__(input_length, output_length) for input_length, output_length in zip(linear_layers, linear_layers[1:])
            ],
            Linear(linear_layers[-1], classes),
            Sigmoid()
        )

    def forward(self, batch):
        return self.layer(batch)


class ConvolutionNormalClosureClassificator(Module):
    def __init__(self, initial_length = 50, up_samplings = 5, linear_layers = [64, 32]):
        super().__init__()
        
        self.unflatten          = Unflatten(dim = 1, unflattened_size = (1, initial_length))
        self.feature_extractor  = ConvolutionFeatureExtractor(initial_length, up_samplings)
        self.classificator      = LinearClassificator(self.feature_extractor.flatten_size, linear_layers)
        self.flatten            = Flatten()

    def forward(self, batch):
        batch = self.unflatten(batch)
        batch = self.feature_extractor(batch)
        batch = self.flatten(batch)
        batch = self.classificator(batch)
        return batch
        