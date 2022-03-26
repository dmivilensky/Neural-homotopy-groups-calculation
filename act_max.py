LENGTH = 50
EPOCHS = 100
LEARNING_RATE = 0.1


from torch.nn import Module, Conv1d, Linear, ReLU, Sigmoid, Sequential, Flatten, Unflatten

class BinaryEncoderConvolution(Module):

    def __init__(self, initial_length = 100, convolutions = [3, 3], linear_layers = [64]):
        super().__init__()
        pre_linear_length = (2 ** len(convolutions)) * (initial_length - 2 * len(convolutions))
        self.layers = Sequential(
                Unflatten(dim=1, unflattened_size=(1, initial_length)),
                *[Conv1d(in_channels=2**idx, out_channels=2 * (2**idx), kernel_size=kernel) for idx, kernel in enumerate(convolutions)],
                Flatten(),
                Sequential(Linear(pre_linear_length, linear_layers[0]), ReLU()),
                *[Sequential(Linear(l1, l2), ReLU()) for l1, l2 in zip(linear_layers, linear_layers[1:])],
                Linear(linear_layers[-1], 1),
                Sigmoid()
        )
    

    def forward(self, word):
        return self.layers(word)

from torch import load

model = BinaryEncoderConvolution(initial_length=LENGTH, convolutions=[3, 3, 3], linear_layers=[32, 16])
with open('parameters', 'rb') as f:
    model.load_state_dict(load(f))

model.eval()


def layer_hook(act_dict, layer_name):
    def hook(module, input, output):
        act_dict[layer_name] = output
    return hook


activation_dictionary = {}
layer_name = 'classifier_final'

model.layers[-2].register_forward_hook(layer_hook(activation_dictionary, layer_name))


from torch import randn, no_grad, tensor, float as tfloat, norm
from torch import sqrt, mul, mean
from torch.nn.functional import pad, relu

def preprocess_word(length):
    def helper(word):
        left_pad = (length - len(word)) // 2
        return pad(tensor(word, dtype=tfloat), (left_pad, length - len(word) - left_pad))
    return helper

from numpy import argmin, arange
from core.free_group import normalize

def postprocess_word(generators_number):
    symbols = arange(-generators_number, generators_number + 1)
    def helper(tens):
        return normalize(list(map(
            lambda v: symbols[argmin(abs(symbols - v))],
            tens.detach().numpy()
        )))
    return helper

from math import floor, ceil
from random import random

def postprocess_word_random(generators_number):
    def round_one(v):
        l, r = floor(v), ceil(v)
        return r if random() < v - l else l
    def helper(tens):
        return normalize(list(
            map(round_one, tens.detach().numpy())
        ))
    return helper


preproc  = preprocess_word(LENGTH)
postproc = postprocess_word_random(generators_number=2)


from core.generators import from_free_group

source = randn(LENGTH).unsqueeze(0)
sources = []
source.requires_grad_(True)

optimum = {'activation' : -float('inf'), 'input': source}

for epoch in range(1, EPOCHS + 1):
    source.requires_grad_(True)
    source.retain_grad()
    model(source)
    
    layer_out = activation_dictionary[layer_name]

    # close to integers
    loss_to_integers = -norm(source - preproc(postproc(source.squeeze())))

    # in range
    loss_not_empty = norm(source)
    loss_in_range = -norm(relu(4 - mul(source, source)))

    (layer_out[0][0] + loss_to_integers + loss_not_empty + loss_in_range).backward()

    source_grad = source.grad / (sqrt(mean(
                mul(source.grad, source.grad))) + 1e-5)

    print(source_grad)
    
    source = source + LEARNING_RATE * source.grad

    if optimum['activation'] < layer_out[0][0]:
        optimum['activation'] = layer_out[0][0]
        optimum['input'] = postproc(source.squeeze())

    sources.append(postproc(source.squeeze()))

#print(optimum['input'], sources)



from core.free_group import is_from_normal_closure, word_as_str

for source in sources:
    print(word_as_str(source), is_from_normal_closure([1], source))

print(word_as_str(optimum['input']), is_from_normal_closure([1], optimum['input']))
