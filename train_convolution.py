LENGTH = 50
EPOCHS = 100
BATCH_SIZE = 64
TRAIN_SIZE  = 10 ** 5
TEST_SIZE   = 10 ** 2
PER_EPOCH = (TRAIN_SIZE + BATCH_SIZE - 1) // BATCH_SIZE
LEARNING_RATE = 10 ** (-5)


from torch.nn import Module, Conv1d, Linear, ReLU, Sigmoid, Sequential, Flatten, Unflatten, BCELoss
from torch.optim import Adam

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


model = BinaryEncoderConvolution(initial_length=LENGTH, convolutions=[3, 3, 3], linear_layers=[32, 16])
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = BCELoss()


from data.utils import RandomFreeGroupDataset
from torch.nn.functional import pad
from torch import tensor, float as tfloat

def preprocess_word(length):
    def helper(word):
        left_pad = (length - len(word)) // 2
        return pad(tensor(word, dtype=tfloat), (left_pad, length - len(word) - left_pad))
    return helper


from core.free_group import is_from_normal_closure

def evaluate_label(generator):
    def helper(word):
        return tensor(is_from_normal_closure(generator, word), dtype=tfloat)
    return helper


from core.generators import from_normal_closure, from_free_group, uniform_hyperbolic_length

gens = [
    from_normal_closure([[1]], length_distribution=uniform_hyperbolic_length(LENGTH)), 
    from_normal_closure([[2]], length_distribution=uniform_hyperbolic_length(LENGTH)),
    from_normal_closure([[1, 2]], length_distribution=uniform_hyperbolic_length(LENGTH)),
    from_free_group(length_distribution=uniform_hyperbolic_length(LENGTH))
]


from core.generators import from_uniform_choice

train_dataset = RandomFreeGroupDataset(
    generator=from_uniform_choice(*gens),
    count=TRAIN_SIZE,
    preprocess_word=preprocess_word(LENGTH),
    evaluate_label=evaluate_label([1])
)

test_dataset = RandomFreeGroupDataset(
    generator=from_uniform_choice(*gens),
    count=100,
    preprocess_word=preprocess_word(LENGTH),
    evaluate_label=evaluate_label([1])
)


from torchmetrics import Accuracy

accuracy_metric = Accuracy(multiclass=False)


from torch import hstack, tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy import mean as npmean

model.train()
weights, gradients, losses, accuracies = [], [], [], []

for epoch in range(1, EPOCHS + 1):
    for X_, y_ in tqdm(DataLoader(train_dataset, batch_size=BATCH_SIZE)):

        optimizer.zero_grad()
        y_pred = model(X_).squeeze()
        loss = criterion(y_pred, y_)
        loss.backward()

        weights.append(hstack(
            tuple(param.flatten() for param in model.parameters())
        ).detach().numpy())
        
        # gradients.append(tensor(loss.grad))

        losses.append(tensor(loss.item()).detach().numpy())

        accuracies.append(accuracy_metric(y_pred, tensor(y_, dtype=int)).detach().numpy())
        
        optimizer.step()

    print(f'Epoch {epoch:03d}, Accuracy: {npmean(accuracies[-PER_EPOCH:-1])}, Loss: {npmean(losses[-PER_EPOCH:-1])}')


from numpy import linalg

def average_per_epoch(metrics):
    return [
        npmean(metrics[epoch*PER_EPOCH : (epoch + 1)*PER_EPOCH]) for epoch in range(EPOCHS)
    ]


from torch import norm

weights_norm = list(map(lambda w: linalg.norm(w - weights[-1]), weights))

iterations  = range(1, len(weights_norm) + 1)
epochs      = range(1, EPOCHS + 1)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))

ax[0, 0].plot(iterations, weights_norm)
ax[0, 0].title.set_text('Weights norm of iterations')
ax[0, 0].set_yscale('log')
ax[0, 0].grid()

ax[0, 1].plot(epochs, average_per_epoch(weights_norm))
ax[0, 1].title.set_text('Weights norm of epochs')
ax[0, 1].set_yscale('log')
ax[0, 1].grid()

ax[1, 0].plot(iterations, losses)
ax[1, 0].title.set_text('Losses of iterations')
ax[1, 0].set_yscale('log')
ax[1, 0].grid()

ax[1, 1].plot(epochs, average_per_epoch(losses))
ax[1, 1].title.set_text('Losses of epochs')
ax[1, 1].set_yscale('log')
ax[1, 1].grid()

fig.savefig('results.png')
plt.close(fig)   

from torch import save, load

with open('parameters', 'wb') as f:
    save(model.state_dict(), f)

new_model = BinaryEncoderConvolution(initial_length=LENGTH, convolutions=[3, 3, 3], linear_layers=[32, 16])
with open('parameters', 'rb') as f:
    new_model.load_state_dict(load(f))

new_model.eval()

for x, y in DataLoader(test_dataset, batch_size=TEST_SIZE):
    y_pred = new_model(x).squeeze()
    print(accuracy_metric(y_pred, tensor(y, dtype=int)))
