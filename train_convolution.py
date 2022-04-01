LENGTH = 50
EPOCHS = 100
BATCH_SIZE = 64
TRAIN_SIZE  = 10 ** 5
PER_EPOCH = (TRAIN_SIZE + BATCH_SIZE - 1) // BATCH_SIZE
LEARNING_RATE = 10 ** (-5)


from models.classificators import ConvolutionNormalClosureClassificator
from core.generators import from_free_group, from_normal_closure, from_choice, from_uniform_choice, uniform_hyperbolic_length
from core.free_group import is_from_normal_closure
from data.utils import RandomFreeGroupDataset, center_preprocess_word as preprocess, is_from_normal_closure_label as evaluate_label
from torch.nn import BCELoss
from torch.optim import Adam

labels = [0b00, 0b01, 0b10, 0b11]
gens = [
    from_free_group(length_distribution=uniform_hyperbolic_length(LENGTH)),
    from_normal_closure([[1]], length_distribution=uniform_hyperbolic_length(LENGTH)), 
    from_normal_closure([[2]], length_distribution=uniform_hyperbolic_length(LENGTH)),
    from_normal_closure([[1, 2]], length_distribution=uniform_hyperbolic_length(LENGTH))
]

trainees = []
for label in labels[1:]:
    model = ConvolutionNormalClosureClassificator(initial_length = LENGTH, up_samplings = 5, linear_layers = [64, 32])
    model.train()
    gen_label = gens[label]
    gen_rest  = from_uniform_choice(*[
        gens[l] for l in set(labels) - set([label])
    ])

    trainees.append({
        'label'     : label,
        'model'     : model,
        'dataset'   : RandomFreeGroupDataset(
                        generator=from_choice((0.4, gen_label), (0.6, gen_rest)),
                        count=TRAIN_SIZE,
                        preprocess_word=preprocess(LENGTH), 
                        evaluate_label=evaluate_label(label)
                    ),
        'optimizer' : Adam(model.parameters(), lr = LEARNING_RATE),
        'criterion' : BCELoss(),
        'weights'   : [],
        'losses'    : [],
        'accuracies': []
    })


from torchmetrics import Accuracy

accuracy_metric = Accuracy(multiclass=False)


from torch import hstack, tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy import mean

for trainee in trainees:
    print(f'Label: {trainee["label"]}')
    for epoch in range(1, EPOCHS + 1):
        for X_, y_ in tqdm(DataLoader(trainee['dataset'], batch_size=BATCH_SIZE)):
            trainee['optimizer'].zero_grad()
            y_pred = trainee['model'](X_).squeeze()
            loss = trainee['criterion'](y_pred, y_)
            loss.backward()

            trainee['weights'].append(hstack(
                tuple(param.flatten() for param in trainee['model'].parameters())
            ).detach().numpy())
            
            # gradients.append(tensor(loss.grad))

            trainee['losses'].append(tensor(loss.item()).detach().numpy())

            trainee['accuracies'].append(accuracy_metric(y_pred, tensor(y_, dtype=int)).detach().numpy())
            
            trainee['optimizer'].step()

        print(f'Epoch {epoch:03d}, Accuracy: {mean(trainee["accuracies"][-PER_EPOCH:-1])}, Loss: {mean(trainee["losses"][-PER_EPOCH:-1])}')


def average_per_epoch(metrics):
    return [
        mean(metrics[epoch*PER_EPOCH : (epoch + 1)*PER_EPOCH]) for epoch in range(EPOCHS)
    ]


from pathlib import Path

p = Path('results', 'convolution_classificators')
p.mkdir(parents=True, exist_ok=True)


from numpy.linalg import norm
import matplotlib.pyplot as plt
from torch import save

iterations  = range(1, PER_EPOCH * EPOCHS + 1)
epochs      = range(1, EPOCHS + 1)

for trainee in trainees:
    weights_norm = list(map(lambda w: norm(w - trainee['weights'][-1]), trainee['weights'])) 
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))

    ax[0, 0].plot(iterations[:-1], weights_norm[:-1])
    ax[0, 0].title.set_text('Weights norm of iterations')
    ax[0, 0].set_yscale('log')
    ax[0, 0].grid()

    ax[0, 1].plot(epochs, average_per_epoch(weights_norm))
    ax[0, 1].title.set_text('Weights norm of epochs')
    ax[0, 1].set_yscale('log')
    ax[0, 1].grid()

    ax[1, 0].plot(iterations[:-1], trainee['losses'][:-1])
    ax[1, 0].title.set_text('Losses of iterations')
    ax[1, 0].set_yscale('log')
    ax[1, 0].grid()

    ax[1, 1].plot(epochs, average_per_epoch(trainee['losses']))
    ax[1, 1].title.set_text('Losses of epochs')
    ax[1, 1].set_yscale('log')
    ax[1, 1].grid()

    fig.savefig(str(p / f'results{trainee["label"]}.png'))

    with open(p / f'parameters{trainee["label"]}', 'wb') as f:
        save(model.state_dict(), f)
