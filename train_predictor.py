import sys
from argparse import ArgumentParser
from pathlib import Path

from torch import save
from torch.utils.data import DataLoader

from utils.transforms import Compose, ToTensor, FromSubgroupLabel, subgroup_by_label

import matplotlib.pyplot as plt
from utils.oracle import Oracle
from tqdm import tqdm
import numpy as np

def cli():
    parser = ArgumentParser()
    parser.add_argument('config', type=Path, nargs=1)
    return parser


def main(path_to_config):
    exec(path_to_config.read_text(), globals())

    label = label_fn()
    device = device_fn()
    model = model_fn().to(device)
    criterion = criterion_fn()
    oracle = Oracle(model, criterion)

    dataset = dataset_fn()

    optimizer = optimizer_fn(oracle)
    metrics = metrics_fn(oracle)
    scheduler = scheduler_fn(optimizer)

    epochs = epochs_fn()
    batch_size = batch_size_fn()
    base_path = base_path_fn()

    base_path.mkdir(exist_ok=True, parents=True)
    (base_path / 'config.py').write_text(path_to_config.read_text())

    print(f'Training label: {label}')
    model.train()

    try:
        for epoch in range(1, epochs):
            losses = []
            for batch in tqdm(DataLoader(dataset, shuffle=True, batch_size=batch_size)):
                batch = (batch[0].to(device), batch[1].to(device))
                optimizer.zero_grad()

                loss = criterion(model, batch)
                loss.backward()
                losses.append(loss.item())
                
                optimizer.step()

                for _, _, metric in metrics:
                    metric(batch)
            print(f'Epoch: {epoch:03d}', f'Loss mean: {np.mean(losses):.4e}', f'Loss variance: {np.std(losses):.4e}')
            scheduler.step()
    except KeyboardInterrupt:
        pass

    with open(base_path / 'weights', 'wb') as f:
        save(model.state_dict(), f)

    nrows, ncols = len(metrics), 2
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (ncols * 15, nrows * 10))

    per_epoch = dataset.__len__() // batch_size + 1

    for i, (name, scale, metric) in enumerate(metrics):
        values = metric.total()

        iterations = len(values)
        epochs = iterations // per_epoch

        axs[i, 0].grid()
        axs[i, 1].grid()
        axs[i, 0].plot(np.arange(1, iterations + 1), values)
        axs[i, 1].plot(
            np.arange(1, epochs + 1), 
            list([np.mean(values[i*per_epoch : (i + 1)*per_epoch]) for i in range(epochs)])
        )
        axs[i, 0].set_yscale(scale)
        axs[i, 1].set_yscale(scale)
        axs[i, 0].title.set_text(f'{name} of iterations')
        axs[i, 1].title.set_text(f'{name} of epochs')

    fig.savefig(str(base_path / 'metrics.png'))

if __name__ == "__main__":
    cli = cli()
    main(
        cli.parse_args(sys.argv[1:]).config[0]
    )
