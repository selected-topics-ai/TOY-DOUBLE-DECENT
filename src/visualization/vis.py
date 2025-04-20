import torch
import numpy as np
from matplotlib.figure import Figure

from tqdm import tqdm
from typing import Any
from matplotlib import pyplot as plt
from src.model import ToyNN

def _draw_in_ax(ax, array_2d: np.ndarray, color: str) -> None:
    ax.set_aspect('equal')
    ax.scatter(array_2d[:, 0], array_2d[:, 1], c=color, s=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(len(array_2d)):
        ax.plot([array_2d[i, 0], 0], [array_2d[i, 1], 0], color=color, linewidth=0.1)


def draw_mini_plots(checkpoints: list[int]):

    T = checkpoints

    fig, axs = plt.subplots(2, len(T), figsize=(len(T), 3))

    for i, t in enumerate(T):
        checkpoint = torch.load(f'../checkpoints/checkpoint_{t}.pth', weights_only=False)
        model = ToyNN(input_dim=10_000, hidden_dim=2)
        model.load_state_dict(checkpoint['model'])

        features = model.W.cpu().detach().numpy()
        hiddens = checkpoint['hiddens']
        loss = checkpoint['loss']

        column = axs[:, i]

        _draw_in_ax(column[0], features, 'blue')
        _draw_in_ax(column[1], hiddens, 'red')

    fig.text(0.05, 0.55, 'Features (Columns of W)', va='center', rotation='horizontal')
    fig.text(0.05, 0.20, 'Hidden vectors (Training set)', va='center', rotation='horizontal')

    fig.tight_layout(pad=2.0, h_pad=1.0, w_pad=2.0)

    fig.show()

if __name__ == '__main__':

    draw_mini_plots([3, 4])