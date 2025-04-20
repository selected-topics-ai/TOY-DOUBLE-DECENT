import torch
import numpy as np

from tqdm import tqdm
from src.model import ToyNN
from matplotlib import pyplot as plt

def visualize_fractional_dimenstions(T, weights_dict, train_hiddens_dict):

    w_dims_dict = {}
    for t, W in tqdm(weights_dict.items()):
        w_dims_dict[t] = ToyNN.dimensionality(torch.tensor(W, dtype=torch.float32)).numpy()

    hidden_dims = {}
    for t, hidden in train_hiddens_dict.items():
        hidden_tensor = torch.tensor(hidden, dtype=torch.float32)
        hidden_dims[t] = (ToyNN.dimensionality(hidden_tensor).numpy())

    import random

    point_deviations = [random.uniform(-0.2, 0.2) for _ in range(max(10_000, max(T)))]
    point_deviations = np.array(point_deviations)

    plt.figure(figsize=(10, 5))

    plt.plot(T, [2 / t for t in T], linestyle='--', linewidth=0.5, color='gray', label='Hidden size / dataset size')

    for t in T:
        points = point_deviations[0: 10_000] + t
        plt.scatter(points, w_dims_dict[t], color='blue', alpha=0.3, label="Features" if t == min(T) else "")
        points = point_deviations[0: t] + t
        plt.scatter(points, hidden_dims[t], color='red', alpha=0.3, label="Hidden" if t == min(T) else "")

    plt.axvspan(0, 750, color='whitesmoke', alpha=0.3)
    plt.axvspan(750, 7500, color='lightgray', alpha=0.3)
    plt.axvspan(7500, 100000, color='darkgray', alpha=0.3)

    plt.text(15, 0.6, 'Small datasets', va='center', rotation='horizontal')
    plt.text(1250, 0.6, 'Middle regime', va='center', rotation='horizontal')
    plt.text(15000, 0.6, 'Large datasets', va='center', rotation='horizontal')

    plt.legend()
    plt.title("Fractional dimension of features and tranining examples")
    plt.xlabel('Dataset size')
    plt.xscale('log')
    plt.tight_layout()
    plt.show()