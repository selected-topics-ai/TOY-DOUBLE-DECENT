import os
import torch
import numpy as np

from src.model import ToyNN

from matplotlib import pyplot as plt

def visualize_norms(T: list[int], path="../checkpoints"):

    norms = {}

    for files, _, file_names in os.walk(path):
        for file_name in file_names:
            file_path = os.path.join(files, file_name)
            model = ToyNN(input_dim=10_000, hidden_dim=2)
            checkpoint = torch.load(file_path, weights_only=False)
            model.load_state_dict(checkpoint["model"])

            t = int(file_name.replace(".pth", "").replace("checkpoint_", ""))
            norms[t] = {
                "W": model.W.detach().cpu().norm(),
                "b": model.b.detach().cpu().norm(),
            }

    w_norms = np.array([norms[t]["W"] for t in T])
    b_norms = np.array([norms[t]["b"] for t in T])

    plt.figure(figsize=(10, 5))

    plt.plot(T, w_norms, marker='o', markersize=5, label='||W||', linewidth=0.5, color='gray')
    plt.plot(T, b_norms, marker='o', markersize=5, label='||b||', linestyle='--', linewidth=0.5, color='gray')

    plt.axvspan(0, 750, color='whitesmoke', alpha=0.3)
    plt.axvspan(750, 7500, color='lightgray', alpha=0.3)
    plt.axvspan(7500, 100000, color='darkgray', alpha=0.3)

    plt.yscale('log')
    plt.xscale('log')

    plt.text(15, 0.2, 'Small datasets', va='center', rotation='horizontal')
    plt.text(1250, 0.2, 'Middle regime', va='center', rotation='horizontal')
    plt.text(15000, 0.2, 'Large datasets', va='center', rotation='horizontal')

    plt.tight_layout()
    plt.legend()
    plt.title("Parameter Norms")
    plt.xlabel('Dataset Size')
    plt.ylabel('Norm')
    plt.show()
