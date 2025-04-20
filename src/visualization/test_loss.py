import numpy as np
from matplotlib import pyplot as plt

def visualize_test_loss(T: list[int], test_losses_dict: dict[int, list[float]]) -> None:

    plt.figure(figsize=(10, 3))
    plt.plot(T, np.array([value[0] for value in test_losses_dict.values()]) * 10_000, marker='o', linewidth=1,
             color='gray')

    plt.yticks([1.0, 1.01, 1.02])
    plt.ylim([0.995, 1.025])
    plt.xlabel("Dataset size")
    plt.title("Test loss. Peaks in the middle regime")

    plt.axvspan(0, 750, color='whitesmoke', alpha=0.3)
    plt.axvspan(750, 7500, color='lightgray', alpha=0.3)
    plt.axvspan(7500, 100000, color='darkgray', alpha=0.3)

    plt.text(15, 1.02, 'Small datasets', va='center', rotation='horizontal')
    plt.text(1250, 1.02, 'Middle regime', va='center', rotation='horizontal')
    plt.text(15000, 1.02, 'Large datasets', va='center', rotation='horizontal')

    plt.axhline(y=1, color='grey', linestyle='--', linewidth=0.5)

    plt.ylabel("Test loss")
    plt.xscale('log')
    plt.show()