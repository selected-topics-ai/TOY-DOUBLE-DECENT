import torch
import torch.nn as nn

from tqdm import tqdm
from src.generate_data import generate_batch
from src.model import ToyNN

def evaluate(T: list[int], path:str="../checkpoints"):

    weights_dict = {}
    train_hiddens_dict = {}
    test_losses_dict = {}
    test_hiddens_dict = {}
    criterion = nn.MSELoss()

    for i, t in tqdm(enumerate(T)):
        test_losses = []
        test_hiddens = []

        checkpoint = torch.load(f'{path}/checkpoint_{t}.pth', weights_only=False)
        model = ToyNN(input_dim=10_000, hidden_dim=2)
        model.load_state_dict(checkpoint['model'])

        test_batch = generate_batch(
            n_batch=1000,
            n_features=10_000,
            sparsity_prob=0.999,
        )

        model.eval()
        with torch.inference_mode():
            output, hiddens = model(test_batch)
            test_loss = criterion(output, test_batch)
            test_losses.append(test_loss.item())
            test_hiddens.append(hiddens)

        test_losses_dict[t] = test_losses
        test_hiddens_dict[t] = test_hiddens
        weights_dict[t] = model.W.detach().cpu().numpy()
        train_hiddens_dict[t] = checkpoint['hiddens']

    return weights_dict, train_hiddens_dict, test_losses_dict, test_hiddens_dict