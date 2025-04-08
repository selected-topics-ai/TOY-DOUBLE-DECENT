import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from .model import ToyNN
from .generate_data import generate_batch

from tqdm import tqdm


def train(model: ToyNN,
          optimizer: optim.Optimizer,
          criterion,
          train_dataset: torch.Tensor,
          n_updates: int,
          warmup_steps: int,
          scheduler: lrs.LRScheduler,
          warmup_scheduler: lrs.LRScheduler) -> tuple[ToyNN, torch.Tensor, torch.Tensor]:

    model.train()

    final_hidden_tensors = None
    final_loss = None

    for i in tqdm(range(n_updates)):
        optimizer.zero_grad()
        outputs, hidden = model(train_dataset)
        loss = criterion(outputs, train_dataset)
        loss.backward()
        optimizer.step()
        if i < warmup_steps:
            warmup_scheduler.step()
        else:
            scheduler.step()

        final_hidden_tensors = hidden
        final_loss = loss.item()

    return model, final_hidden_tensors, final_loss


if __name__ == '__main__':

    T = [3, 5, 6, 10, 15, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]

    hidden_dim = 2
    n_features = 10_000
    sparsity_prob = 0.999
    device = "mps"
    l2_reg = 1e-2
    lr = 1e-3
    n_updates = 50000
    warmup_steps = 2500

    for t in T:
        print(f"Training with dataset size {t}")
        dataset = generate_batch(n_batch=t,
                                 n_features=n_features,
                                 sparsity_prob=sparsity_prob,
                                 device=device)

        model = ToyNN(input_dim=n_features, hidden_dim=hidden_dim, device=device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=l2_reg)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=n_updates,
                                                         eta_min=0)

        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer,
                                                       start_factor=0.01,
                                                       end_factor=1.0,
                                                       total_iters=warmup_steps)

        fitted_model, hiddens, loss = train(model=model,
              optimizer=optimizer,
              criterion=criterion,
              train_dataset=dataset,
              n_updates=n_updates,
              warmup_steps=warmup_steps,
              scheduler=scheduler,
              warmup_scheduler=warmup_scheduler)

        torch.save({
            "model": fitted_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hiddens": hiddens.cpu().detach().numpy(),
            "loss": loss
        }, f"./checkpoints/checkpoint_{t}.pth")
