import torch
import einops
from torch import nn

class ToyNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim:int, device="mps"):
        super(ToyNN, self).__init__()
        self.W = nn.Parameter(torch.randn((input_dim, hidden_dim), device=device))
        nn.init.xavier_normal_(self.W)
        self.b = nn.Parameter(torch.zeros(input_dim, device=device))
    
    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.matmul(X, self.W)
        out = torch.matmul(h, self.W.t())
        out = out + self.b
        out = torch.relu(out)

        return out, h.detach()

    @staticmethod
    def dimensionality(X: torch.Tensor):
        squared_norms = einops.reduce(X.pow(2), "... batch d_hidden -> ... batch", "sum")
        data_normed = X / X.norm(dim=-1, keepdim=True)
        interference = einops.einsum(
            data_normed, X, "... batch_i d_hidden, ... batch_j d_hidden -> ... batch_i batch_j"
        )
        polysemanticity = einops.reduce(interference.pow(2), "... batch_i batch_j -> ... batch_i", "sum")
        assert squared_norms.shape == polysemanticity.shape
        return squared_norms / polysemanticity