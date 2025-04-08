import torch

def generate_batch(n_batch: int, n_features: int, sparsity_prob: float, device="mps") -> torch.Tensor:
    mask = torch.rand(n_batch, n_features) > sparsity_prob
    random_values = torch.rand(n_batch, n_features)
    sparse_vectors = random_values * mask
    norms = torch.norm(sparse_vectors, p=2, dim=1, keepdim=True)
    norms[norms == 0] = 1.0
    normalized_vectors = sparse_vectors / norms
    return normalized_vectors.to(device)
