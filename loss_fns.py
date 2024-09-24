import torch
import torch.nn as nn
from layers import activation_fns


class MSELOSS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.tensor, target: torch.tensor):
        assert input.shape == target.shape
        return torch.mean(torch.pow(input - target, 2))


class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.tensor, target: torch.tensor):
        assert input.shape == target.shape
        epsilon = 1e-12
        # Clamp the input to avoid log(0)
        input = torch.clamp(input, epsilon, 1 - epsilon)
        return -torch.mean((target * torch.log(input)) + ((1 - target) * torch.log(1 - input)))


# Initialize the criterion
criterion = BinaryCrossEntropy()


x = activation_fns.sigmoid_fn(torch.randn([10, 1]))

targets = torch.randint(0, 2, [10, 1]).float()

loss_fn = criterion(x, targets)

# Compute loss using PyTorch's built-in BCELoss
loss_torch = nn.BCELoss()(x, targets)

print(loss_fn.item(), "\n--------------------------\n", loss_torch.item())
