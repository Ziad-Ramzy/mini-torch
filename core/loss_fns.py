import torch
import torch.nn as nn


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


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.tensor, target: torch.tensor):
        assert input.shape == target.shape
        input = torch.softmax(input, dim=-1)
        input = torch.clamp(input, min=1e-7, max=1 - 1e-7)
        loss = -torch.sum(target * torch.log(input), dim=-1)
        return torch.mean(loss)
