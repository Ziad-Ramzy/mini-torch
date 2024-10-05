import torch
import torch.nn as nn
import mini_torch.core.Functional as F


# Sigmoid activation function: squashes input to a range between 0 and 1.
class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return 1/(1 + torch.exp(-x))


# ReLU activation function: returns input directly if positive, otherwise returns 0.

class RelU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.max(torch.tensor(0), x)


# Tanh activation function: squashes input to a range between -1 and 1.
# Clamp applied to avoid overflow/NaN errors during exponentiation.
class TanH(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = torch.clamp(x, -20, 20)
        exp_x = torch.exp(x)
        exp_neg_x = torch.exp(-x)
        return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)


# GeLU activation function: used in models like transformers, it's a smoother alternative to ReLU.
class GelU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        sqrt_term = torch.sqrt(torch.tensor(
            2 / torch.pi, dtype=x.dtype)) * (x + 0.044715 * torch.pow(x, 3))
        return 0.5 * x * (1 + F.tanh_fn(sqrt_term))


# LeakyReLU activation function: similar to ReLU, but allows a small, non-zero slope for negative values.
class LeakyRelU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.max(0.1 * x, x)


# Softmax function: normalizes input to a probability distribution (sums to 1).
class SoftMax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        exp_x = torch.exp(x)
        return torch.exp(x) / torch.sum(exp_x)


# Swish activation function: multiplies the input with the sigmoid of the input.
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * F.sigmoid_fn(x)
