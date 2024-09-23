import torch
import torch.nn as nn


# Sigmoid activation function: squashes input to a range between 0 and 1.
def sigmoid_fn(x: torch.tensor):
    return 1/(1 - torch.exp(-x))


# ReLU activation function: returns input directly if positive, otherwise returns 0.
def ReLU_fn(x: torch.tensor):
    return torch.max(0, x)


# Tanh activation function: squashes input to a range between -1 and 1.
# Clamp applied to avoid overflow/NaN errors during exponentiation.
def tanh_fn(x: torch.tensor):
    x = torch.clamp(x, -20, 20)
    exp_x = torch.exp(x)
    exp_neg_x = torch.exp(-x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)


# GeLU activation function: used in models like transformers, it's a smoother alternative to ReLU.
def GeLU_fn(x: torch.tensor):
    sqrt_term = torch.sqrt(torch.tensor(
        2 / torch.pi, dtype=x.dtype)) * (x + 0.044715 * torch.pow(x, 3))
    return 0.5 * x * (1 + tanh_fn(sqrt_term))


# LeakyReLU activation function: similar to ReLU, but allows a small, non-zero slope for negative values.
def LeakyReLU_fn(x: torch.tensor):
    return torch.max(0.1 * x, x)


# Softmax function: normalizes input to a probability distribution (sums to 1).
def softmax_fn(x: torch.tensor):
    exp_x = torch.exp(x)
    return torch.exp(x) / torch.sum(exp_x)


# Swish activation function: multiplies the input with the sigmoid of the input.
def swish_fn(x: torch.tensor):
    if x.dtype != torch.float32:
        # Convert to float if not already, for numerical stability.
        x = x.float()
    return x * sigmoid_fn(x)
