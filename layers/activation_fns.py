import torch
import torch.nn as nn


def sigmoid_fn(x: torch.tensor):
    return 1/(1 - torch.exp(-x))


def ReLU_fn(x: torch.tensor):
    return torch.max(0, x)


def tanh_fn(x: torch.tensor):

    # Using torch.clamp for enhanced numerical stability prevention of NaNs, and controlled outputs.

    x = torch.clamp(x, -20, 20)
    exp_x = torch.exp(x)
    exp_neg_x = torch.exp(-x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)


def GeLU_fn(x: torch.tensor):

    sqrt_term = torch.sqrt(torch.tensor(
        2 / torch.pi, dtype=x.dtype)) * (x + 0.044715 * torch.pow(x, 3))
    return 0.5 * x * (1 + tanh_fn(sqrt_term))


def LeakyReLU_fn(x: torch.tensor):
    return torch.max(0.1 * x, x)


def softmax_fn(x: torch.tensor):
    exp_x = torch.exp(x)
    return torch.exp(x) / torch.sum(exp_x)


def swish_fn(x: torch.tensor):

    if x.dtype != torch.float32:
        x = x.float()
    return x * sigmoid_fn(x)
