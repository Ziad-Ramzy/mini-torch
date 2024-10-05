import torch

# Sigmoid activation function: squashes input to a range between 0 and 1.


def sigmoid_fn(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))

# ReLU activation function: returns input directly if positive, otherwise returns 0.


def relu_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.tensor(0, dtype=x.dtype), x)

# Tanh activation function: squashes input to a range between -1 and 1.
# Clamp applied to avoid overflow/NaN errors during exponentiation.


def tanh_fn(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -20, 20)  # Clamping to avoid overflow
    exp_x = torch.exp(x)
    exp_neg_x = torch.exp(-x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

# GeLU activation function: used in models like transformers, it's a smoother alternative to ReLU.


def gelu_fn(x: torch.Tensor) -> torch.Tensor:
    sqrt_term = torch.sqrt(torch.tensor(
        2 / torch.pi, dtype=x.dtype)) * (x + 0.044715 * torch.pow(x, 3))
    return 0.5 * x * (1 + tanh_fn(sqrt_term))

# Leaky ReLU activation function: similar to ReLU, but allows a small, non-zero slope for negative values.


def leaky_relu_fn(x: torch.Tensor, negative_slope: float = 0.1) -> torch.Tensor:
    # Return x if x is larger than 0
    # else, return negative_slope * x
    return torch.where(x > 0, x, negative_slope * x)

# Softmax function: normalizes input to a probability distribution (sums to 1).


def softmax_fn(x: torch.Tensor) -> torch.Tensor:
    exp_x = torch.exp(x - torch.max(x))  # For numerical stability
    return exp_x / torch.sum(exp_x, dim=0)

# Swish activation function: multiplies the input with the sigmoid of the input.


def swish_fn(x: torch.Tensor) -> torch.Tensor:
    return x * sigmoid_fn(x)
