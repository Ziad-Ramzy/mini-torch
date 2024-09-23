import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, input: int, output: int):
        super().__init__()
        self.x = torch.nn.Parameter(torch.randn(input, output))
        self.b = torch.nn.Parameter(torch.zeros(output))

    def forward(self, A: torch.tensor):
        if A.shape[1] != self.x.shape[0]:
            raise RuntimeError(f"Shape mismatch: Input shape {
                               A.shape} can't broadcast with the weights {self.x.shape}")
        y = A @ self.x + self.b
        return y
