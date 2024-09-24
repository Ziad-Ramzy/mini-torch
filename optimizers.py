from __future__ import print_function
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.nn as nn
from typing import Iterable, Dict, Any


class StochasticGradientDescent(Optimizer):
    def __init__(self, params: Iterable[torch.Tensor] | Iterable[Dict[str, Any]], lr=0.001) -> None:
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.params = params
        self.lr = lr

    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                with torch.no_grad():
                    param -= self.lr * param.grad.data


class Adam(Optimizer):
    def __init__(self, params: Iterable[torch.Tensor] | Iterable[Dict[str, Any]], lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        defaults = {'lr': lr, 'betas': betas, 'eps': eps}
        super().__init__(params, defaults)
        self.state = {}

    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state.setdefault(param, {})
                if 'm' not in state:
                    state['m'] = torch.zeros_like(param)
                if 'v' not in state:
                    state['v'] = torch.zeros_like(param)
                if 't' not in state:
                    state['t'] = 0

                state['t'] += 1
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                grad = param.grad.data
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * grad ** 2
                m_hat = state['m'] / (1 - beta1 ** state['t'])
                v_hat = state['v'] / (1 - beta2 ** state['t'])
                param -= lr * m_hat / (torch.sqrt(v_hat) + eps)


def get_datasets(n_train=1024, n_valid=1024,
                 input_shape=[3, 32, 32], target_shape=[],
                 n_classes=None):
    """Construct and return random number datasets"""
    train_x = torch.randn([n_train] + input_shape)
    valid_x = torch.randn([n_valid] + input_shape)
    if n_classes is not None:
        train_y = torch.randint(
            n_classes, [n_train] + target_shape, dtype=torch.long)
        valid_y = torch.randint(
            n_classes, [n_valid] + target_shape, dtype=torch.long)
    else:
        train_y = torch.randn([n_train] + target_shape)
        valid_y = torch.randn([n_valid] + target_shape)
    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(valid_x, valid_y)
    return train_dataset, valid_dataset, {}


class SimpleNN(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(
            16 * (input_shape[1] // 2) * (input_shape[2] // 2), n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x


def test_gradient_descent():
    # Parameters
    n_train = 1024
    n_valid = 1024
    input_shape = [3, 32, 32]
    n_classes = 10
    lr = 0.001
    num_epochs = 100

    # Generate datasets
    train_dataset, valid_dataset, _ = get_datasets(
        n_train=n_train, n_valid=n_valid, input_shape=input_shape, n_classes=n_classes)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = SimpleNN(input_shape, n_classes)
    criterion = nn.CrossEntropyLoss()  # For classification tasks
    optimizer = StochasticGradientDescent(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# Run the test
if __name__ == "__main__":
    test_gradient_descent()
