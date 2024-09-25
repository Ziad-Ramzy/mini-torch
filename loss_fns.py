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


custom_criterion = CrossEntropy()

# Sample data for testing
# Here we assume 3 classes (for example), change as needed
num_classes = 3
batch_size = 10

# Random input logits and one-hot encoded targets
# Raw output from the model
input_logits = torch.randn(batch_size, num_classes)
targets = torch.zeros(batch_size, num_classes)
targets[torch.arange(batch_size), torch.randint(
    0, num_classes, (batch_size,))] = 1  # Random one-hot targets

# Compute loss using the custom CrossEntropy
custom_loss = custom_criterion(input_logits, targets)

# Compute loss using PyTorch's built-in CrossEntropyLoss
# Note: CrossEntropyLoss expects raw logits and class indices, not one-hot
targets_indices = targets.argmax(dim=1)  # Convert one-hot to indices
torch_loss = nn.CrossEntropyLoss()(input_logits, targets_indices)

# Print the losses
print(f"Custom CrossEntropy Loss: {custom_loss.item(
)}\n--------------------------\nPyTorch CrossEntropy Loss: {torch_loss.item()}")
