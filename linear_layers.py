
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import loss_fns


class Linear(nn.Module):
    def __init__(self, input: int, output: int):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(input, output))
        torch.nn.init.kaiming_uniform_(self.w)
        self.b = torch.nn.Parameter(torch.zeros(output))

    def forward(self, A: torch.tensor):
        if A.shape[1] != self.x.shape[0]:
            raise RuntimeError(f"Shape mismatch: Input shape {
                               A.shape} can't broadcast with the weights {self.x.shape}")
        y = A @ self.w + self.b
        return y


class Bilinear(nn.Module):
    def __init__(self, input1: int, input2: int, output: int):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(output, input1, input2))
        torch.nn.init.kaiming_uniform_(self.w)
        self.b = torch.nn.Parameter(torch.zeros(output))

    def forward(self, X1: torch.Tensor, X2: torch.Tensor):
        output = torch.einsum("bi,oij,bj->bo", X1, self.w, X2)
        """ 
            b: batch size
            i: first input shape
            j: second input shape
            o: output shape
        """
        output += self.b
        return output


# Hyperparameters

# Hyperparameters
batch_size = 64
num_classes = 10  # For MNIST
epochs = 20
learning_rate = 0.001

# Load MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images
])

# Download and load the training and test datasets
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = Bilinear(input1=28*28, input2=28*28, output=num_classes)

# Define loss function and optimizer
criterion = loss_fns.CrossEntropy()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in tqdm(range(epochs)):
    model.train()  # Set model to training mode
    running_loss = 0.0  # Variable to track loss

    for images, labels in train_loader:
        # Flatten the images to match input dimensions
        images_flat = images.view(images.size(0), -1)  # (batch_size, 28*28)

        # One-hot encode labels for the custom loss function
        targets = torch.zeros(images.size(0), num_classes).to(images.device)
        # Create one-hot encoded targets
        targets.scatter_(1, labels.view(-1, 1), 1)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images_flat, images_flat)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    # Print average loss for the epoch
    average_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}')

# Testing Loop
correct = 0
total = 0

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for images, labels in test_loader:
        images_flat = images.view(images.size(0), -1)  # Flatten
        targets = torch.zeros(images.size(0), num_classes).to(images.device)
        # Create one-hot encoded targets
        targets.scatter_(1, labels.view(-1, 1), 1)

        outputs = model(images_flat, images_flat)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
