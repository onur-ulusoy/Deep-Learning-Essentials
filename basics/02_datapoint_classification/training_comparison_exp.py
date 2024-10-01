import torch
import torch.nn as nn
import numpy as np

# PyTorch Neural Network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 18)
        self.fc2 = nn.Linear(18, 4)
        self.fc3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.relu(z1)
        z2 = self.fc2(a1)
        a2 = self.relu(z2)
        z3 = self.fc3(a2)
        a3 = self.sigmoid(z3)

        return a3

# Initialize the model
model = NeuralNetwork()

# Define input
torch.manual_seed(42)
X_torch = torch.randn(1, 2)  # 1 sample, 2 features

# Forward pass in PyTorch
output_torch = model(X_torch)
print("PyTorch output:", output_torch)

# Extracting weights and biases from PyTorch model
W1 = model.fc1.weight.detach().numpy()  # Shape: (18, 2)
b1 = model.fc1.bias.detach().numpy()    # Shape: (18,)
W2 = model.fc2.weight.detach().numpy()  # Shape: (4, 18)
b2 = model.fc2.bias.detach().numpy()    # Shape: (4,)
W3 = model.fc3.weight.detach().numpy()  # Shape: (1, 4)
b3 = model.fc3.bias.detach().numpy()    # Shape: (1,)

# Convert input from torch to numpy
X_numpy = X_torch.detach().numpy()

# Implement forward pass in NumPy
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# NumPy forward pass using PyTorch weights and biases
z1_numpy = np.dot(X_numpy, W1.T) + b1
a1_numpy = relu(z1_numpy)

z2_numpy = np.dot(a1_numpy, W2.T) + b2
a2_numpy = relu(z2_numpy)

z3_numpy = np.dot(a2_numpy, W3.T) + b3
a3_numpy = sigmoid(z3_numpy)

print("NumPy forward pass result:", a3_numpy)

# Compare the PyTorch and NumPy outputs
difference = np.abs(output_torch.detach().numpy() - a3_numpy)
print("Difference between PyTorch and NumPy results:", difference)
