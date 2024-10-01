""" This is the script I created to resolve problem in numpy nn, comparing and experimenting step by step with the torch one """
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


""" Forward Pass Comparison """
print("\nForward Pass Comparison")

# Forward pass in PyTorch
output_torch = model.forward(X_torch)
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
z1_numpy = np.matmul(X_numpy, W1.T) + b1
a1_numpy = relu(z1_numpy)

z2_numpy = np.matmul(a1_numpy, W2.T) + b2
a2_numpy = relu(z2_numpy)

z3_numpy = np.matmul(a2_numpy, W3.T) + b3
a3_numpy = sigmoid(z3_numpy)

print("NumPy forward pass result:", a3_numpy)

# Compare the PyTorch and NumPy outputs
difference = np.abs(output_torch.detach().numpy() - a3_numpy)
print("Difference between PyTorch and NumPy results:", difference)




""" Backward Pass Comparison """
print("\nBackward Pass Comparison")

# Define binary cross-entropy loss function in NumPy
def binary_cross_entropy(y_true, y_pred):
    # Binary Cross-Entropy Loss
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# Backward pass in NumPy
def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def backward_pass_numpy(X, y, y_pred, W1, W2, W3, a1, a2):
    m = y.shape[0]

    # Output layer error
    dz3 = y_pred - y  # dL/dz3
    dw3 = np.matmul(a2.T, dz3) / m  # dL/dW3
    db3 = np.sum(dz3, axis=0, keepdims=True)  # dL/db3

    # Second hidden layer error
    dz2 = np.matmul(dz3, W3) * relu_derivative(a2)  # dL/dz2
    dw2 = np.matmul(a1.T, dz2) / m  # dL/dW2
    db2 = np.sum(dz2, axis=0, keepdims=True)  # dL/db2

    # First hidden layer error
    dz1 = np.matmul(dz2, W2) * relu_derivative(a1)  # dL/dz1
    dw1 = np.matmul(X.T, dz1) / m  # dL/dW1
    db1 = np.sum(dz1, axis=0, keepdims=True)  # dL/db1

    return dw1, db1, dw2, db2, dw3, db3

# True labels for the input
y_true = np.array([[1]])

# Compute the loss in NumPy
loss_numpy = binary_cross_entropy(y_true, a3_numpy)
print("NumPy Loss:", loss_numpy)

# Perform the backward pass in NumPy
dw1_numpy, db1_numpy, dw2_numpy, db2_numpy, dw3_numpy, db3_numpy = backward_pass_numpy(X_numpy, y_true, a3_numpy, W1, W2, W3, a1_numpy, a2_numpy)

# Now perform the backward pass in PyTorch
criterion = torch.nn.BCELoss()

# Compute the loss in PyTorch
y_true_torch = torch.tensor(y_true, dtype=torch.float32)
loss_torch = criterion(output_torch, y_true_torch)
print("PyTorch Loss:", loss_torch.item())

# Perform backpropagation in PyTorch
loss_torch.backward()

# Extract gradients from PyTorch
dw1_torch = model.fc1.weight.grad.detach().numpy()
db1_torch = model.fc1.bias.grad.detach().numpy()
dw2_torch = model.fc2.weight.grad.detach().numpy()
db2_torch = model.fc2.bias.grad.detach().numpy()
dw3_torch = model.fc3.weight.grad.detach().numpy()
db3_torch = model.fc3.bias.grad.detach().numpy()

# Compare losses
loss_diff = np.abs(loss_numpy - loss_torch.item())
print("Difference in Loss:", loss_diff)

# Compare gradients for each layer

# Layer 1 gradients comparison
dw1_diff = np.linalg.norm(dw1_numpy.T - dw1_torch)
db1_diff = np.linalg.norm(db1_numpy - db1_torch)
print("Difference in dw1:", dw1_diff)
print("Difference in db1:", db1_diff)

# Layer 2 gradients comparison
dw2_diff = np.linalg.norm(dw2_numpy.T - dw2_torch)
db2_diff = np.linalg.norm(db2_numpy - db2_torch)
print("Difference in dw2:", dw2_diff)
print("Difference in db2:", db2_diff)

# Layer 3 gradients comparison
dw3_diff = np.linalg.norm(dw3_numpy.T - dw3_torch)
db3_diff = np.linalg.norm(db3_numpy - db3_torch)
print("Difference in dw3:", dw3_diff)
print("Difference in db3:", db3_diff)
