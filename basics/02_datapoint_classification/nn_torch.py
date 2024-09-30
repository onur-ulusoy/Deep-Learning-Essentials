import torch
import torch.nn as nn
import torch.optim as optim
import pickle, os

# Neural network for desired architecture, created using PyTorch
class NN_torch(nn.modules):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.01):
        super(NN_torch.self).__init__()
        self.learning_rate = learning_rate

        # Xavier initialization is default in nn.Linear, so we don't need to manually initialize.
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)

        # Using ReLU activation and Sigmoid for the output layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Binary Cross-Entropy Loss for binary classification
        self.criterion = nn.BCELoss()