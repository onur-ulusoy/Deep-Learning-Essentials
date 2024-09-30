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

    # Forward propagation
    def forward(self, X):
        z1 = self.fc1(X)
        self.a1 = self.relu(z1)

        z2 = self.fc2(self.a1)
        self.a2 = self.relu(z2)

        z3 = self.fc3(self.a2)
        self.a3 = self.sigmoid(z3)

    
    def train_model(self, X_train, y_train, epochs = 1000):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        # forward pass
        y_pred = self.forward()

        #calculate loss
        loss = self.criterion(y_pred, y_train)

        # Backward pass
        loss.backward()

        # Update params
        with torch.no_grad():
            for param in self.parameters():
                param -= self.learning_rate * param.grad

        self.zero_grad()