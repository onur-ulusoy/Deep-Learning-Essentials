# NN_torch.py
import torch
import torch.nn as nn
import torch.optim as optim
import pickle, os

class NN_torch(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.01, l2_lambda = 0.02, dropout_p1=0.25, dropout_p2=0.25):
        super(NN_torch, self).__init__()
        
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda  # Regularization strength

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_p1)

        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_p2)

        self.fc3 = nn.Linear(hidden2_size, output_size)
        
        # Define optimizer and loss function
        #self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Initialize weights
        nn.init.uniform_(self.fc1.weight, -0.01, 0.01)
        nn.init.uniform_(self.fc2.weight, -0.01, 0.01)
        nn.init.uniform_(self.fc3.weight, -0.01, 0.01)
        
        # Initialize biases
        nn.init.uniform_(self.fc1.bias, 0, 1)
        nn.init.uniform_(self.fc2.bias, 0, 1)
        nn.init.uniform_(self.fc3.bias, 0, 1)
        
        print("fc1 weight shape:", self.fc1.weight.shape)
        print("fc1 bias shape:", self.fc1.bias.shape)
        print("fc2 weight shape:", self.fc2.weight.shape)
        print("fc2 bias shape:", self.fc2.bias.shape)
        print("fc3 weight shape:", self.fc3.weight.shape)
        print("fc3 bias shape:", self.fc3.bias.shape)
    
    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.relu1(z1)
        a1 = self.dropout1(a1)
        
        z2 = self.fc2(a1)
        a2 = self.relu2(z2)
        a2 = self.dropout2(a2)
        
        z3 = self.fc3(a2)
        a3 = z3  # Linear activation for output
        return a3
    
    def train_model(self, X_train, y_train, epochs=1000):
        # Convert NumPy arrays to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        print("y_train:",y_train[0])

        m = y_train.shape[0]

        self.train()

        for epoch in range(epochs):
            # Zero the gradients
            #self.optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.forward(X_train)
            #print(y_pred[0])
            loss = self.criterion(y_pred, y_train)
            
            # Compute L2 regularization (exclude biases)
            weights_sum = (
                self.fc1.weight.pow(2.0).sum() +
                self.fc2.weight.pow(2.0).sum() +
                self.fc3.weight.pow(2.0).sum()
            )
            # Scale the regularization term to match NumPy implementation
            loss = loss + (self.l2_lambda / (2 * m)) * weights_sum
            # Backward pass and optimization
            loss.backward()
            #self.optimizer.step()

            #update parameters
            with torch.no_grad():
                for param in self.parameters():
                    param -= self.learning_rate * param.grad
            
            self.zero_grad()
            
            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    # Save model's weights and biases
    def save_model(self, y_original_min, y_original_max):
        model_data = {
            'fc1_weight': self.fc1.weight.data,
            'fc1_bias': self.fc1.bias.data,
            'fc2_weight': self.fc2.weight.data,
            'fc2_bias': self.fc2.bias.data,
            'fc3_weight': self.fc3.weight.data,
            'fc3_bias': self.fc3.bias.data,
            'y_original_min': y_original_min,
            'y_original_max': y_original_max
        }

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'trained_model.pkl')

        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model weights, biases, and y_original_min/max saved successfully at {file_path}.")