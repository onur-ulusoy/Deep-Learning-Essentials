# NN_torch.py
import torch
import torch.nn as nn
import torch.optim as optim
import pickle, os

class NN_torch(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.01):
        super(NN_torch, self).__init__()
        
        self.learning_rate = learning_rate
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, output_size)
        
        # Define optimizer and loss function
        #self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        """ # Initialize weights
        nn.init.uniform_(self.fc1.weight, -0.01, 0.01)
        nn.init.uniform_(self.fc2.weight, -0.01, 0.01)
        nn.init.uniform_(self.fc3.weight, -0.01, 0.01)
        
        # Initialize biases
        nn.init.uniform_(self.fc1.bias, 0, 1)
        nn.init.uniform_(self.fc2.bias, 0, 1)
        nn.init.uniform_(self.fc3.bias, 0, 1) """
        
        print("fc1 weight shape:", self.fc1.weight.shape)
        print("fc1 bias shape:", self.fc1.bias.shape)
        print("fc2 weight shape:", self.fc2.weight.shape)
        print("fc2 bias shape:", self.fc2.bias.shape)
        print("fc3 weight shape:", self.fc3.weight.shape)
        print("fc3 bias shape:", self.fc3.bias.shape)
    
    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.relu1(z1)
        
        z2 = self.fc2(a1)
        a2 = self.relu2(z2)
        
        z3 = self.fc3(a2)
        a3 = z3  # Linear activation for output
        return a3
    
    def train_model(self, X_train, y_train, epochs=1000):
        # Convert NumPy arrays to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        print("y_train:",y_train[0])
        for epoch in range(epochs):
            # Zero the gradients
            #self.optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.forward(X_train)
            #print(y_pred[0])
            loss = self.criterion(y_pred, y_train)
            
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
    def save_model(self):
        model_data = {
            'fc1_weight': self.fc1.weight.data,
            'fc1_bias': self.fc1.bias.data,
            'fc2_weight': self.fc2.weight.data,
            'fc2_bias': self.fc2.bias.data,
            'fc3_weight': self.fc3.weight.data,
            'fc3_bias': self.fc3.bias.data
        }

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'trained_model_torch.pkl')

        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model weights and biases saved successfully at {file_path}.")
