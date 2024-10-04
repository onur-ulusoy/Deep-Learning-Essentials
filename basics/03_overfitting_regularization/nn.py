import numpy as np
import pickle, os
import torch
# Neural network for desired architecture, created using only numpy
class NN:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate = 0.01):
        self.learning_rate = learning_rate

        ### basic initialization
        self.W1 = np.random.rand(input_size, hidden1_size)*0.01
        self.W2 = np.random.rand(hidden1_size, hidden2_size)*0.01
        self.W3 = np.random.rand(hidden2_size, output_size)*0.01

        self.b1 = np.random.rand(1, hidden1_size)
        self.b2 = np.random.rand(1, hidden2_size)
        self.b3 = np.random.rand(1, output_size)

        print("W1 shape: ", self.W1.shape)
        print("b1 shape: ", self.b1.shape)
        print("W2 shape: ", self.W2.shape)
        print("b2 shape: ", self.b2.shape)
        print("W3 shape: ", self.W3.shape)
        print("b3 shape: ", self.b3.shape)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)
            
    # Forward Propagation
    def forward_pass(self, X):
        self.z1 = np.matmul(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)  # First hidden layer activation (ReLU)

        self.z2 = np.matmul(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)  # Second hidden layer activation (ReLU)

        self.z3 = np.matmul(self.a2, self.W3) + self.b3
        self.a3 = self.z3  # Output layer linear activation

        return self.a3
    
    # Backward Propagation
    def backward_pass(self, X, y, y_pred):
        m = y.shape[0]

        # Compute derivative of loss w.r.t z3
        dz3 = y_pred - y  # Shape: (m, output_size)
        #print("dz3[0]:",dz3[0])
        # Gradients for W3 and b3
        self.dw3 = np.matmul(self.a2.T, dz3) / m  # Shape: (hidden2_size, output_size)
        self.db3 = np.sum(dz3, axis=0, keepdims=True) / m  # Shape: (1, output_size)

        # Backpropagate to second hidden layer (apply derivative on z2)
        dz2 = np.matmul(dz3, self.W3.T) * self.relu_derivative(self.z2)
        self.dw2 = np.matmul(self.a1.T, dz2) / m  # Shape: (hidden1_size, hidden2_size)
        self.db2 = np.sum(dz2, axis=0, keepdims=True) / m  # Shape: (1, hidden2_size)

        # Backpropagate to first hidden layer (apply derivative on z1)
        dz1 = np.matmul(dz2, self.W2.T) * self.relu_derivative(self.z1)
        self.dw1 = np.matmul(X.T, dz1) / m  # Shape: (input_size, hidden1_size)
        self.db1 = np.sum(dz1, axis=0, keepdims=True) / m  # Shape: (1, hidden1_size)

        self.update_params()

    def update_params(self):
        self.W3 -= self.learning_rate * self.dw3
        self.W2 -= self.learning_rate * self.dw2
        self.W1 -= self.learning_rate * self.dw1

        self.b3 -= self.learning_rate * self.db3
        self.b2 -= self.learning_rate * self.db2
        self.b1 -= self.learning_rate * self.db1
    
    def calculate_loss(self, y, y_pred):
        """
        Calculate the Mean Squared Error (MSE) loss between true values and predictions.
        """
        loss = np.mean((y - y_pred) ** 2)
        return loss
    
    def train_model(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass to get predictions
            y_pred = self.forward_pass(X)
            #print(y[0], y_pred[0])
            # Backward pass to update weights
            self.backward_pass(X, y, y_pred)
            #print("db3[0]",self.db3)
            # Every 100 epochs, calculate and print the loss
            if epoch % 100 == 0:
                loss = self.calculate_loss(y, y_pred)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
                #print(self.dw2)

    def save_model(self):
        # Convert weights and biases to PyTorch tensors if they aren't already
        model_data = {
            # I made weights transposed so that it could be compatible with test script (torch)
            'fc1_weight': torch.tensor(self.W1.T, dtype=torch.float32) if not isinstance(self.W1, torch.Tensor) else self.W1,
            'fc1_bias': torch.tensor(self.b1, dtype=torch.float32) if not isinstance(self.b1, torch.Tensor) else self.b1,
            'fc2_weight': torch.tensor(self.W2.T, dtype=torch.float32) if not isinstance(self.W2, torch.Tensor) else self.W2,
            'fc2_bias': torch.tensor(self.b2, dtype=torch.float32) if not isinstance(self.b2, torch.Tensor) else self.b2,
            'fc3_weight': torch.tensor(self.W3.T, dtype=torch.float32) if not isinstance(self.W3, torch.Tensor) else self.W3,
            'fc3_bias': torch.tensor(self.b3, dtype=torch.float32) if not isinstance(self.b3, torch.Tensor) else self.b3
        }

        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'trained_model.pkl')

        # Save to a file (pickle format)
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model weights and biases saved successfully at {file_path}.")


