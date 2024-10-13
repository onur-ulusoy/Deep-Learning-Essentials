import numpy as np
import pickle, os
import torch
import torch.nn as nn

# Neural network for desired architecture, created using only numpy
class NN:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate = 0.01):
        self.learning_rate = learning_rate
        """ basic initialization
        self.W1 = np.random.rand(input_size, hidden1_size)*0.01
        self.W2 = np.random.rand(hidden1_size, hidden2_size)*0.01
        self.W3 = np.random.rand(hidden2_size, output_size)*0.01 """

        """ self.b1 = np.random.rand(1, hidden1_size)
        self.b2 = np.random.rand(1, hidden2_size)
        self.b3 = np.random.rand(1, output_size)

        # applying Xavier or He initialization:
        self.W1 = self.he_init(input_size, hidden1_size)
        self.W2 = self.he_init(hidden1_size, hidden2_size)
        self.W3 = self.xavier_init(hidden2_size, output_size)  # For sigmoid layer """

        #torch initialization
        torch.manual_seed(41)
        nn_model = SimpleNeuralNetwork(input_size, hidden1_size, hidden2_size, output_size)

        # Retrieve the weights and biases as NumPy arrays
        self.W1 = nn_model.fc1.weight.detach().numpy().T
        self.b1 = nn_model.fc1.bias.detach().numpy().reshape(1, hidden1_size)

        self.W2 = nn_model.fc2.weight.detach().numpy().T
        self.b2 = nn_model.fc2.bias.detach().numpy().reshape(1, hidden2_size)

        self.W3 = nn_model.fc3.weight.detach().numpy().T
        self.b3 = nn_model.fc3.bias.detach().numpy().reshape(1, output_size)



        print("W1 shape: ", self.W1.shape)
        print("b1 shape: ", self.b1.shape)
        print("W2 shape: ", self.W2.shape)
        print("b2 shape: ", self.b2.shape)
        print("W3 shape: ", self.W3.shape)
        print("b3 shape: ", self.b3.shape)

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def leaky_relu(self, z, alpha=0.01):
        return np.where(z > 0, z, z * alpha)

    def leaky_relu_derivative(self, z, alpha=0.01):
        return np.where(z > 0, 1, alpha)
    
    def xavier_init(self, size_in, size_out):
        return np.random.randn(size_in, size_out) * np.sqrt(1 / size_in)

    def he_init(self, size_in, size_out):
        return np.random.randn(size_in, size_out) * np.sqrt(2 / size_in)
            
    # Forward Propagation
    def forward_pass(self, X):
        self.z1 = np.matmul(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)  # First hidden layer activation (ReLU)

        self.z2 = np.matmul(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)  # Second hidden layer activation (ReLU)

        self.z3 = np.matmul(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)  # Output layer activation (Sigmoid)

        """ print("z1 shape: ", self.z1.shape)
        print("z2 shape: ", self.z2.shape)
        print("z3 shape: ", self.z3.shape) """

        return self.a3
    
    # Backward Propagation
    def backward_pass(self, X, y, y_pred):
        m = y.shape[0]
        """ print(m) """

        #dz3 means dL/dz3 that is derivative of loss/cost function wrt z3. 
        """ expression simplifies after applying chain rule, refer 02_backward_propagation_gd.md in math concepts. 
        We have done softmax there but appearantly it is valid also for sigmoid """
        dz3 = y_pred - y  # Difference between predicted probability and true label

        self.dw3 = np.matmul(self.a2.T, dz3) / m
        self.db3 = np.sum(dz3, axis=0, keepdims=True) / m

        """ print(dz3.shape)
        print(self.dw3.shape)
        print(self.db3.shape) """

        dz2 = np.matmul(dz3, self.W3.T) * self.relu_derivative(self.a2)  # Derivative of ReLU
        self.dw2 = np.matmul(self.a1.T, dz2) / m
        self.db2 = np.sum(dz2, axis=0, keepdims=True) / m

        """ print(dz2.shape)
        print(self.dw2.shape)
        print(self.db2.shape) """

        dz1 = np.matmul(dz2, self.W2.T) * self.relu_derivative(self.a1)  # Derivative of ReLU
        self.dw1 = np.matmul(X.T, dz1) / m
        self.db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.update_params()

    def update_params(self):
        self.W3 -= self.learning_rate * self.dw3
        self.W2 -= self.learning_rate * self.dw2
        self.W1 -= self.learning_rate * self.dw1

        self.b3 -= self.learning_rate * self.db3
        self.b2 -= self.learning_rate * self.db2
        self.b1 -= self.learning_rate * self.db1
    
    def calculate_loss(self, y, y_pred):
        # Binary Cross-Entropy Loss
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass to get predictions
            y_pred = self.forward_pass(X)
            #print(y_pred[0])
            
            # Backward pass to update weights
            self.backward_pass(X, y, y_pred)

            # Every 100 epochs, calculate and print the loss
            if epoch % 100 == 0:
                loss = self.calculate_loss(y, y_pred)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
                #print(self.dw2)

    def save_model(self):
        # Save only model weights and biases
        model_data = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3
        }

        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'trained_model.pkl')

        # Save to a file (pickle format)
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model weights and biases saved successfully at {file_path}.")

## torch nn class just to retrieve initial weights and biases to get the same results with torch
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()

        # Use nn.Linear to initialize weights and biases automatically
        self.fc1 = nn.Linear(input_size, hidden1_size)   # First hidden layer
        self.fc2 = nn.Linear(hidden1_size, hidden2_size) # Second hidden layer
        self.fc3 = nn.Linear(hidden2_size, output_size)  # Output layer


