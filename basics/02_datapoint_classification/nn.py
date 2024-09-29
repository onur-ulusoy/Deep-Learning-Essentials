import numpy as np
import pickle, os

# Neural network for desired architecture, created using only numpy
class NN:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.W1 = np.random.rand(input_size, hidden1_size)*0.01
        self.b1 = np.random.rand(1, hidden1_size)
        self.W2 = np.random.rand(hidden1_size, hidden2_size)*0.01
        self.b2 = np.random.rand(1, hidden2_size)
        self.W3 = np.random.rand(hidden2_size, output_size)*0.01
        self.b3 = np.random.rand(1, output_size)

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

        #dz3 means dL/dz3 that is derivative of z3 wrt. loss/cost function
        dz3 = y_pred - y  # Difference between predicted probability and true label
        #print(y.T)

        self.dw3 = np.matmul(self.a2.T, dz3) / m
        self.db3 = np.sum(dz3, axis=0, keepdims=True)

        """ print(dz3.shape)
        print(self.dw3.shape)
        print(self.db3.shape) """

        dz2 = np.matmul(dz3, self.W3.T) * self.relu_derivative(self.a2)  # Derivative of ReLU
        self.dw2 = np.matmul(self.a1.T, dz2) / m
        self.db2 = np.sum(dz2, axis=0, keepdims=True)

        """ print(dz2.shape)
        print(self.dw2.shape)
        print(self.db2.shape) """

        dz1 = np.matmul(dz2, self.W2.T) * self.relu_derivative(self.a1)  # Derivative of ReLU
        self.dw1 = np.matmul(X.T, dz1) / m
        self.db1 = np.sum(dz1, axis=0, keepdims=True)

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




