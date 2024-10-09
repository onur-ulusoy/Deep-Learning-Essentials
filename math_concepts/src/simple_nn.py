import numpy as np

# Simple neural network class to be used in educational documents in math_concepts directory
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        print("W1 shape:", self.W1.shape)
        print("W1:", self.W1)

        print("b1 shape:", self.b1.shape)
        print("b1:", self.b1)

        print("W2 shape:", self.W2.shape)
        print("W2:", self.W2)

        print("b2 shape:", self.b2.shape)
        print("b2:", self.b2)

        self.learning_rate = learning_rate

    def relu(self, Z):
        """ReLU activation function"""
        return np.maximum(0, Z)
    
    def relu_derivative(self, A):
        """Derivative of ReLU"""
        return (A > 0).astype(float)
    
    def softmax(self, Z):
        """Softmax activation function"""
        # To improve numerical stability, subtract the max from each row
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def forward_pass(self, X):
        """Perform forward propagation"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)  # Activation from hidden layer
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)  # Activation from output layer
        
        return self.a2
    
    def calculate_loss(self, y, y_pred):
        """Compute categorical cross-entropy loss"""
        # To avoid log(0), add a small epsilon inside the log
        epsilon = 1e-8
        loss = -np.mean(np.sum(y * np.log(y_pred + epsilon), axis=1))
        return loss
    
    def backward_pass(self, X, y, y_pred):
        """Perform backward propagation and update weights and biases"""
        m = y.shape[0]  # Number of samples
        
        # Compute derivative of loss w.r. to z2
        dz2 = y_pred - y  # (m x output_size)
        
        # Compute gradients for W2 and b2
        self.dw2 = np.dot(self.a1.T, dz2) / m  # (hidden_size x output_size)
        self.db2 = np.sum(dz2, axis=0, keepdims=True) / m  # (1 x output_size)
        
        # Compute derivative of loss w.r. to a1
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.a1)  # (m x hidden_size)
        
        # Compute gradients for W1 and b1
        self.dw1 = np.dot(X.T, dz1) / m  # (input_size x hidden_size)
        self.db1 = np.sum(dz1, axis=0, keepdims=True) / m  # (1 x hidden_size)
        
        # Update parameters
        self.update_params()
    
    def update_params(self):
        """Update weights and biases using gradient descent"""
        self.W2 -= self.learning_rate * self.dw2
        self.b2 -= self.learning_rate * self.db2
        self.W1 -= self.learning_rate * self.dw1
        self.b1 -= self.learning_rate * self.db1
    
    def train(self, X, y, epochs=1000):
        """Train the neural network"""
        for epoch in range(1, epochs + 1):
            # Forward pass
            y_pred = self.forward_pass(X)
            
            # Compute and print loss every 100 epochs
            if epoch % 100 == 0 or epoch == 1:
                loss = self.calculate_loss(y, y_pred)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
            
            # Backward pass
            self.backward_pass(X, y, y_pred)
    
    def predict(self, X):
        """Make predictions with the trained network"""
        y_pred = self.forward_pass(X)
        return np.argmax(y_pred, axis=1)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(435)
    
    # Define network architecture
    input_size = 3
    hidden_size = 4
    output_size = 2  # Number of classes
    learning_rate = 0.1
    epochs = 1
    
    # Initialize the neural network
    nn = SimpleNeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    
    # Create training data with 2 samples
    # Inputs: 2 samples, each with 3 integer features
    X_train = np.random.randint(0, 10, (2, input_size))
    
    # Outputs: 2 samples, each with 2 binary labels (one-hot encoded)
    # For simplicity, let's create random one-hot encoded labels
    y_train_indices = np.random.randint(0, output_size, 2)
    y_train = np.zeros((2, output_size))
    y_train[np.arange(2), y_train_indices] = 1
    
    print("Training Data:")
    print("Inputs:\n", X_train)
    print("Labels (One-Hot Encoded):\n", y_train)
    print("\nStarting training...\n")
    
    # Train the network
    nn.train(X_train, y_train, epochs)
    
    # Make predictions on the training data
    predictions = nn.predict(X_train)
    print("\nPredictions after training:")
    print(predictions)
