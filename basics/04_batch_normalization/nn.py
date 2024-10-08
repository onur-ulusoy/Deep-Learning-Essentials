import numpy as np

# Neural network for desired architecture, created using only numpy
class NN:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.01, l2_lambda=0.02, dropout_p1=0.25, dropout_p2=0.25, momentum=0.9):
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_p1 = dropout_p1
        self.dropout_p2 = dropout_p2
        self.momentum = momentum  # Momentum for running statistics
        self.epsilon = 1e-5    # Small constant for numerical stability

        ### Basic initialization
        self.W1 = np.random.uniform(-0.01, 0.01, (input_size, hidden1_size)).astype(np.float32)
        self.W2 = np.random.uniform(-0.01, 0.01, (hidden1_size, hidden2_size)).astype(np.float32)
        self.W3 = np.random.uniform(-0.01, 0.01, (hidden2_size, output_size)).astype(np.float32)

        self.b1 = np.random.uniform(0, 1, (1, hidden1_size)).astype(np.float32)
        self.b2 = np.random.uniform(0, 1, (1, hidden2_size)).astype(np.float32)
        self.b3 = np.random.uniform(0, 1, (1, output_size)).astype(np.float32)

        # BatchNorm Parameters for First Hidden Layer
        """ @gamma: learnable parameter for multiplying (scaling) normalized values to adjust the output to be in optimal variance
            @beta: learnable parameter for shifting normalized values to adjust the output to be in optimal mean
            
            running variables are used in test in order to find the normalized value, that reflect common mean and var for entire training
        """
        self.gamma1 = np.ones((1, hidden1_size), dtype=np.float32)
        self.beta1 = np.zeros((1, hidden1_size), dtype=np.float32)
        self.running_mean1 = np.zeros((1, hidden1_size), dtype=np.float32)
        self.running_var1 = np.ones((1, hidden1_size), dtype=np.float32)

        # BatchNorm Parameters for Second Hidden Layer
        self.gamma2 = np.ones((1, hidden2_size), dtype=np.float32)
        self.beta2 = np.zeros((1, hidden2_size), dtype=np.float32)
        self.running_mean2 = np.zeros((1, hidden2_size), dtype=np.float32)
        self.running_var2 = np.ones((1, hidden2_size), dtype=np.float32)

        print("W1 shape: ", self.W1.shape)
        print("b1 shape: ", self.b1.shape)
        print("gamma1 shape:", self.gamma1.shape)
        print("beta1 shape:", self.beta1.shape)
        print("W2 shape: ", self.W2.shape)
        print("b2 shape: ", self.b2.shape)
        print("gamma2 shape:", self.gamma2.shape)
        print("beta2 shape:", self.beta2.shape)
        print("W3 shape: ", self.W3.shape)
        print("b3 shape: ", self.b3.shape)

    def relu(self, z):
        return np.maximum(0, z)
    
    # BatchNorm Forward Function
    def batch_norm_forward(self, x, gamma, beta, running_mean, running_var, training=True):
        if training:
            # Find mean and variance of all elements in given matrix x, outputs of neurons before activation
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)

            # Normalize x to have mean of 0 and variance of 1
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)

            # Scale and shift to the optimal range
            out = gamma * x_normalized + beta

            # Update Running Mean and Var along the way in every batch
            running_mean[:] = self.momentum * running_mean + (1 - self.momentum) * batch_mean
            running_var[:] = self.momentum * running_var + (1 - self.momentum) * batch_var

            # Store for backward pass
            self.x_normalized = x_normalized
            self.batch_var = batch_var
            self.x_centered = x - batch_mean

        else: # Test mode
            # Normalize x to have mean of 0 and variance of 1 using running statistics because basically there is no batch mean and var in test
            x_normalized = (x - running_mean) / np.sqrt(running_var + self.epsilon)
            # Scale and shift to the optimal range
            out = gamma * x_normalized + beta

        return out
    
    # Forward Propagation
    def forward_pass(self, X, training=False):
        self.z1 = np.matmul(X, self.W1) + self.b1  # Linear transformation

        # BatchNorm hidden layer 1
        self.a1 = self.batch_norm_forward(self.z1, self.gamma1, self.beta1, self.running_mean1, self.running_var1, training)  
        self.a1 = self.relu(self.a1)  # ReLU activation

        # dropout hidden layer 1
        if training:
            # Apply dropout to a1
            self.dropout_mask1 = (np.random.rand(*self.a1.shape) > self.dropout_p1) / (1.0 - self.dropout_p1)
            self.a1 *= self.dropout_mask1

        self.z2 = np.matmul(self.a1, self.W2) + self.b2  # Linear transformation

        # BatchNorm hidden layer 1
        self.a2 = self.batch_norm_forward(self.z2, self.gamma2, self.beta2, self.running_mean2, self.running_var2, training)
        self.a2 = self.relu(self.a2)  # ReLU activation

        # dropout hidden layer 2
        if training:
            # Apply dropout to a2
            self.dropout_mask2 = (np.random.rand(*self.a2.shape) > self.dropout_p2) / (1.0 - self.dropout_p2)
            self.a2 *= self.dropout_mask2

        self.z3 = np.matmul(self.a2, self.W3) + self.b3  # Linear transformation
        self.a3 = self.z3  # Output layer linear activation

        return self.a3
    
    def batch_norm_backward():
        pass

    def relu_derivative():
        pass

    # Backward Propagation
    def backward_pass(self, X, y, y_pred, training=True):
        m = y.shape[0]

        # Compute derivative of loss w.r.t z3
        dz3 = (y_pred - y) / m  # Shape: (m, output_size)
        # Gradients for W3 and b3
        self.dw3 = np.matmul(self.a2.T, dz3) + self.l2_lambda * self.W3  # Shape: (hidden2_size, output_size)
        self.db3 = np.sum(dz3, axis=0, keepdims=True)  # Shape: (1, output_size)

        # Backpropagate to second hidden layer
        da2 = np.matmul(dz3, self.W3.T)  # Shape: (m, hidden2_size)

        if training:
            da2 *= self.dropout_mask2  # Apply dropout mask

        # BatchNorm Backward for second hidden layer
        dz2 = da2 * self.relu_derivative(self.a2)
        dz2, dgamma2, dbeta2 = self.batch_norm_backward(dz2, self.gamma2, self.x_normalized, self.batch_var) # override dz2

        # Gradients for W2 and b2, are found with normalized 
        self.dw2 = np.matmul(self.a1.T, dz2) + self.l2_lambda * self.W2  # Shape: (hidden1_size, hidden2_size)
        self.db2 = np.sum(dz2, axis=0, keepdims=True)  # Shape: (1, hidden2_size)

        # Backpropagate to first hidden layer
        da1 = np.matmul(dz2, self.W2.T)  # Shape: (m, hidden1_size)

        if training:
            da1 *= self.dropout_mask1  # Apply dropout mask

        # BatchNorm Backward for first hidden layer
        dz1 = da1 * self.relu_derivative(self.a1)
        dz1, dgamma1, dbeta1 = self.batch_norm_backward(dz1, self.gamma1, self.x_normalized, self.batch_var) # override dz1

        # Gradients for W1 and b1
        self.dw1 = np.matmul(X.T, dz1) + self.l2_lambda * self.W1  # Shape: (input_size, hidden1_size)
        self.db1 = np.sum(dz1, axis=0, keepdims=True)  # Shape: (1, hidden1_size)

        # Store gradients for BatchNorm parameters
        self.dgamma1 = dgamma1
        self.dbeta1 = dbeta1
        self.dgamma2 = dgamma2
        self.dbeta2 = dbeta2

        self.update_params()