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
    
    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)
    
    # BatchNorm Forward Function
    def batch_norm_forward(self, x, gamma, beta, running_mean, running_var, training=True):
        if training:
            # Find mean and variance of all samples as a vector
            # mean or variance of k-th feature of all the samples is written to k-th column of 1xn vector
            """ x: [[0.45787162 0.72803223]
                    [0.4663173  0.6708576 ]]
                batch_mean: [[0.46209446 0.6994449 ]] batch var: [[1.7832379e-05 8.1723440e-04]]
                x_normalized: [[-0.80044127  0.9939384 ]
                              [ 0.80044127 -0.99393636]] """
            
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)
            # print("x:", x)
            # print("batch_mean:", batch_mean, "batch var:", batch_var)

            # Normalize x to have mean of 0 and variance of 1
            x_centered = x - batch_mean
            x_normalized = x_centered / np.sqrt(batch_var + self.epsilon)
            # print("x_normalized:", x_normalized)

            # Scale and shift to the optimal range
            out = gamma * x_normalized + beta
            # print("out:", out)

            # Update Running Mean and Var along the way in every batch
            running_mean[:] = self.momentum * running_mean + (1 - self.momentum) * batch_mean
            running_var[:] = self.momentum * running_var + (1 - self.momentum) * batch_var

            return out, x_normalized, batch_var, x_centered


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
        self.z1, self.x_normalized1, self.batch_var1, self.x_centered1 = self.batch_norm_forward(
            self.z1, self.gamma1, self.beta1, self.running_mean1, self.running_var1, training)
        
        self.a1 = self.relu(self.z1)  # ReLU activation
        
        # dropout hidden layer 1
        if training:
            # Apply dropout to a1
            self.dropout_mask1 = (np.random.rand(*self.a1.shape) > self.dropout_p1) / (1.0 - self.dropout_p1)
            self.a1 *= self.dropout_mask1

        self.z2 = np.matmul(self.a1, self.W2) + self.b2  # Linear transformation

        # BatchNorm hidden layer 1
        self.z2, self.x_normalized2, self.batch_var2, self.x_centered2 = self.batch_norm_forward(
            self.z2, self.gamma2, self.beta2, self.running_mean2, self.running_var2, training)
        
        self.a2 = self.relu(self.z2)  # ReLU activation

        # dropout hidden layer 2
        if training:
            # Apply dropout to a2
            self.dropout_mask2 = (np.random.rand(*self.a2.shape) > self.dropout_p2) / (1.0 - self.dropout_p2)
            self.a2 *= self.dropout_mask2

        self.z3 = np.matmul(self.a2, self.W3) + self.b3  # Linear transformation
        self.a3 = self.z3  # Output layer linear activation

        return self.a3
    
    # BatchNorm Backward Function
    """ to compute the gradients, derivatives of the loss with respect to the inputs and the learnable parameters """
    def batch_norm_backward(self, dout, gamma, x_normalized, batch_var, x_centered):
        m = dout.shape[0]

        """ out = gamma*out+beta -> dout/dbeta = 1
        dLoss/dbeta = dLoss/dout * dout/dbeta = dLoss/dout
        we sum up entire batch of dLoss/dout """
        dbeta = np.sum(dout, axis=0, keepdims=True)
        # print("dout:", dout)
        # print("dbeta:", dbeta)
        """ The parameter gamma is multiplied by the normalized output => dout/dgamma = out,
        dLoss/dgamma = dLoss/dout * dout/dgamma = dLoss/dout * out, where out is x_normalized 
        so the dL/dgamma is: """
        dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
        # print("dgamma:", dgamma)

        dx_normalized = dout * gamma
        dvar = np.sum(dx_normalized * x_centered * -0.5 * (batch_var + self.epsilon)**(-1.5), axis=0, keepdims=True)
        dmean = np.sum(dx_normalized * -1 / np.sqrt(batch_var + self.epsilon), axis=0, keepdims=True) + \
                dvar * np.mean(-2 * x_centered, axis=0, keepdims=True)

        dx = dx_normalized / np.sqrt(batch_var + self.epsilon) + \
             dvar * 2 * x_centered / m + \
             dmean / m

        return dx, dgamma, dbeta
    
    # Backward Propagation
    def backward_pass(self, X, y, y_pred, training=True):
        m = y.shape[0]

        # Compute derivative of loss w.r.t z3
        # Equation is straightforward because linear activation is used at output with MSE Loss.
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
        dz2, dgamma2, dbeta2 = self.batch_norm_backward(
            dz2, self.gamma2, self.x_normalized2, self.batch_var2, self.x_centered2) # override dz2

        # Gradients for W2 and b2, are found with normalized 
        self.dw2 = np.matmul(self.a1.T, dz2) + self.l2_lambda * self.W2  # Shape: (hidden1_size, hidden2_size)
        self.db2 = np.sum(dz2, axis=0, keepdims=True)  # Shape: (1, hidden2_size)

        # Backpropagate to first hidden layer
        da1 = np.matmul(dz2, self.W2.T)  # Shape: (m, hidden1_size)

        if training:
            da1 *= self.dropout_mask1  # Apply dropout mask

        # BatchNorm Backward for first hidden layer
        dz1 = da1 * self.relu_derivative(self.a1)
        dz1, dgamma1, dbeta1 = self.batch_norm_backward(
            dz1, self.gamma1, self.x_normalized1, self.batch_var1, self.x_centered1) # override dz1

        # Gradients for W1 and b1
        self.dw1 = np.matmul(X.T, dz1) + self.l2_lambda * self.W1  # Shape: (input_size, hidden1_size)
        self.db1 = np.sum(dz1, axis=0, keepdims=True)  # Shape: (1, hidden1_size)

        # Store gradients for BatchNorm parameters
        self.dgamma1 = dgamma1
        self.dbeta1 = dbeta1
        self.dgamma2 = dgamma2
        self.dbeta2 = dbeta2

        self.update_params()

    def update_params(self):
        # Update weights and biases
        self.W3 -= self.learning_rate * self.dw3
        self.W2 -= self.learning_rate * self.dw2
        self.W1 -= self.learning_rate * self.dw1

        self.b3 -= self.learning_rate * self.db3
        self.b2 -= self.learning_rate * self.db2
        self.b1 -= self.learning_rate * self.db1

        # Update BatchNorm parameters
        self.gamma2 -= self.learning_rate * self.dgamma2
        self.beta2 -= self.learning_rate * self.dbeta2

        self.gamma1 -= self.learning_rate * self.dgamma1
        self.beta1 -= self.learning_rate * self.dbeta1

# Code block to observe and test batch normalization algorithms
if __name__ == "__main__":
    # Create input data (2 samples, 2 features)
    X = np.array([
        [2, -3],
        [1, 4]
    ], dtype=np.float32)

    # Create target labels (2 samples, 2 classes) as one-hot vectors
    y = np.array([
        [1, 0],  # Class 0
        [0, 1]   # Class 1
    ], dtype=np.float32)

    # Initialize the neural network
    input_size = 2
    hidden1_size = 2
    hidden2_size = 2
    output_size = 2
    learning_rate = 0.01

    nn = NN(input_size, hidden1_size, hidden2_size, output_size, learning_rate)

    # Forward pass
    y_pred = nn.forward_pass(X, training=True)
    print("\nOutput of forward pass:\n", y_pred)

    # Backward pass
    nn.backward_pass(X, y, y_pred, training=True)
    print("\nBackward pass completed.")