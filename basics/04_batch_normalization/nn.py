import numpy as np
import torch, pickle, os

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
            
            # We have to return the other parameters as None for test to not raise run time error, because it expects 4 params
            return out, None, None, None 
    
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
    def batch_norm_backward(self, dout, gamma, x_normalized, batch_var, x_centered):
        """
        /**
        * @brief Performs the backward pass for batch normalization.
        *
        * This method computes the gradients of the loss with respect to the input 
        * of the BatchNorm layer and its learnable parameters gamma and beta.
        *
        * @param dout 
        *        Gradient of the loss with respect to the output of the BatchNorm layer.
        *        Shape: (batch_size, num_features)
        *
        * @param gamma 
        *        Learnable scale parameter for the BatchNorm layer.
        *        Shape: (1, num_features)
        *
        * @param x_normalized 
        *        Normalized input,  output of the BatchNorm layer.
        *        computed as (x - mean) / sqrt(variance + epsilon) during fw. prop.
        *        Shape: (batch_size, num_features)
        *
        * @param batch_var 
        *        Variance of the batch computed during the forward pass.
        *        Shape: (1, num_features)
        *
        * @param x_centered 
        *        Centered input, computed as (x - mean).
        *        Shape: (batch_size, num_features)
        *
        * @return 
        *        A tuple containing:
        *        - dx: Gradient of the loss with respect to the input of the BatchNorm layer.
        *              Shape: (batch_size, num_features)
        *        - dgamma: Gradient of the loss with respect to the gamma parameter.
        *                  Shape: (1, num_features)
        *        - dbeta: Gradient of the loss with respect to the beta parameter.
        *                 Shape: (1, num_features)
        */
        """
       
        m = dout.shape[0]

        """ 
        Compute the gradient with respect to beta (dbeta):
        out = gamma * x_normalized + beta 
        => dout/dbeta = 1
        dLoss/dbeta = dLoss/dout * dout/dbeta = dLoss/dout
        => dLoss/dbeta = sum(dout) over the batch
        """
        dbeta = np.sum(dout, axis=0, keepdims=True)
        # print("dout:", dout)
        # print("dbeta:", dbeta)

        """ 
        Compute the gradient with respect to gamma (dgamma):
        out = gamma * x_normalized + beta
        => dout/dgamma = x_normalized
        dLoss/dgamma = dLoss/dout * dout/dgamma = dLoss/dout * x_normalized
        => dLoss/dgamma = sum(dout * x_normalized) over the batch
        """
        dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)

        """ 
        Compute the gradient with respect to the normalized input (dx_normalized):
        Since out = gamma * x_normalized + beta,
        => dout/dx_normalized = gamma
        => dLoss/dx_normalized = dout * gamma
        """
        dx_normalized = dout * gamma

        """ 
        Compute the gradient with respect to variance (dvar):
        The variance affects the normalization as follows:
        x_normalized = (x - mean) / sqrt(var + epsilon)
        => dLoss/dvar = sum(dLoss/dx_normalized * (x - mean) * (-0.5) * (var + epsilon)^(-1.5)) over the batch
        This accounts for how changes in variance influence the normalized inputs.
        """
        dvar = np.sum(
            dx_normalized * x_centered * -0.5 * (batch_var + self.epsilon)**(-1.5), 
            axis=0, 
            keepdims=True
        )

        """ 
        Compute the gradient with respect to mean (dmean):
        The mean affects the normalization in two ways:
        1. Directly through (x - mean)
        2. Indirectly through the variance (since variance is computed based on the mean)
        
        Therefore, the total gradient with respect to mean is:
        dLoss/dmean = sum(dLoss/dx_normalized * (-1) / sqrt(var + epsilon)) over the batch 
                    + dvar * mean(-2 * (x - mean)) over the batch
        """
        dmean = (
            np.sum(
                dx_normalized * -1 / np.sqrt(batch_var + self.epsilon), 
                axis=0, 
                keepdims=True
            ) 
            + 
            dvar * np.mean(-2 * x_centered, axis=0, keepdims=True)
        )

        """ 
        Compute the gradient with respect to the input x (dx):
        Combining the gradients from the normalization, variance, and mean:
        dx = (dLoss/dx_normalized) / sqrt(var + epsilon)
            + (dLoss/dvar) * 2 * (x - mean) / m
            + (dLoss/dmean) / m
        This ensures that all components influencing the input x are accounted for in the gradient.
        """
        dx = (
            dx_normalized / np.sqrt(batch_var + self.epsilon) 
            + 
            dvar * 2 * x_centered / m 
            + 
            dmean / m
        )

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

    def calculate_loss(self, y, y_pred):
        """
        Calculate the Mean Squared Error (MSE) loss between true values and predictions.
        """
        loss = np.mean((y - y_pred) ** 2)

        # Compute weights sum for L2 regularization
        weights_sum = np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)) + np.sum(np.square(self.W3))

        m = y.shape[0]
        # Compute L2 regularization term
        l2_loss = self.l2_lambda / (2 * m) * weights_sum
        loss += l2_loss
        return loss

    def train_model(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass to get predictions
            y_pred = self.forward_pass(X, training=True)
            # Backward pass to update weights
            self.backward_pass(X, y, y_pred, training=True)
            # Every 100 epochs, calculate and print the loss
            if epoch % 100 == 0:
                loss = self.calculate_loss(y, y_pred)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def save_model(self, y_original_min, y_original_max):
        # Convert weights, biases, and BatchNorm parameters to PyTorch tensors if they aren't already
        model_data = {
            # Weights and biases
            'fc1_weight': torch.tensor(self.W1.T, dtype=torch.float32) if not isinstance(self.W1, torch.Tensor) else self.W1,
            'fc1_bias': torch.tensor(self.b1, dtype=torch.float32) if not isinstance(self.b1, torch.Tensor) else self.b1,
            'fc2_weight': torch.tensor(self.W2.T, dtype=torch.float32) if not isinstance(self.W2, torch.Tensor) else self.W2,
            'fc2_bias': torch.tensor(self.b2, dtype=torch.float32) if not isinstance(self.b2, torch.Tensor) else self.b2,
            'fc3_weight': torch.tensor(self.W3.T, dtype=torch.float32) if not isinstance(self.W3, torch.Tensor) else self.W3,
            'fc3_bias': torch.tensor(self.b3, dtype=torch.float32) if not isinstance(self.b3, torch.Tensor) else self.b3,
            # BatchNorm parameters for first hidden layer
            'gamma1': torch.tensor(self.gamma1, dtype=torch.float32) if not isinstance(self.gamma1, torch.Tensor) else self.gamma1,
            'beta1': torch.tensor(self.beta1, dtype=torch.float32) if not isinstance(self.beta1, torch.Tensor) else self.beta1,
            'running_mean1': torch.tensor(self.running_mean1, dtype=torch.float32) if not isinstance(self.running_mean1, torch.Tensor) else self.running_mean1,
            'running_var1': torch.tensor(self.running_var1, dtype=torch.float32) if not isinstance(self.running_var1, torch.Tensor) else self.running_var1,
            # BatchNorm parameters for second hidden layer
            'gamma2': torch.tensor(self.gamma2, dtype=torch.float32) if not isinstance(self.gamma2, torch.Tensor) else self.gamma2,
            'beta2': torch.tensor(self.beta2, dtype=torch.float32) if not isinstance(self.beta2, torch.Tensor) else self.beta2,
            'running_mean2': torch.tensor(self.running_mean2, dtype=torch.float32) if not isinstance(self.running_mean2, torch.Tensor) else self.running_mean2,
            'running_var2': torch.tensor(self.running_var2, dtype=torch.float32) if not isinstance(self.running_var2, torch.Tensor) else self.running_var2,
            'y_original_min': y_original_min,
            'y_original_max': y_original_max
        }

        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'trained_model.pkl')

        # Save to a file (pickle format)
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model weights, biases, BatchNorm parameters, and y_original_min/max saved successfully at {file_path}.")   

# Basic code block to observe and test batch normalization algorithms if there is any runtime error 
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

    nn.train_model(X,y,epochs=1000)
    nn.save_model(1,1)