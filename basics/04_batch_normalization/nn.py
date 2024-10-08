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
            
            running variables are used in test in order to find the normalized value
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