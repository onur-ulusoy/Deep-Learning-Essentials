import numpy as np

# Neural network for desired architecture, created using only numpy
class NN:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.W1 = np.random.rand(input_size, hidden1_size)*0.01
        self.b1 = np.random.rand(1, hidden1_size)
        self.W2 = np.random.rand(input_size, hidden2_size)*0.01
        self.b2 = np.random.rand(1, hidden2_size)
        self.W3 = np.random.rand(input_size, output_size)*0.01
        self.b3 = np.random.rand(1, output_size)

        print("W1 shape: ", self.W1.shape)
        print("b1 shape: ", self.b1.shape)
        print("W2 shape: ", self.W2.shape)
        print("b2 shape: ", self.b2.shape)
        print("W3 shape: ", self.W3.shape)
        print("b3 shape: ", self.b3.shape)

    def forward_pass(x, W1, b1, W2, b2):
        pass
