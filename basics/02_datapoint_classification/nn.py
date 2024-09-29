import numpy as np

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
    
    # Forward Propagation
    def forward_pass(self, X):
        self.z1 = np.matmul(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1) # first hidden layer activation

        self.z2 = np.matmul(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.matmul(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)

        print("z1 shape: ", self.z1.shape)
        print("z2 shape: ", self.z2.shape)
        print("z3 shape: ", self.z3.shape)

        return self.a3
    
    

