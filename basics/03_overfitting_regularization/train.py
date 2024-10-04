from polynomial_data import PolynomialDataGenerator
import numpy as np
from nn import NN
from nn_torch import NN_torch
import hyperparams as hp

# Instantiate the data generator
train_data = PolynomialDataGenerator(degree=hp.degree,
                                        num_points=hp.num_points,
                                        noise_level=hp.noise_level,
                                        scale_factor=hp.scale_factor,
                                        polynomial_seed=hp.polynomial_seed,
                                        noise_seed=hp.noise_seed_training)

# Retrieve the data
X, y = train_data.get_data()

# Plot the data
#train_data.plot_data()
y = y.reshape(-1, 1)

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Neural Network Architecture
input_size = 1
hidden1_size = 88
hidden2_size = 48
output_size = 1

np.random.seed(41)
network = NN(input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.001)
#network.save_model()


# Initialize the PyTorch neural network
network = NN_torch(input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.001)

# Train the network
network.train_model(X, y, epochs=20000)
network.save_model()