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

y_train_min = y.min()
y_train_max = y.max()
print(y.min())
print(y.max())
y_train_scaled = (y - y_train_min) / (y_train_max - y_train_min)
y = y_train_scaled

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")


np.random.seed(41)
network = NN(hp.input_size, hp.hidden1_size, hp.hidden2_size, hp.output_size, learning_rate=0.3, l2_lambda=0.01, dropout_p1=0.05, dropout_p2=0.05)
#network.save_model()


# Initialize the PyTorch neural network
#network = NN_torch(hp.input_size, hp.hidden1_size, hp.hidden2_size, hp.output_size, learning_rate=0.3, l2_lambda=0.01, dropout_p1=0.05, dropout_p2=0.05)

# Train the network
network.train_model(X, y, epochs=50000)
network.save_model()