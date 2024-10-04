from polynomial_data import PolynomialDataGenerator
import numpy as np
from nn import NN

# parameters for the polynomial data
degree = 5
num_points = 50
noise_level = 4
seed = 41

# Instantiate the data generator
train_data = PolynomialDataGenerator(degree=degree,
                                        num_points=num_points,
                                        noise_level=noise_level,
                                        scale_factor=0.001,
                                        seed=seed)

# Retrieve the data
X, y = train_data.get_data()

# Plot the data
train_data.plot_data()
y = y.reshape(-1, 1)

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Neural Network Architecture
input_size = 1
hidden1_size = 8
hidden2_size = 4
output_size = 1

np.random.seed(41)
network = NN(input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.01)
network.train(X,y,epochs=8)
#network.save_model()