from basics.dataset_gen_scripts.polynomial_data import PolynomialDataGenerator
import numpy as np
from nn import NN
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

# Calculate the mean and standard deviation of the target labels (y)
y_mean = y.mean()
y_std = y.std()

# Standardize y (make mean 0 and std 1)
y_train_standardized = (y - y_mean) / y_std

# Print shapes (same as before)
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Uncomment below to train numpy based nn and save model
np.random.seed(41)
network = NN(hp.input_size, 
             hp.hidden1_size, 
             hp.hidden2_size, 
             hp.output_size, 
             hp.learning_rate, 
             hp.l2_lambda, 
             hp.dropout_p1, 
             hp.dropout_p2,
             hp.batchnorm_momentum)

# Train the network
network.train_model(X, y_train_standardized, epochs=hp.epochs)
network.save_model(y_mean, y_std)