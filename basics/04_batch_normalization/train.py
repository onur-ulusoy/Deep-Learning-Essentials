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

y_original_min = y.min()
y_original_max = y.max()
print(y.min())
print(y.max())
y_train_scaled = (y - y_original_min) / (y_original_max - y_original_min)

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
             hp.dropout_p2)

# Train the network
network.train_model(X, y_train_scaled, epochs=hp.epochs)
network.save_model(y_original_min, y_original_max)