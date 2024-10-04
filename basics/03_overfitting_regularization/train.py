from polynomial_data import PolynomialDataGenerator
import matplotlib.pyplot as plt

# parameters for the polynomial data
degree = 5
num_points = 50
noise_level = 20000
seed = 41

# Instantiate the data generator
train_data = PolynomialDataGenerator(degree=degree,
                                        num_points=num_points,
                                        noise_level=noise_level,
                                        seed=seed)

# Retrieve the data
X, y = train_data.get_data()
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Plot the data
train_data.plot_data()

