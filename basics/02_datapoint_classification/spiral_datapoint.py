import numpy as np
import matplotlib.pyplot as plt

# Function to generate extended spiral data
def generate_extended_spiral_data(num_points, noise=0.2):
    theta = np.linspace(0, 4 * np.pi, num_points)  # Increased range of theta for extended spirals
    r_a = theta  # Radius for the first spiral (adjust this scaling for extension)
    data_a = np.array([r_a * np.cos(theta), r_a * np.sin(theta)]).T
    data_a += np.random.randn(num_points, 2) * noise  # Add some noise

    r_b = theta  # Radius for the second spiral
    data_b = np.array([r_b * np.cos(theta + np.pi), r_b * np.sin(theta + np.pi)]).T  # Shift by pi for second spiral
    data_b += np.random.randn(num_points, 2) * noise  # Add some noise
    
    return data_a, data_b

# Number of points for each spiral
num_points = 250  # Increased the number of points for a smoother spiral

# Generate extended spiral data
spiral1, spiral2 = generate_extended_spiral_data(num_points)

# Plotting the data
plt.figure(figsize=(6, 6))
plt.scatter(spiral1[:, 0], spiral1[:, 1], color='blue', label='Spiral 1')
plt.scatter(spiral2[:, 0], spiral2[:, 1], color='red', label='Spiral 2')
plt.axis('equal')  # Equal scaling for both axes
plt.legend()
plt.title('Extended Spiral Data Points')
plt.show()
