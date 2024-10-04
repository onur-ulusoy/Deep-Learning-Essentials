import numpy as np
import matplotlib.pyplot as plt

class PolynomialDataGenerator:
    def __init__(self, degree, num_points, noise_level=2, scale_factor=0.001, polynomial_seed=None, noise_seed=None):
        """
        Initializes the PolynomialDataGenerator.

        Parameters:
        - degree (int): Degree of the polynomial.
        - num_points (int): Number of data points to generate.
        - noise_level (float): Standard deviation of Gaussian noise to add to the targets.
        - seed (int, optional): Seed for random number generation for reproducibility.
        """
        self.degree = degree
        self.num_points = num_points
        self.noise_level = noise_level
        self.scale_factor = scale_factor
        self.polynomial_seed = polynomial_seed
        self.noise_seed = noise_seed
        self.coefficients = None
        self.X = None
        self.y = None
        self._generate_polynomial()
        self._generate_data()

    def _generate_polynomial(self):
        """
        Generates random coefficients for the polynomial based on the specified degree and seed.
        """
        if self.polynomial_seed is not None:
            print("Polynomial seed is:", self.polynomial_seed)
            np.random.seed(self.polynomial_seed)
        # Generate coefficients from a uniform distribution between -10 and 10
        self.coefficients = np.random.uniform(-self.scale_factor, self.scale_factor, self.degree + 1)
        print(f"Generated polynomial coefficients (highest degree first): {self.coefficients}")

    def _generate_data(self):
        """
        Generates input features X and target values y with added Gaussian noise.
        """
        # Generate X values uniformly distributed between -10 and 10
        self.X = np.linspace(-10, 10, self.num_points)
        # Calculate y values based on the polynomial
        self.y = np.polyval(self.coefficients, self.X)

        if self.noise_seed is not None:
            print("Noise seed is:", self.noise_seed)
            np.random.seed(self.noise_seed)

        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_level, self.num_points)
        self.y += noise
        print(f"Generated {self.num_points} data points with noise level {self.noise_level}.")

    def get_data(self):
        """
        Returns the generated data points.

        Returns:
        - X (np.ndarray): Input features.
        - y (np.ndarray): Target values.
        """
        return self.X.reshape(-1, 1), self.y

    def plot_data(self):
        """
        Plots the generated data points along with the underlying polynomial curve.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(self.X, self.y, color='blue', label='Noisy Data Points', alpha=0.6)
        
        # Plot the true polynomial curve without noise
        X_curve = np.linspace(np.min(self.X), np.max(self.X), 500)
        y_curve = np.polyval(self.coefficients, X_curve)
        plt.plot(X_curve, y_curve, color='red', label='True Polynomial', linewidth=2)
        
        plt.title(f"Generated Data for {self.degree}th Degree Polynomial")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()
