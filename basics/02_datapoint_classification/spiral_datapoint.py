import numpy as np
import matplotlib.pyplot as plt

class SpiralData:
    def __init__(self, num_points=250, noise=0.2, revolutions=4):
        """
        Initialize the SpiralData class with parameters for generating spirals.
        
        :param num_points: Number of points per spiral
        :param noise: The noise level added to the points
        :param revolutions: Number of revolutions (affects the length of the spiral)
        """
        self.num_points = num_points
        self.noise = noise
        self.revolutions = revolutions
        self.spiral1 = None
        self.spiral2 = None

    def generate_data(self):
        """
        Generate the spiral data for two spirals with added noise.
        """
        theta = np.linspace(0, self.revolutions * np.pi, self.num_points)  # Extended range for revolutions
        r_a = theta  # Radius for the first spiral
        r_b = theta  # Radius for the second spiral

        # First spiral (blue)
        self.spiral1 = np.array([r_a * np.cos(theta), r_a * np.sin(theta)]).T
        self.spiral1 += np.random.randn(self.num_points, 2) * self.noise  # Adding noise

        # Second spiral (red, shifted by pi)
        self.spiral2 = np.array([r_b * np.cos(theta + np.pi), r_b * np.sin(theta + np.pi)]).T
        self.spiral2 += np.random.randn(self.num_points, 2) * self.noise  # Adding noise

    def plot_data(self):
        """
        Plot the generated spiral data using matplotlib.
        """
        if self.spiral1 is None or self.spiral2 is None:
            raise ValueError("Spiral data not generated. Call `generate_data()` first.")
        
        plt.figure(figsize=(6, 6))
        plt.scatter(self.spiral1[:, 0], self.spiral1[:, 1], color='blue', label='Spiral 1')
        plt.scatter(self.spiral2[:, 0], self.spiral2[:, 1], color='red', label='Spiral 2')
        plt.axis('equal')  # Equal scaling for both axes
        plt.legend()
        plt.title('Spiral Data Points')
        plt.show()
