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

    def get_coordinates(self):
        """
        Get the x and y coordinates of the spirals.
        
        :return: Tuple of x and y coordinates for spiral1 and spiral2
        """
        if self.spiral1 is None or self.spiral2 is None:
            raise ValueError("Spiral data not generated. Call `generate_data()` first.")
        
        # Spiral 1 coordinates (x1, y1)
        x1, y1 = self.spiral1[:, 0], self.spiral1[:, 1]
        
        # Spiral 2 coordinates (x2, y2)
        x2, y2 = self.spiral2[:, 0], self.spiral2[:, 1]
        
        return (x1, y1), (x2, y2)
    
    def get_labeled_data(self):
        """ combine spiral coordinates and create labels """
        # Combine coordinates into a single feature array
        X = np.vstack((self.spiral1, self.spiral2))

        # Create labels: 0 for blue, 1 for red
        y = np.array([0] * self.num_points + [1] * self.num_points)

        return X,y
    
    def plot_data(self):
        """
        Plot the generated spiral data using matplotlib.
        """
        # Get the coordinates of the two spirals
        (x1, y1), (x2, y2) = self.get_coordinates()
        
        # Plotting
        plt.figure(figsize=(6, 6))
        plt.scatter(x1, y1, color='blue', label='Spiral 1')
        plt.scatter(x2, y2, color='red', label='Spiral 2')
        plt.axis('equal')  # Equal scaling for both axes
        plt.legend()
        plt.title('Extended Spiral Data Points')
        plt.show()