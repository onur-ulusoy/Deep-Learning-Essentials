import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from polynomial_data import PolynomialDataGenerator
from nn_torch import NN_torch
import hyperparams as hp

class ModelTesterTorch:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.network = NN_torch(self.input_size, self.hidden1_size, self.hidden2_size, self.output_size)

    # Function to load the saved model from the script's path
    def load_model(self, model_path=None):
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'trained_model_torch.pkl')

        # Load the model weights and biases from the pickle file
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            print(f"Model loaded successfully from {model_path}.")

            # Assign the loaded weights and biases to the new model
            with torch.no_grad():  # Disable gradient calculation
                self.network.fc1.weight.data = model_data['fc1_weight']
                self.network.fc1.bias.data = model_data['fc1_bias']
                self.network.fc2.weight.data = model_data['fc2_weight']
                self.network.fc2.bias.data = model_data['fc2_bias']
                self.network.fc3.weight.data = model_data['fc3_weight']
                self.network.fc3.bias.data = model_data['fc3_bias']
        else:
            raise FileNotFoundError(f"No saved model found at {model_path}.")

    # Function to run the test on generated polynomial data
    def run_test(self, degree=2, num_points=40, noise_level=0.1, scale_factor=0.001, seed=7):
        np.random.seed(seed)

        # Generate test data similar to training data
        test_data = PolynomialDataGenerator(degree=degree,
                                            num_points=num_points,
                                            noise_level=noise_level,
                                            scale_factor=scale_factor,
                                            seed=seed)
        #test_data.plot_data()
        # Get test data points and labels
        X_test, y_test = test_data.get_data()
        y_test = y_test.reshape(-1, 1)

        # Perform a forward pass using the loaded model
        self.network.eval()  # Set model to evaluation mode
        with torch.no_grad():
            y_pred = self.network.forward(torch.tensor(X_test, dtype=torch.float32)).numpy()

        # Plot the predictions
        self.plot_predictions(X_test, y_test, y_pred)

        # Calculate and print the mean squared error
        mse = np.mean((y_test - y_pred)**2)
        print(f"Mean Squared Error on the test data: {mse:.4f}")

    # Function to plot predictions
    def plot_predictions(self, X_test, y_test, y_pred):
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test, y_test, label='True Values', color='blue')
        plt.scatter(X_test, y_pred, label='Predicted Values', color='red')
        plt.title('Model Predictions vs True Values')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

# Main function to test the model
if __name__ == "__main__":

    # Create an instance of the ModelTesterTorch class
    tester = ModelTesterTorch(hp.input_size, hp.hidden1_size, hp.hidden2_size, hp.output_size)

    # Load the saved model
    tester.load_model()

    # Run the test
    tester.run_test(degree=2, num_points=50, noise_level=0, scale_factor=0.1, seed=hp.seed)
