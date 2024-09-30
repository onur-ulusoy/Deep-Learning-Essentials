import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from spiral_datapoint import SpiralData
from nn import NN

class ModelTester:
    def __init__(self, input_size=2, hidden1_size=8, hidden2_size=4, output_size=1):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.network = NN(self.input_size, self.hidden1_size, self.hidden2_size, self.output_size)

    # Function to load the saved model from the script's path
    def load_model(self, model_path=None):
        # Get the directory where the script is located
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'trained_model.pkl')

        # Load the model weights and biases from the pickle file
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            print(f"Model loaded successfully from {model_path}.")
            
            # Assign the loaded weights and biases to the model
            self.network.W1 = model_data['W1']
            self.network.b1 = model_data['b1']
            self.network.W2 = model_data['W2']
            self.network.b2 = model_data['b2']
            self.network.W3 = model_data['W3']
            self.network.b3 = model_data['b3']
        else:
            raise FileNotFoundError(f"No saved model found at {model_path}.")

    # Function to run the test on generated spiral data
    def run_test(self, seed=7, num_points=40, noise=0.2, revolutions=4):
        np.random.seed(seed)
        
        # Generate test data
        test_data = SpiralData(num_points=num_points, noise=noise, revolutions=revolutions)
        test_data.generate_data()
        
        # Get test data points and labels
        X_test, y_test = test_data.get_labeled_data()
        y_test = y_test.reshape(-1, 1)

        # Perform a forward pass using the loaded model
        y_pred = self.network.forward_pass(X_test)

        # Classify predictions: if probability > 0.5, classify as 1, else 0
        y_pred_class = (y_pred > 0.5).astype(int)

        # Calculate accuracy
        accuracy = np.mean(y_pred_class == y_test) * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
        
        # Plot test points with predicted classes and values
        self.plot_predictions(X_test, y_pred_class, y_pred, draw_numbers=True)
    
    # Function to plot test points with their predicted classes and optional labels
    def plot_predictions(self, X, y_pred_class, y_pred, draw_numbers=False):
        plt.figure(figsize=(6, 6))

        # Plot points predicted as class 0
        plt.scatter(X[y_pred_class[:, 0] == 0][:, 0], X[y_pred_class[:, 0] == 0][:, 1], color='blue', label='Class 0')

        # Plot points predicted as class 1
        plt.scatter(X[y_pred_class[:, 0] == 1][:, 0], X[y_pred_class[:, 0] == 1][:, 1], color='red', label='Class 1')

        # Annotate points with predicted values if draw_numbers is True
        if draw_numbers:
            for i in range(len(X)):
                plt.annotate(f"{y_pred[i][0]:.2f}", (X[i][0], X[i][1]), fontsize=8, ha='right')

        plt.title("Test Data with Predicted Classes")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()
