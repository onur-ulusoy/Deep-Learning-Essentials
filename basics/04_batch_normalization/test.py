
import os, pickle, torch
import numpy as np
import matplotlib.pyplot as plt
from basics.dataset_gen_scripts.polynomial_data import PolynomialDataGenerator
from nn import NN
import hyperparams as hp

# Test class fw. prop with NumPy based NN
class ModelTester:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.network = NN(
            self.input_size, 
            self.hidden1_size, 
            self.hidden2_size, 
            self.output_size,
        )
    
    # Function to load the saved model from the script's path
    def load_model(self, model_path=None):
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'trained_model.pkl')

        # Load the model weights, biases, and BatchNorm parameters from the pickle file
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            print(f"Model loaded successfully from {model_path}.")

            # Assign the loaded weights and biases to the network
            with torch.no_grad():  # Disable gradient calculation
                # Weights and biases
                self.network.W1 = model_data['fc1_weight'].numpy().T if isinstance(model_data['fc1_weight'], torch.Tensor) else model_data['fc1_weight']
                self.network.b1 = model_data['fc1_bias'].numpy() if isinstance(model_data['fc1_bias'], torch.Tensor) else model_data['fc1_bias']
                self.network.W2 = model_data['fc2_weight'].numpy().T if isinstance(model_data['fc2_weight'], torch.Tensor) else model_data['fc2_weight']
                self.network.b2 = model_data['fc2_bias'].numpy() if isinstance(model_data['fc2_bias'], torch.Tensor) else model_data['fc2_bias']
                self.network.W3 = model_data['fc3_weight'].numpy().T if isinstance(model_data['fc3_weight'], torch.Tensor) else model_data['fc3_weight']
                self.network.b3 = model_data['fc3_bias'].numpy() if isinstance(model_data['fc3_bias'], torch.Tensor) else model_data['fc3_bias']

                # BatchNorm parameters for first hidden layer
                self.network.gamma1 = model_data['gamma1'].numpy() if isinstance(model_data['gamma1'], torch.Tensor) else model_data['gamma1']
                self.network.beta1 = model_data['beta1'].numpy() if isinstance(model_data['beta1'], torch.Tensor) else model_data['beta1']
                self.network.running_mean1 = model_data['running_mean1'].numpy() if isinstance(model_data['running_mean1'], torch.Tensor) else model_data['running_mean1']
                self.network.running_var1 = model_data['running_var1'].numpy() if isinstance(model_data['running_var1'], torch.Tensor) else model_data['running_var1']

                # BatchNorm parameters for second hidden layer
                self.network.gamma2 = model_data['gamma2'].numpy() if isinstance(model_data['gamma2'], torch.Tensor) else model_data['gamma2']
                self.network.beta2 = model_data['beta2'].numpy() if isinstance(model_data['beta2'], torch.Tensor) else model_data['beta2']
                self.network.running_mean2 = model_data['running_mean2'].numpy() if isinstance(model_data['running_mean2'], torch.Tensor) else model_data['running_mean2']
                self.network.running_var2 = model_data['running_var2'].numpy() if isinstance(model_data['running_var2'], torch.Tensor) else model_data['running_var2']

                # Store original min and max for scaling
                self.y_original_min = model_data['y_original_min']
                self.y_original_max = model_data['y_original_max']

        else:
            raise FileNotFoundError(f"No saved model found at {model_path}.")

    # Function to run the test on generated polynomial data
    def run_test(self, degree, num_points, noise_level, scale_factor, polynomial_seed, noise_seed):

        # Generate test data similar to training data
        test_data = PolynomialDataGenerator(
            degree=degree,
            num_points=num_points,
            noise_level=noise_level,
            scale_factor=scale_factor,
            polynomial_seed=polynomial_seed,
            noise_seed=noise_seed
        )
        # test_data.plot_data()

        # Get test data points and labels
        X_test, y_test = test_data.get_data()
        y_test = y_test.reshape(-1, 1)

        # Scale the test labels using stored min and max
        y_test_scaled = (y_test - self.y_original_min) / (self.y_original_max - self.y_original_min)

        # Perform a forward pass using the loaded model
        y_pred_scaled = self.network.forward_pass(X_test, training=False).astype(np.float32)
        y_pred_original = y_pred_scaled * (self.y_original_max - self.y_original_min) + self.y_original_min

        # Calculate and print the mean squared error (Normalized error)
        mse = np.mean((y_test_scaled - y_pred_scaled)**2)
        print(f"Mean Squared Error on the test data: {mse:.4f}")

        # Plot the predictions
        self.plot_predictions(X_test, y_test, y_pred_original)

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

    # Create an instance of the ModelTester class
    tester = ModelTester(
        hp.input_size, 
        hp.hidden1_size, 
        hp.hidden2_size, 
        hp.output_size
    )

    # Load the saved model
    tester.load_model()

    # Run the test
    tester.run_test(
        degree=hp.degree, 
        num_points=hp.num_points, 
        noise_level=hp.noise_level, 
        scale_factor=hp.scale_factor, 
        polynomial_seed=hp.polynomial_seed, 
        noise_seed=hp.noise_seed_test
    )
    