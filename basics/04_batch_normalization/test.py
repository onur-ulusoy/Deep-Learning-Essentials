
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
    