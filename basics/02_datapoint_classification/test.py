import numpy as np
import pickle, os
import matplotlib.pyplot as plt
from spiral_datapoint import SpiralData
from nn import NN

# Function to load the saved model from the script's path
def load_model():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'trained_model.pkl')

    # Load the model weights and biases from the pickle file
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        print(f"Model loaded successfully from {file_path}.")
        return model_data
    else:
        raise FileNotFoundError(f"No saved model found at {file_path}.")

# Function to plot test points with their predicted classes
def plot_predictions(X, y_pred_class):
    plt.figure(figsize=(6, 6))

    # Plot points predicted as class 0
    plt.scatter(X[y_pred_class[:, 0] == 0][:, 0], X[y_pred_class[:, 0] == 0][:, 1], color='blue', label='Class 0')

    # Plot points predicted as class 1
    plt.scatter(X[y_pred_class[:, 0] == 1][:, 0], X[y_pred_class[:, 0] == 1][:, 1], color='red', label='Class 1')

    plt.title("Test Data with Predicted Classes")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

# Load model weights and biases
model_data = load_model()

# Create a new NN instance with the same architecture, initialized with zeros
# (We'll load the saved weights below)
input_size = 2
hidden1_size = 8
hidden2_size = 4
output_size = 1

network = NN(input_size, hidden1_size, hidden2_size, output_size)

# Assign the loaded weights and biases to the new model
network.W1 = model_data['W1']
network.b1 = model_data['b1']
network.W2 = model_data['W2']
network.b2 = model_data['b2']
network.W3 = model_data['W3']
network.b3 = model_data['b3']

# Create a new test dataset with different seed and less noise
np.random.seed(7)
test_data = SpiralData(num_points=40, noise=0.2, revolutions=4)
test_data.generate_data()

# Get test data points and labels
X_test, y_test = test_data.get_labeled_data()
y_test = y_test.reshape(-1, 1)

# Perform a forward pass using the loaded model
y_pred = network.forward_pass(X_test)

# Classify predictions: if probability > 0.5, classify as 1, else 0
y_pred_class = (y_pred > 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(y_pred_class == y_test) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

test_data.plot_data()
# Plot test points with predicted classes
plot_predictions(X_test, y_pred_class)
