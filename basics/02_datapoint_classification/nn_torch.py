import torch
import torch.nn as nn
import torch.optim as optim
import pickle, os
import matplotlib.pyplot as plt

# Set seed for reproducibility
#torch.manual_seed(41)

# Neural network for desired architecture, created using PyTorch
class NN_torch(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.01, seed=42):
        torch.manual_seed(seed)
        super(NN_torch, self).__init__()
        self.learning_rate = learning_rate

        # Xavier initialization is default in nn.Linear, so we don't need to manually initialize.
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)

        # Using ReLU activation and Sigmoid for the output layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Binary Cross-Entropy Loss for binary classification
        self.criterion = nn.BCELoss()

        # To store training history
        self.loss_history = []
        self.weights_history = [[] for _ in range(3)]
        self.biases_history = [[] for _ in range(3)]
        # To store gradients history
        self.grad_weights_history = [[] for _ in range(3)]
        self.grad_biases_history = [[] for _ in range(3)]


    # Forward propagation
    def forward(self, X):
        z1 = self.fc1(X)
        self.a1 = self.relu(z1)

        z2 = self.fc2(self.a1)
        self.a2 = self.relu(z2)

        z3 = self.fc3(self.a2)
        self.a3 = self.sigmoid(z3)

        return self.a3

    
    def train_model(self, X_train, y_train, epochs = 1000):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        for epoch in range(epochs):
            # forward pass
            y_pred = self.forward(X_train)

            #calculate loss
            self.loss = self.criterion(y_pred, y_train)

            # Backward pass
            self.loss.backward()

            self.store_history()

            # Update params
            with torch.no_grad():
                for param in self.parameters():
                    param -= self.learning_rate * param.grad

            self.zero_grad()

            # Print the loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {self.loss.item():.4f}')

        self.save_model()

    def plot_training(self):
        epochs = len(self.loss_history)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Progress', fontsize=16)

        # Plot loss over epochs
        axs[0, 0].plot(range(epochs), self.loss_history, 'r', label="Loss")
        axs[0, 0].set_title("Loss")
        axs[0, 0].set_xlabel("Epochs")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].legend()  # Add a legend for the loss plot

        # Plot weights for each layer
        for i in range(3):
            if len(self.weights_history[i]) == epochs:  # Ensure history is filled
                axs[0, 1].plot(range(epochs), self.weights_history[i], label=f'Layer {i+1} Weights')

        # Only show legend if there are valid plots
        if any(len(self.weights_history[i]) == epochs for i in range(3)):
            axs[0, 1].legend()

        axs[0, 1].set_title("Weights")
        axs[0, 1].set_xlabel("Epochs")
        axs[0, 1].set_ylabel("Weight Values")

        # Plot biases for each layer
        for i in range(3):
            if len(self.biases_history[i]) == epochs:  # Ensure history is filled
                axs[1, 0].plot(range(epochs), self.biases_history[i], label=f'Layer {i+1} Biases')

        # Only show legend if there are valid plots
        if any(len(self.biases_history[i]) == epochs for i in range(3)):
            axs[1, 0].legend()

        axs[1, 0].set_title("Biases")
        axs[1, 0].set_xlabel("Epochs")
        axs[1, 0].set_ylabel("Bias Values")

        # Plot gradients for each layer
        for i in range(3):
            if len(self.grad_weights_history[i]) == epochs:  # Ensure history is filled
                axs[1, 1].plot(range(epochs), self.grad_weights_history[i], label=f'Layer {i+1} dW')
            if len(self.grad_biases_history[i]) == epochs:  # Ensure history is filled
                axs[1, 1].plot(range(epochs), self.grad_biases_history[i], '--', label=f'Layer {i+1} dB')

        # Only show legend if there are valid plots
        if any(len(self.grad_weights_history[i]) == epochs for i in range(3)):
            axs[1, 1].legend()

        axs[1, 1].set_title("Gradients (Weights and Biases)")
        axs[1, 1].set_xlabel("Epochs")
        axs[1, 1].set_ylabel("Gradient Values")

        plt.tight_layout()
        plt.show()

    def store_history(self):
        # Append loss to history
        self.loss_history.append(self.loss.item())   

        # Store weights and biases history for plotting
        self.weights_history[0].append(self.fc1.weight.data.mean().item())
        self.weights_history[1].append(self.fc2.weight.data.mean().item())
        self.weights_history[2].append(self.fc3.weight.data.mean().item())
        self.biases_history[0].append(self.fc1.bias.data.mean().item())
        self.biases_history[1].append(self.fc2.bias.data.mean().item())
        self.biases_history[2].append(self.fc3.bias.data.mean().item())

        # Store gradients for weights and biases for plotting
        self.grad_weights_history[0].append(self.fc1.weight.grad.data.mean().item())
        self.grad_weights_history[1].append(self.fc2.weight.grad.data.mean().item())
        self.grad_weights_history[2].append(self.fc3.weight.grad.data.mean().item())
        self.grad_biases_history[0].append(self.fc1.bias.grad.data.mean().item())
        self.grad_biases_history[1].append(self.fc2.bias.grad.data.mean().item())
        self.grad_biases_history[2].append(self.fc3.bias.grad.data.mean().item())

    # Save model's weights and biases
    def save_model(self):
        model_data = {
            'fc1_weight': self.fc1.weight.data,
            'fc1_bias': self.fc1.bias.data,
            'fc2_weight': self.fc2.weight.data,
            'fc2_bias': self.fc2.bias.data,
            'fc3_weight': self.fc3.weight.data,
            'fc3_bias': self.fc3.bias.data
        }

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'trained_model_torch.pkl')

        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model weights and biases saved successfully at {file_path}.")