import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate some data: y = 2x + 1 with some noise
np.random.seed(0)
x = np.linspace(-1, 1, 100)
y = 2 * x + 1 + np.random.normal(0, 0.1, x.shape)

# Initialize parameters (weights and bias)
weight = np.random.randn()
bias = np.random.randn()

# Hyperparameters
learning_rate = 0.01
epochs = 120

# Define the simple linear model
def forward_pass(x, weight, bias):
    return weight * x + bias

# Define the Mean Squared Error (MSE) loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define the gradient calculation for weights and bias
def compute_gradients(x, y_true, y_pred):
    dL_dw = -2 * np.mean(x * (y_true - y_pred))
    dL_db = -2 * np.mean(y_true - y_pred)
    return dL_dw, dL_db

# Training function with visualization
fig, axs = plt.subplots(4, 1, figsize=(10, 18))
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.3)

# Initialize plots
ax1, ax2, ax3, ax4 = axs

# Plot for original data and fitted line
ax1.scatter(x, y, label='Original data')
fitted_line, = ax1.plot(x, forward_pass(x, weight, bias), label='Fitted line', color='r')
input_text = ax1.text(0.1, 0.5, '', transform=ax1.transAxes)
weight_text = ax1.text(0.5, 0.6, '', transform=ax1.transAxes)
bias_text = ax1.text(0.5, 0.4, '', transform=ax1.transAxes)
output_text = ax1.text(0.9, 0.5, '', transform=ax1.transAxes)
epoch_loss_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-2, 4)
ax1.set_xlabel('Input')
ax1.set_ylabel('Output')
ax1.legend()

# Plot for loss
loss_values = []
loss_line, = ax2.plot([], [], label='Loss', color='b')
ax2.set_xlim(0, epochs)
ax2.set_ylim(0, 1)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

# Plot for gradient dL_dw
dw_values = []
dw_line, = ax3.plot([], [], label='dL_dw', color='g')
ax3.set_xlim(0, epochs)
ax3.set_ylim(-1, 1)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('dL_dw')
ax3.legend()

# Plot for gradient dL_db
db_values = []
db_line, = ax4.plot([], [], label='dL_db', color='m')
ax4.set_xlim(0, epochs)
ax4.set_ylim(-1, 1)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('dL_db')
ax4.legend()

def update(frame):
    global weight, bias, x, y, learning_rate

    # Forward pass: Calculate predicted values using current weight and bias
    y_pred = forward_pass(x, weight, bias)
    
    # Calculate loss: Measure how far off the predictions are from the actual values
    loss = mse_loss(y, y_pred)
    
    # Compute gradients: Determine how to adjust the weight and bias to reduce the loss
    dL_dw, dL_db = compute_gradients(x, y, y_pred)
    
    # Update weights and bias: Adjust the parameters based on the gradients
    weight -= learning_rate * dL_dw
    bias -= learning_rate * dL_db

    # Update predictions with new weight and bias for visualization
    y_pred = forward_pass(x, weight, bias)
    fitted_line.set_ydata(y_pred)
    
    # Update texts: Display current values of weight, bias, and loss
    input_text.set_text(f'Input: x')
    weight_text.set_text(f'Weight: {weight:.4f}')
    bias_text.set_text(f'Bias: {bias:.4f}')
    output_text.set_text(f'Output: y')
    epoch_loss_text.set_text(f'Epoch: {frame+1}, Loss: {loss:.4f}')
    
    # Update loss plot
    loss_values.append(loss)
    loss_line.set_data(range(len(loss_values)), loss_values)
    ax2.set_ylim(0, max(loss_values) * 1.1)  # Adjust y-axis limit to fit the loss values

    # Update dL_dw plot
    dw_values.append(dL_dw)
    dw_line.set_data(range(len(dw_values)), dw_values)
    ax3.set_ylim(min(dw_values) * 1.1, max(dw_values) * 1.1)  # Adjust y-axis limit to fit dL_dw values

    # Update dL_db plot
    db_values.append(dL_db)
    db_line.set_data(range(len(db_values)), db_values)
    ax4.set_ylim(min(db_values) * 1.1, max(db_values) * 1.1)  # Adjust y-axis limit to fit dL_db values

    return fitted_line, input_text, weight_text, bias_text, output_text, epoch_loss_text, loss_line, dw_line, db_line

# Animation
ani = FuncAnimation(fig, update, frames=range(epochs), blit=True, repeat=False)

# Enable slow motion mode
plt.show()

print('Finished Training')