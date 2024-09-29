import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class AnimationNN:
    def __init__(self, nn, X, y):
        # Store the neural network object and training data
        self.nn = nn
        self.X = X
        self.y = y

        # Initialize lists to store the history of loss, weights, and gradients
        self.loss_history = []
        self.W1_history = []
        self.b1_history = []
        self.W2_history = []
        self.b2_history = []
        self.W3_history = []
        self.b3_history = []

        self.dW1_history = []
        self.db1_history = []
        self.dW2_history = []
        self.db2_history = []
        self.dW3_history = []
        self.db3_history = []

        # Create figure and axes for subplots
        self.fig, self.axes = plt.subplots(3, 4, figsize=(15, 10))
        self.fig.suptitle('Neural Network Training Visualization')

        # Plot initialization
        self.initialize_plots()

    def initialize_plots(self):
        # Initialize Loss plot
        self.axes[0, 0].set_title("Loss")
        self.loss_line, = self.axes[0, 0].plot([], [], 'r-')
        
        # Initialize W1, b1, W2, b2, W3, b3 plots
        self.W1_line, = self.axes[0, 1].plot([], [], 'g-', label="W1")
        self.b1_line, = self.axes[0, 2].plot([], [], 'b-', label="b1")
        self.W2_line, = self.axes[1, 1].plot([], [], 'g-', label="W2")
        self.b2_line, = self.axes[1, 2].plot([], [], 'b-', label="b2")
        self.W3_line, = self.axes[2, 1].plot([], [], 'g-', label="W3")
        self.b3_line, = self.axes[2, 2].plot([], [], 'b-', label="b3")
        
        # Initialize dW1, db1, dW2, db2, dW3, db3 plots
        self.dW1_line, = self.axes[0, 3].plot([], [], 'g--', label="dW1")
        self.db1_line, = self.axes[0, 3].plot([], [], 'b--', label="db1")
        self.dW2_line, = self.axes[1, 3].plot([], [], 'g--', label="dW2")
        self.db2_line, = self.axes[1, 3].plot([], [], 'b--', label="db2")
        self.dW3_line, = self.axes[2, 3].plot([], [], 'g--', label="dW3")
        self.db3_line, = self.axes[2, 3].plot([], [], 'b--', label="db3")

        # Set axis labels
        for ax in self.axes.flat:
            ax.set_xlim(0, 1000)
            ax.set_ylim(-1, 1)  # Adjust based on your specific parameter range

    def update_plot_data(self, epoch, loss):
        # Update histories
        self.loss_history.append(loss)
        self.W1_history.append(self.nn.W1.mean())
        self.b1_history.append(self.nn.b1.mean())
        self.W2_history.append(self.nn.W2.mean())
        self.b2_history.append(self.nn.b2.mean())
        self.W3_history.append(self.nn.W3.mean())
        self.b3_history.append(self.nn.b3.mean())

        self.dW1_history.append(self.nn.dw1.mean())
        self.db1_history.append(self.nn.db1.mean())
        self.dW2_history.append(self.nn.dw2.mean())
        self.db2_history.append(self.nn.db2.mean())
        self.dW3_history.append(self.nn.dw3.mean())
        self.db3_history.append(self.nn.db3.mean())

        # Update lines data for animation
        self.loss_line.set_data(range(len(self.loss_history)), self.loss_history)
        self.W1_line.set_data(range(len(self.W1_history)), self.W1_history)
        self.b1_line.set_data(range(len(self.b1_history)), self.b1_history)
        self.W2_line.set_data(range(len(self.W2_history)), self.W2_history)
        self.b2_line.set_data(range(len(self.b2_history)), self.b2_history)
        self.W3_line.set_data(range(len(self.W3_history)), self.W3_history)
        self.b3_line.set_data(range(len(self.b3_history)), self.b3_history)

        self.dW1_line.set_data(range(len(self.dW1_history)), self.dW1_history)
        self.db1_line.set_data(range(len(self.db1_history)), self.db1_history)
        self.dW2_line.set_data(range(len(self.dW2_history)), self.dW2_history)
        self.db2_line.set_data(range(len(self.db2_history)), self.db2_history)
        self.dW3_line.set_data(range(len(self.dW3_history)), self.dW3_history)
        self.db3_line.set_data(range(len(self.db3_history)), self.db3_history)

        # Redraw the plot
        self.fig.canvas.draw()

    def animate(self, epochs, interval=100):
        def update(epoch):
            y_pred = self.nn.forward_pass(self.X)  # Perform forward pass
            self.nn.backward_pass(self.X, self.y, y_pred)  # Perform backpropagation
            
            loss = self.nn.calculate_loss(self.y, y_pred)
            self.update_plot_data(epoch, loss)  # Update the plots

            if epoch % interval == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        anim = FuncAnimation(self.fig, update, frames=epochs, repeat=False, interval=200)
        plt.show()
