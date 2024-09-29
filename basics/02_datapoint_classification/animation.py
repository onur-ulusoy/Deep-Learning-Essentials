import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class AnimationTraining:
    def __init__(self, nn, X, y, real_time=True):
        self.nn = nn
        self.X = X
        self.y = y
        self.real_time = real_time

        self.loss_history = []
        self.W_history = [[] for _ in range(3)]
        self.b_history = [[] for _ in range(3)]
        self.dW_history = [[] for _ in range(3)]
        self.db_history = [[] for _ in range(3)]

        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))
        self.fig.suptitle('Neural Network Training Visualization', fontsize=16)

        self.initialize_plots()

    def initialize_plots(self):
        self.axes[0, 0].set_title("Loss", fontsize=14)
        self.axes[0, 0].set_xlabel("Epochs", fontsize=12)
        self.axes[0, 0].set_ylabel("Loss", fontsize=12)
        self.loss_line, = self.axes[0, 0].plot([], [], 'r-')

        titles = ["Weights", "Biases", "Gradients"]
        colors = ['r', 'g', 'b']
        
        for i, (ax, title) in enumerate(zip(self.axes.flat[1:], titles)):
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Epochs", fontsize=12)
            for j in range(3):
                if i < 2:
                    line, = ax.plot([], [], f'{colors[j]}-', label=f'Layer {j+1}')
                    setattr(self, f'{title[0].lower()}{j+1}_line', line)
                else:
                    w_line, = ax.plot([], [], f'{colors[j]}-', label=f'dW_{j+1}')
                    b_line, = ax.plot([], [], f'{colors[j]}--', label=f'db_{j+1}')
                    setattr(self, f'dw{j+1}_line', w_line)
                    setattr(self, f'db{j+1}_line', b_line)
            ax.legend(fontsize=10)

        for ax in self.axes.flat:
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.set_xlim(0, 1000)
            ax.set_ylim(-1, 1)

    def update_plot_data(self, epoch, loss):
        self.loss_history.append(loss)
        
        for i in range(3):
            self.W_history[i].append(getattr(self.nn, f'W{i+1}').mean())
            self.b_history[i].append(getattr(self.nn, f'b{i+1}').mean())
            self.dW_history[i].append(getattr(self.nn, f'dw{i+1}').mean())
            self.db_history[i].append(getattr(self.nn, f'db{i+1}').mean())

        self.loss_line.set_data(range(len(self.loss_history)), self.loss_history)

        for i in range(3):
            getattr(self, f'w{i+1}_line').set_data(range(len(self.W_history[i])), self.W_history[i])
            getattr(self, f'b{i+1}_line').set_data(range(len(self.b_history[i])), self.b_history[i])
            getattr(self, f'dw{i+1}_line').set_data(range(len(self.dW_history[i])), self.dW_history[i])
            getattr(self, f'db{i+1}_line').set_data(range(len(self.db_history[i])), self.db_history[i])

        for ax in self.axes.flat:
            ax.set_xlim(0, max(1000, epoch))
            ax.relim()
            ax.autoscale_view()

        if self.real_time:
            self.fig.canvas.draw()

    def animate(self, epochs, interval=100):
        def update(epoch):
            y_pred = self.nn.forward_pass(self.X)
            self.nn.backward_pass(self.X, self.y, y_pred)
            
            loss = self.nn.calculate_loss(self.y, y_pred)
            self.update_plot_data(epoch, loss)

            if epoch % interval == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        if self.real_time:
            anim = FuncAnimation(self.fig, update, frames=epochs, repeat=False, interval=200)
            plt.show()
        else:
            for epoch in range(epochs):
                update(epoch)
            self.show_final_plots()

    def show_final_plots(self):
        self.update_plot_data(len(self.loss_history) - 1, self.loss_history[-1])
        plt.tight_layout()
        plt.show()