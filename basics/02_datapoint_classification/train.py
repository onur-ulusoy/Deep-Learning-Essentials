from basics.dataset_gen_scripts.spiral_datapoint import SpiralData
import numpy as np
from nn import NN
from nn_torch import NN_torch
from animation import AnimateTraining

np.random.seed(42)
# Create an instance of SpiralData
train_data = SpiralData(num_points=250, noise=0.2, revolutions=4)
train_data.generate_data()
#train_data.plot_data()

# Get the coordinates
(x1, y1), (x2, y2) = train_data.get_coordinates()

# X is a vstack of points, y is 1dim array of labels, 0 is blue and 1 is red
X,y = train_data.get_labeled_data() 
y = y.reshape(-1, 1)

# X.shape is (500,2)
# y.shape is (500,1)

# Neural Network Architecture
input_size = 2
hidden1_size = 18
hidden2_size = 4
output_size = 1

np.random.seed(41)

# Uncomment below to train numpy based nn and save model
network = NN(input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.1)
network.train(X,y,epochs=10000)
network.save_model()

# Uncomment below to train torch nn and save model
""" network_torch = NN_torch(input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.1, seed= 41)
network_torch.train_model(X,y, epochs=10000) """
#network_torch.plot_training()

# Uncomment below to train numpy based nn with animation functionality
# (Educational)
""" animator = AnimateTraining(network, X, y, real_time=False)
# Perform training and animate the process
animator.animate(epochs=10000, interval=100) """

""" network_torch.train_model(X,y,epochs=10000)
network_torch.plot_training() """
#network.train(X, y, epochs=1000)
