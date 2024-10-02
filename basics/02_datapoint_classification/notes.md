# The Problem
This is a binary classification problem where data points are generated parametrically using the dedicated `SpiralData` class from `spiral_datapoint.py`. The data points are classified into two categories: blue (0) and red (1). The problem is addressed using a basic neural network structure.

## Neural Network Architecture
The neural network consists of an input layer, two hidden layers, and an output layer, with neuron sizes of 1x18x4x1. ReLU is used as the activation function in the hidden layers, while a sigmoid function is applied at the output layer. The architecture is largely influenced by the problem and its parameters. As this is a simple neural network intended for learning purposes, no fine-tuning, learning rate optimization, or regularization techniques have been applied.

## Scripts
I developed several scripts for this project:
- `nn.py`: This script implements the neural network using only NumPy to better understand its internal mechanics.
- `nn_torch.py`: Here, I used the Torch framework to implement the same neural network.
- `spiral_datapoint.py`: This script initializes the data points for training and testing.
- `tester.py`: Contains test classes for evaluating the models trained and saved using both `nn.py` and `nn_torch.py`.
- `train.py` and `train_and_test.ipynb`: These scripts implement the training process for both versions of the neural network.
- `training_comparison_exp.py`: An experimental script designed to explore the differences between the custom (NumPy-based) and Torch-based implementations of the training scripts.


