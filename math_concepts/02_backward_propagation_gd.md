# Backward Propagation and Gradient Descent Explained

In this section, we will break down the math behind the backward propagation process and the gradient descent algorithm. These two concepts form the backbone of neural network training and are foundational across all types of deep learning training. Backward propagation implements the gradient descent algorithm, executed in every iteration after forward propagation during training to progressively adjust model parameters and bring the output predictions closer to the correct labels.

## Logic & Calculations
We will perform calculations and explanations using the exact same neural network structure from the [Forward Propagation Explained](01_forward_propagation.md) section. Therefore, the network architecture and parameters will not be reintroduced here.

Backpropagation aims to adjust all the learnable parameters in every layer of a neural network after obtaining predictions from forward propagation. It updates the parameters to minimize the loss. To understand how a parameter affects the loss, it uses the gradient descent algorithm.

Backpropagation consists of two main components in a simple neural network: calculating the gradients (using the chain rule) and updating the parameters based on these gradients.



