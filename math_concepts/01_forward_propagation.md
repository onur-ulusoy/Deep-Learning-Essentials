# Forward Propagation Explained

In this section, we will break down the math behind the forward propagation algorithm, which is the core of neural network training. Forward propagation is performed in every iteration during training and also in the testing phase.

## Neural Network Structure in Training

In any neural network, there is always an input layer and an output layer, with a variable number of hidden layers, which can range from 0 to infinity. To simplify the explanation and help visualize the concept, we will use a small neural network with the following structure: 3 neurons in the input layer, 4 neurons in a single hidden layer, 2 neurons in the output layer, 2 samples in a batch. This setup will allow me to illustrate how forward propagation works in a straightforward way, without losing key insights.

### Batches & Samples

A batch is a training unit that consists of a set of samples processed together in one iteration during the training of a model. Samples in a batch share the same weight and bias tensors (tensors are a type of multi-dimensional array) and contribute to adjusting these parameters together during training.

### Weights and Biases

They are parameters to be adjusted during training to give the best results out of approximated function. Except for input layer (actually units in input layers are not actually neurons), every neuron in every layer is connected each neuron in previous layer with a weight specicific to this connection. value of neuron is obtained via multiplying all these weights with activations of previous layer neurons and adding bias at the end. bias is the tensor specific to each layer. 



