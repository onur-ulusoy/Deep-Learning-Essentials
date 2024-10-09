# Forward Propagation Explained

In this section, we will break down the math behind the forward propagation algorithm, which is the core of neural network training. Forward propagation is performed in every iteration during training and also in the testing phase.

## Neural Network Structure in Training

In any neural network, there is always an input layer and an output layer, with a variable number of hidden layers, which can range from 0 to infinity. To simplify the explanation and help visualize the concept, we will use a small neural network with the following structure: 3 neurons in the input layer, 4 neurons in a single hidden layer, 2 neurons in the output layer, 2 samples in a batch. This setup will allow me to illustrate how forward propagation works in a straightforward way, without losing key insights.

### Batches & Samples

A batch is a training unit that consists of a set of samples processed together in one iteration during the training of a model. Samples in a batch share the same weight and bias tensors (tensors are a type of multi-dimensional array) and contribute to adjusting these parameters together during training.

### Weights and Biases

They are parameters to be adjusted during training to give the best results out of approximated function. Except for input layer (actually units in input layers are not actually neurons), every neuron in every layer is connected each neuron in previous layer with a weight specicific to this connection. value of neuron is obtained via multiplying all these weights with activations of previous layer neurons and adding bias at the end. bias is the tensor specific to each layer. 

### Activations

In every layer after input layer, an element-wise activation function is applied to the neurons, giving the system its nonlinear properties. The choice of activation function depends on the specific problem, and there are many options available, such as ReLU, Sigmoid, Tanh, etc.

## Layer Calculations

Below is an illustration of a neural network during the training phase. 

- **Blue dots** represent neurons. Neurons on the same horizontal axis belong to the same sample, while neurons on the vertical axis represent different samples.
- **Green arrows** depict the connections between neurons, which carry weights (for simplicity, only connections to two target neurons are illustrated).
- **W1 and W2** are the weight tensors between layers.
- **b1 and b2** are the bias tensors added to each layer's activations.


![NN layers in training](img/nn_layers_training.png)

You can think of these layers as matrix representations in this example. If you look at the front view, the input layer is a 2x3, the hidden layer is a 2x4, and the output layer is a 2x2 matrix. Each connection has a weight value, and the value of a single neuron is calculated as follows. For example, the first neuron in the hidden layer is computed as:

For the first sample:
$$ z_{11} = x_{11} \cdot w_{11} + x_{12} \cdot w_{21} + x_{13} \cdot w_{31} + b_{11} $$

Similarly, for the second neuron in the hidden layer:
$$ z_{12} = x_{11} \cdot w_{12} + x_{12} \cdot w_{22} + x_{13} \cdot w_{32} + b_{12} $$

We continue with the rest of the neurons in the hidden layer:
$$ z_{13} = x_{11} \cdot w_{13} + x_{12} \cdot w_{23} + x_{13} \cdot w_{33} + b_{13} $$

$$ z_{14} = x_{11} \cdot w_{14} + x_{12} \cdot w_{24} + x_{13} \cdot w_{34} + b_{14} $$

For the second sample, the process is similar:
$$ z_{21} = x_{21} \cdot w_{11} + x_{22} \cdot w_{21} + x_{23} \cdot w_{31} + b_{11} $$

$$ z_{22} = x_{21} \cdot w_{12} + x_{22} \cdot w_{22} + x_{23} \cdot w_{32} + b_{12} $$

$$ z_{23} = x_{21} \cdot w_{13} + x_{22} \cdot w_{23} + x_{23} \cdot w_{33} + b_{13} $$

$$ z_{24} = x_{21} \cdot w_{14} + x_{22} \cdot w_{24} + x_{23} \cdot w_{34} + b_{14} $$

### Matrix Representation

We can represent these equations in matrix form as:

**Input matrix (X)** of size $2 \times 3$:
$$ X = \begin{bmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \end{bmatrix} $$

**Weight matrix ($W_1$)** of size $3 \times 4$:
$$ W_1 = \begin{bmatrix} w_{11} & w_{12} & w_{13} & w_{14} \\ w_{21} & w_{22} & w_{23} & w_{24} \\ w_{31} & w_{32} & w_{33} & w_{34} \end{bmatrix} $$

**Bias vector ($b_1$)** of size $1 \times 4$:
$$ b_1 = \begin{bmatrix} b_{11} & b_{12} & b_{13} & b_{14} \end{bmatrix} $$

**Output ($Z_1$)** is the resulting matrix after applying forward propagation to the hidden layer:
$$ Z_1 = \begin{bmatrix} z_{11} & z_{12} & z_{13} & z_{14} \\ z_{21} & z_{22} & z_{23} & z_{24} \end{bmatrix} $$

### Matrix Multiplication for Forward Propagation

Using matrix multiplication and adding the bias term, we can express the output $Z_1$ as:
$$ Z_1 = X \cdot W_1 + b_1 $$

**Note:** In this operation, when we add the bias vector $b_1$ of size $1 \times 4$ to the resulting matrix $Z_1$ from $X \cdot W_1$ of size $2 \times 4$, broadcasting is utilized. Broadcasting allows $b_1$ to be effectively expanded to match the dimensions of $Z_1$, so that each element of $b_1$ is added to every row of the matrix $Z_1$. This enables the operation to be valid even though the two matrices originally have different dimensions.

### Key Insights

- The same set of weights $W_1$ and biases $b_1$ are used for both samples in the batch.
- The forward propagation algorithm applies the weights and biases individually to each sample, meaning that each neuron in the hidden layer is calculated using only the corresponding neurons from the input layer.
- No information is shared between samples during forward propagation. Each sample generates its own hidden layer activations based on the entire weight matrix $W_1$, but without cross-sample interaction.

### Numerical Example using NumPy

Let's implement the previous example using NumPy to see concrete results. This structure could represent a basic classification problem, such as classifying points in a 3D space (with 3 input feature coordinates) into two categories: red or blue. The output will be two neurons representing these two categories, with softmax activation applied to the output layer. Softmax is commonly used in binary classification problems because it outputs probabilities that represent the likelihood of belonging to each class (either blue or red).

#### Architecture Overview:

- **Input Layer:** 3 input features (e.g., 3D coordinates)
- **Hidden Layer:** 1 hidden layer with 4 neurons and ReLU activation
- **Output Layer:** 2 neurons in the output layer, with softmax activation for classification

It’s important to note that this example is solely for demonstration purposes to understand the logic behind forward propagation and is not intended for actual training. With only 2 samples, proper training is not feasible. However, this setup will allow us to observe the forward pass of data through the network.

#### Network Architecture Definition:

```python
# Define network architecture
input_size = 3
hidden_size = 4
output_size = 2  # Number of classes, either blue or red
```

#### Initialization:

We begin by initializing our `X_train` matrix (randomly generated for this example), which fits the input layer with a shape of 2x3:

```python
# Create training data with 2 samples
# Inputs: 2 samples, each with 3 integer features
X_train = np.random.randint(0, 10, (2, input_size))
```

This will output `X_train` like so:

```
Inputs:
 [[2 8 6]
 [1 2 3]]
```

Next, we define the `y_train` matrix, representing the true labels corresponding to `X_train`. This fits the output layer with a shape of 2x2 (one-hot encoded labels for two classes):

```python
# Outputs: 2 samples, each with 2 binary labels (one-hot encoded)
# For simplicity, let's create random one-hot encoded labels
y_train_indices = np.random.randint(0, output_size, 2)
y_train = np.zeros((2, output_size))
y_train[np.arange(2), y_train_indices] = 1
```

This will output `y_train` like so:

```
Labels (One-Hot Encoded):
 [[1. 0.]
 [0. 1.]]
```

So, we have samples [2 8 6] with output label [1. 0.] and [1 2 3], with output label [0. 1.], totalling of 2 samples.

We initialize weights and biases with small random values. Weight and bias initialization is a broader topic, so we will keep it simple for now:

```python
# Initialize weights and biases with small random values
# Parameters between input and hidden layers
self.W1 = np.random.randn(input_size, hidden_size) * 0.01
self.b1 = np.zeros((1, hidden_size))

# Parameters between hidden and output layers
self.W2 = np.random.randn(hidden_size, output_size) * 0.01
self.b2 = np.zeros((1, output_size))
```

This will output the following weights and biases:

```
W1 shape: (3, 4)
W1: [[-0.00885127 -0.00528875 -0.00062892  0.01330022]
     [ 0.00045693  0.00057812  0.00184987  0.00878082]
     [-0.00098337  0.01788674 -0.00878302  0.00338431]]

b1 shape: (1, 4)
b1: [[0. 0. 0. 0.]]

W2 shape: (4, 2)
W2: [[ 0.00263928 -0.00881749]
     [-0.0067974  -0.00869512]
     [-0.00402919  0.01688775]
     [-0.01814645 -0.00337234]]

b2 shape: (1, 2)
b2: [[0. 0.]]
```

#### Forward Propagation:

As explained earlier in matrix notation, we perform operations to find `z1` and `z2`, which are the pre-activation neuron value tensors in the hidden and output layers respectively (`z1` for hidden layer, `z2` for output layer).

```python
self.z1 = np.matmul(X, self.W1) + self.b1
```

After performing the matrix operations, we get:

```
z1: [[-0.01994736  0.10459831 -0.03915699  0.09200355]
     [-0.01088753  0.05057299 -0.02327823  0.03287711]]
```

Next, we apply the ReLU activation function, which returns the value itself if it's greater than 0 and 0 otherwise, to obtain the activated layer tensors.

```python
self.a1 = self.relu(self.z1)  # Activation for hidden layer
```

After applying the activation, we get the `a1` matrix. As we can see, the element-wise activation function has been applied:

```
a1: [[0.          0.10459831  0.          0.09200355]
     [0.          0.05057299  0.          0.03287711]]
```

