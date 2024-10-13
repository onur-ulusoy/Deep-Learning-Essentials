# Batch Normalization Explained

Batch Normalization is a technique used to improve the training of deep neural networks by normalizing the inputs to each layer. This normalization stabilizes the learning process and significantly reduces the number of training epochs required to train deep networks.

## Overview

The main idea behind Batch Normalization is to normalize the inputs of each layer so that they have a mean of zero and a variance of one. This helps in mitigating the problem of **internal covariate shift**, where the distribution of inputs to layers in a neural network changes during training, slowing down the training process.

## Forward Pass


**Given an input batch** $\mathbf{X} = \{ \mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(m)} \}^T$, **where** $m$ **is the batch size (number of samples), and each sample** $\mathbf{x}^{(i)}$ **is a** $1 \times n$ **vector**:

$$
\mathbf{x}^{(i)} = \begin{bmatrix} x_{k_1}^{(i)} & x_{k_2}^{(i)} & \dots & x_{k_n}^{(i)} \end{bmatrix}
$$

Batch Normalization performs the following steps during the forward pass:

### 1. Compute Batch Mean and Variance

For each feature (dimension) $k$:

- **Batch Mean**:
  $$
  \mu_k = \frac{1}{m} \sum_{i=1}^{m} x_k^{(i)}
  $$
  
- **Batch Variance**:
  $$
  \sigma_k^2 = \frac{1}{m} \sum_{i=1}^{m} \left( x_k^{(i)} - \mu_k \right)^2
  $$

Here, $x_k^{(i)}$ represents the value of the $k$-th feature (class) in the $i$-th sample.

### 2. Normalize the Batch

Normalize each input $\mathbf{x}^{(i)}$:

$$
\hat{x}_k^{(i)} = \frac{x_k^{(i)} - \mu_k}{\sqrt{\sigma_k^2 + \epsilon}}
$$

- $\epsilon$ is a small constant added for numerical stability.

### 3. Scale and Shift

Apply learnable scale ($\gamma_k$) and shift ($\beta_k$) parameters:

$$
y_k^{(i)} = \gamma_k \hat{x}_k^{(i)} + \beta_k
$$

- $\gamma_k$ and $\beta_k$ are parameters learned during training that allow the model to recover the original distribution if necessary.

### 4. Output

The normalized, scaled, and shifted output $\mathbf{y}^{(i)}$ is then passed to the next layer in the network.