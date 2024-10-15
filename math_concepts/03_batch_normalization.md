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

After this stage, elements of the tensor is between values [-1, 1]
### 3. Scale and Shift

Apply learnable scale ($\gamma_k$) and shift ($\beta_k$) parameters:

$$
y_k^{(i)} = \gamma_k \hat{x}_k^{(i)} + \beta_k
$$

- $\gamma_k$ and $\beta_k$ are parameters learned during training (in backpropagation) that allow the model to recover the original distribution if necessary.

### 4. Output

The normalized, scaled, and shifted output $\mathbf{y}^{(i)}$ is then passed to the next layer in the network.

## Running Estimates for Inference (Testing)

During training, we maintain running estimates of the mean and variance for use during inference (testing):

- **Running Mean**:
  $$
  \mu_{\text{running}, k} = \alpha \mu_{\text{running}, k} + (1 - \alpha) \mu_k
  $$
  
- **Running Variance**:
  $$
  \sigma_{\text{running}, k}^2 = \alpha \sigma_{\text{running}, k}^2 + (1 - \alpha) \sigma_k^2
  $$

- $\alpha$ is the momentum term (typically close to 1, e.g., 0.9 or 0.99).

During inference, the normalization uses these running estimates:

$$
\hat{x}_k^{(i)} = \frac{x_k^{(i)} - \mu_{\text{running}, k}}{\sqrt{\sigma_{\text{running}, k}^2 + \epsilon}}
$$


### Why is Batch Normalization Also Used in Testing?

In **training**, batch normalization (BN) normalizes the input using the **batch mean** and **batch variance**, which helps in stabilizing and speeding up training by keeping the activations within a certain range.

However, during **testing**, we don't have the luxury of computing statistics from batches since inference is typically done on one or a few samples. Instead, we use the **running mean** and **running variance** that were estimated during training.

The key reasons for using batch normalization during testing are:

1. **Consistency:** By using the running statistics (mean and variance), the model maintains consistency between training and testing. If we used batch statistics during testing, the model could behave unpredictably since the statistics on small or individual test samples might not match the distribution learned during training.

2. **Stability:** The running mean and variance are smoother, more reliable estimates of the true data distribution. They ensure that even if input data during testing is not exactly like the training set, the model's behavior remains stable and well-generalized.

In summary, batch normalization during testing ensures that the model leverages the learned distribution (via running mean and variance) for consistent and robust inference.

## Backward Pass

During backpropagation, we need to compute the gradients of the loss $L$ with respect to the inputs $x_k^{(i)}$, as well as $\gamma_k$ and $\beta_k$.

Let:

- $\delta_k^{(i)} = \frac{\partial L}{\partial y_k^{(i)}}$ (gradient from upstream).

$\delta_k^{(i)}$ represents the gradient of the loss $L$ with respect to the output $y_k^{(i)}$ from the Batch Normalization layer. It is also called the upstream gradient, indicating the error signal coming from the next layer during backpropagation. It is `dout` in `batch_norm_backward` method in [example neural network](/basics/04_batch_normalization/nn.py).

$y_k^{(i)}$: This represents the output of the Batch Normalization layer for feature $k$ and sample $i$ after applying normalization, scaling by $\gamma_k$, and shifting by $\beta_k$. Essentially, $y_k^{(i)}$ is the final output after Batch Normalization is applied to the input $x_k^{(i)}$.

### 1. Gradients with Respect to Scale and Shift Parameters

- **For $\gamma_k$**:
  $$
  \frac{\partial L}{\partial \gamma_k} = \sum_{i=1}^{m} \delta_k^{(i)} \hat{x}_k^{(i)}
  $$

- **For $\beta_k$**:
  $$
  \frac{\partial L}{\partial \beta_k} = \sum_{i=1}^{m} \delta_k^{(i)}
  $$

### 2. Gradient with Respect to Normalized Input

$$
\frac{\partial L}{\partial \hat{x}_k^{(i)}} = \delta_k^{(i)} \gamma_k
$$

### 3. Gradients with Respect to Input

Compute $\frac{\partial L}{\partial x_k^{(i)}}$:

$$
\frac{\partial L}{\partial x_k^{(i)}} = \frac{1}{m} \frac{1}{\sqrt{\sigma_k^2 + \epsilon}} \left( m \frac{\partial L}{\partial \hat{x}_k^{(i)}} - \sum_{j=1}^{m} \frac{\partial L}{\partial \hat{x}_k^{(j)}} - \hat{x}_k^{(i)} \sum_{j=1}^{m} \left( \frac{\partial L}{\partial \hat{x}_k^{(j)}} \hat{x}_k^{(j)} \right) \right)
$$

This equation can be derived step by step.

#### Derivation Steps

**Let**:

- $\hat{x}_k^{(i)} = \frac{x_k^{(i)} - \mu_k}{\sqrt{\sigma_k^2 + \epsilon}}$
- $\sigma_k^2 + \epsilon = \tilde{\sigma}_k^2$

**Compute $\frac{\partial L}{\partial x_k^{(i)}}$**:

1. **Gradient w.r.t. $x_k^{(i)}$**:

   $$
   \frac{\partial L}{\partial x_k^{(i)}} = \frac{\partial L}{\partial \hat{x}_k^{(i)}} \cdot \frac{\partial \hat{x}_k^{(i)}}{\partial x_k^{(i)}}
   $$

2. **Compute $\frac{\partial \hat{x}_k^{(i)}}{\partial x_k^{(i)}}$**:

   $$
   \frac{\partial \hat{x}_k^{(i)}}{\partial x_k^{(i)}} = \frac{1}{\sqrt{\tilde{\sigma}_k^2}} - \frac{(x_k^{(i)} - \mu_k)}{(\tilde{\sigma}_k^2)^{3/2}} \cdot \frac{\partial \sigma_k^2}{\partial x_k^{(i)}}
   $$
   
   Since $\sigma_k^2 = \frac{1}{m} \sum_{j=1}^{m} (x_k^{(j)} - \mu_k)^2$, the derivative $\frac{\partial \sigma_k^2}{\partial x_k^{(i)}}$ involves terms from all samples in the batch.

3. **Compute $\frac{\partial \mu_k}{\partial x_k^{(i)}}$ and $\frac{\partial \sigma_k^2}{\partial x_k^{(i)}}$**:

   - $\frac{\partial \mu_k}{\partial x_k^{(i)}} = \frac{1}{m}$
   - $\frac{\partial \sigma_k^2}{\partial x_k^{(i)}} = \frac{2}{m} (x_k^{(i)} - \mu_k) - \frac{2}{m} \sum_{j=1}^{m} (x_k^{(j)} - \mu_k) \cdot \frac{\partial \mu_k}{\partial x_k^{(i)}}$

4. **Assemble the Gradient**:

   After computing the necessary partial derivatives, substitute back to find $\frac{\partial L}{\partial x_k^{(i)}}$.

### 4. Simplified Vectorized Form

In practice, we implement a vectorized version for efficiency.

Let:

- $\mathbf{\hat{X}}$: Matrix of normalized inputs
- $\mathbf{\delta}$: Matrix of upstream gradients

Compute:

1. **Gradients w.r.t. $\gamma$ and $\beta$**:

   $$
   \frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m} \mathbf{\delta}^{(i)} \odot \mathbf{\hat{X}}^{(i)}
   $$
   
   $$
   \frac{\partial L}{\partial \beta} = \sum_{i=1}^{m} \mathbf{\delta}^{(i)}
   $$

2. **Gradient w.r.t. $\hat{\mathbf{X}}$**:

   $$
   \frac{\partial L}{\partial \hat{\mathbf{X}}} = \mathbf{\delta} \odot \gamma
   $$

3. **Gradient w.r.t. Input $\mathbf{X}$**:

   $$
   \frac{\partial L}{\partial \mathbf{X}} = \frac{1}{m} \left( \tilde{\sigma}^{-1} \left( m \frac{\partial L}{\partial \hat{\mathbf{X}}} - \sum \frac{\partial L}{\partial \hat{\mathbf{X}}} - \hat{\mathbf{X}} \odot \sum \left( \frac{\partial L}{\partial \hat{\mathbf{X}}} \odot \hat{\mathbf{X}} \right) \right) \right)
   $$

- $\odot$ denotes element-wise multiplication.
- Summations are over the batch dimension.