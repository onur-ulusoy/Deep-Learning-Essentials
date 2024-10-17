# The Problem
This is a regression problem where training and testing data points are generated using a dedicated script for polynomial data generation. The goal was to create a neural network structure that is prone to overfitting, allowing me to observe and address overfitting issues to improve the model's generalization using regularization techniques.

## Neural Network Architecture
The neural network consists of an input layer, two hidden layers, and an output layer, with neuron sizes of 1x88x48x1. ReLU is used as the activation function in the hidden layers, while a linear function is applied at the output layer. I made hidden layer sizes large so that it could overfit faster.

## Scripts
- `nn.py`: Implements the neural network using only NumPy, providing a deeper understanding of its internal mechanics.
- `nn_torch.py`: Implements the same neural network using the PyTorch framework for easier deployment and comparison.
- `hyperparams.py`: Initializes and stores hyperparameters for consistent use across both training and testing scripts.
- `polynomial_data.py`: Generates polynomial data points in a parametrized way for training and testing (moved [here](/basics/dataset_gen_scripts/polynomial_data.py)).
- `test.py`: Contains a model testing class to evaluate both the NumPy-based (`nn.py`) and PyTorch-based (`nn_torch.py`) networks.
- `training.py`: Implements the training process for both `nn.py` and `nn_torch.py` models.

## Experiences & Difficulties I Encountered
I started by developing a NumPy-based neural network, similar to the approach I used in the previous problem, implementing the training components step by step. For this regression problem, I used ReLU for the hidden layers and a linear activation function for the output layer, which is standard for regression tasks.

I was able to parameterize the polynomial data points, allowing me to experiment with different neural network architectures and data point ranges to facilitate learning and promote overfitting. Since no normalization was applied to the data, I tried to select x and y values that were naturally close to a normal distribution. When I scaled the data too far from ideal values (by increasing or decreasing the scale factor, which I had implemented in the scripts), the gradient descent algorithm struggled to reduce the loss and fit the polynomial. **I plan** to address this issue by applying normalization in future iterations.

Initially, I considered using a 5th-degree polynomial for the problem, but this proved too challenging with the current network architecture. As my primary focus was on overfitting, I opted for a 2nd-degree polynomial to make the problem easier to solve. **I also plan** to explore how to modify the neural network to approximate higher-degree polynomials, such as 5th degree or beyond.

Below is a graph showing my training dataset.

![Training Dataset](img/training_dataset.png)

## Overfitting
Overfitting occurs when a model learns the training dataset too well, achieving a very low training loss. However, because the model cannot generalize the learned patterns, it struggles on unseen data, leading to a higher test loss. In such cases, the model effectively memorizes the training data rather than learning a flexible solution that generalizes to new inputs.

I trained the model for 50,000 epochs, used learning rate 0.001 achieving a final training loss of **0.0206**. When tested on data with the **same noise seed** as the training set, the model produces the same mean squared error (MSE) loss of **0.0206**, successfully approximating the polynomial function and noise pattern.

![Overfitting with Same Noise Seed](img/overfit_same_seed.png)

However, when tested with a **different noise seed**, the model reveals its inability to generalize. The model still tries to approximate values specific to the training noise seed, and this issue persists across different noise variations. Instead of generalizing the training data, the model memorized it. As shown in the plot below, the red curves are nearly identical to those in the previous plot, indicating that the model is overfitted. The loss with the new noise seed is **0.0570**, nearly three times the training loss, highlighting how the model is highly specialized to the original seed rather than being adaptable.

![Overfitting with Different Noise Seed](img/overfit_different_seed.png)


## Bias, Variance, Underfitting, and Overfitting
- **Bias** refers to the assumptions made by the model about the relationship between input and output. High bias means the model is overly simplistic and may not capture the complexity of the data, leading to underfitting.
- **Variance** measures the sensitivity of the model to variations in the training data. High variance usually occurs with complex models that closely fit the training data, but struggle to generalize to new data, leading to overfitting.

There is a tradeoff between bias and variance:
- **High bias** results in the model making too many assumptions, which can cause it to underfit on unseen data, meaning it won't capture the underlying patterns in the data.
- **High variance** causes the model to be too sensitive to the training data, memorizing it rather than learning general patterns. This can lead to overfitting, where the model performs well on training data but poorly on test data.

The figure below, taken from [Towards Data Science](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229), illustrates the bias-variance tradeoff clearly.

![Bias-Variance Tradeoff](img/bias_variance.png)

Additionally, this [YouTube video](https://www.youtube.com/watch?v=a6YH6EbM9xA&list=LL&index=5) provides a great summary of bias and variance.

### Ways to Prevent Underfitting:
- Train the model for more epochs.
- Increase the complexity of the model (e.g., more neurons or layers).
- Improve the network architecture to better suit the problem.

### Ways to Prevent Overfitting:
- Train with more data.
- Apply regularization techniques, such as L1 or L2 regularization.
- Use dropout or early stopping during training.
- Improve the network architecture to better balance complexity and generalization.


## Regularization
To address overfitting and improve generalization within the same number of training epochs, I implemented L2 regularization in both the NumPy-based and PyTorch-based neural networks.

**Regularization** is a technique designed to prevent the model's weights from growing too large by adding a penalty term to the loss function. This discourages the network from relying too heavily on any single input, ensuring that the model generalizes better to unseen data. When weights become too large, the loss increases, and during backpropagation, the model is penalized (in a way similar to negative reinforcement in reinforcement learning). As a result, the model tends to balance its focus across inputs rather than overfitting to specific patterns.

- **L1 Regularization**: Adds a penalty proportional to the absolute value of the weights, driving them towards zero.
- **L2 Regularization**: Adds a penalty proportional to the square of the weights, penalizing larger weights more heavily, which helps smooth out the model's predictions.

### L2 Regularization in NumPy Implementation
In the NumPy-based network, I implemented L2 regularization by calculating the sum of the squared weights and adding it to the total loss function. 

I incorporated the L2 term during backpropagation, adjusting the weight gradients accordingly. This additional term arises naturally from applying the chain rule to the L2-regularized loss function. The division by 2 is a common practice in optimization because it simplifies the derivative of the squared weights during backpropagation.

```python
self.dw3 = np.matmul(self.a2.T, dz3) / m + self.l2_lambda / m * self.W3
```

This ensures that the gradient updates account for the regularization term, keeping the weights from becoming too large.

In order to print losses correctly, here's how I computed the L2 penalty:

```python
weights_sum = np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)) + np.sum(np.square(self.W3))
# Compute L2 regularization term
l2_loss = self.l2_lambda / (2 * m) * weights_sum
loss += l2_loss
```

Where:
- `self.W1`, `self.W2`, and `self.W3` are the weights of the layers.
- `m` is the number of samples in the batch.
- `self.l2_lambda` is the regularization strength.

### Results with L2 Regularization
After testing, I observed that the model learned the pattern in a more generalized way, reducing overfitting and improving performance on unseen data. The figures below illustrate the results of testing with the **same noise seed** and with a **different noise seed** after applying L2 regularization:

- **Same Noise Seed:**

![L2 Regularization with Same Seed](img/l2reg_same_seed.png)

- **Different Noise Seed:**

![L2 Regularization with Different Seed](img/l2reg_different_seed.png)


### Dropout
After implementing L2 regularization, I also decided to apply the dropout technique for further improvement in generalization.

Dropout works by deactivating a random subset of neurons during each forward and backward propagation step in training. Each neuron has a probability of being "dropped," meaning it won't participate in that particular pass. This technique prevents the model from becoming overly reliant on any specific neurons, promoting more robust learning and better generalization. In essence, it serves as another form of regularization by randomly setting some weights to zero, forcing the model to distribute learning across multiple features.

In the NumPy-based neural network, I implemented a mask to deactivate neurons with a certain probability. This approach zeroes out the activations of randomly selected neurons, preventing them from transmitting any information to subsequent layers. Here's an example of how I implemented it:

```python
if training:
    # Apply dropout to a1
    self.dropout_mask1 = (np.random.rand(*self.a1.shape) > self.dropout_p1) / (1.0 - self.dropout_p1)
    self.a1 *= self.dropout_mask1
```

This code ensures that, during training, a fraction of the neurons in layer one are deactivated based on the dropout probability `dropout_p1`. In the backward pass, the same mask is applied to ensure consistency:

```python
# Apply dropout mask to dz2
dz2 *= self.dropout_mask2
```

### Results with L2 Regularization & Dropout
As expected, combining L2 regularization with dropout yielded better results. The model was able to generalize more effectively, approximating the function with improved flexibility.

Below are the plots demonstrating the results with L2 regularization and dropout applied:

- **Same Noise Seed:**

![L2 Regularization with Dropout and Same Seed](img/l2reg_dropout_same_seed.png)

- **Different Noise Seed:**

![L2 Regularization with Dropout and Different Seed](img/l2reg_dropout_different_seed.png)


## Cons and Pros of Regularization and Dropout
These are the observations and trade-offs I encountered during the implementation of regularization and dropout techniques.

### Cons:
- **Slower Convergence**: After implementing dropout, the decrease in loss during the training stage became slower. The network struggled to converge to a lower loss value.
- **Slight Underfitting**: For some data points, the model slightly underfit. If I had further increased the L2 regularization coefficient or dropout probabilities, it might have underfit even more.
- **Higher Final Loss**: The total loss at the end of training was higher than without regularization or dropout, which indicates a trade-off between preventing overfitting and maintaining low training loss.

### Pros:
- **Prevention of Overfitting**: Regularization and dropout successfully mitigated overfitting. The model generalized better to unseen data, capturing the broader pattern.
- **Improved Test Performance**: Test loss was lower than before, and, interestingly, it was even lower than the training loss, showing the model's enhanced ability to generalize beyond the training data.

## What I Learned

- **Importance of Regularization**: Regularization, especially L2, is crucial in preventing overfitting. It penalizes large weights, encouraging the model to find more balanced solutions and generalize better.
  
- **Dropout's Effectiveness**: Dropout proved to be a powerful technique to prevent over-reliance on specific neurons, helping the model to avoid overfitting and improve its generalization.

- **Trade-offs Between Bias and Variance**: Balancing bias and variance is critical. Higher regularization and dropout reduce variance (overfitting) but may increase bias, leading to potential underfitting.

- **Parameter Tuning**: The choice of regularization coefficient and dropout probabilities has a significant impact. Too much regularization or dropout can lead to underfitting, while too little results in overfitting.

- **Learning Rate Sensitivity**: Without normalization or scaling the data properly, the model struggled with gradient descent, making it harder to converge to a lower loss, highlighting the importance of proper data scaling.

- **Understanding Neural Network Mechanics**: Building neural networks from scratch using NumPy deepened my understanding of forward and backward propagation, weight updates, and gradient flow, which helped me debug issues effectively.

- **Architecture Design**: Layer and neuron size selection has a big impact on learning capacity. Larger layers led to faster overfitting, but adding regularization and dropout addressed this.

- **Generalization vs. Memorization**: Overfitting led the model to memorize the training data, while regularization techniques forced the model to learn the underlying patterns, enabling better performance on unseen data.
  

## Normalization of y Values Update
At the beginning of this problem, I initially attempted to model a 5th-degree polynomial. However, increasing the polynomial degree also increased the range of y-axis values (e.g., from -1000 to 1000). Since I was using Mean Squared Error (MSE) as the loss function, the loss values became excessively high, proportional to the square of the errors. This led to an **exploding gradient problem**, where the system couldn't stabilize the gradients, preventing it from minimizing the error effectively. 

To avoid this issue, I initially worked with manually selected y values in an optimal range. Later, I implemented a scaling technique to normalize the y values in the training script. Before passing inputs to the neural network, I scaled them using the following approach:

```python
y_original_min = y.min()
y_original_max = y.max()
y_train_scaled = (y - y_original_min) / (y_original_max - y_original_min)
```

Additionally, I modified the model saving process in the neural network classes to store `y_original_min` and `y_original_max`. This allowed me to restore these parameters when testing the model after loading it, ensuring I could convert the scaled predictions back to their original values:

```python
y_pred_original = y_pred_scaled * (self.y_original_max - self.y_original_min) + self.y_original_min
```

Of course, `y_pred_scaled` is obtained from the forward propagation.

I calculated the MSE error using the scaled values for both training and testing, like so:

```python
mse = np.mean((y_test_scaled - y_pred_scaled)**2)
```

### Adjustments Post-Normalization
After implementing normalization, I was able to increase the learning rate from 0.1 to 0.3, as it worked better for fitting a 5th-degree polynomial. The previous learning rate of 0.1 caused underfitting and high bias when working with more complex functions like the 5th-degree polynomial, although 0.1 remains better suited for 2nd or 3rd-degree functions. The learning rate should be chosen based on the complexity of the problem.

I also reduced the L2 regularization coefficient from 0.02 to 0.01 and lowered the dropout probability from 0.25 to 0.05. Normalizing the y values decreased the need for such high regularization and dropout values.

Below is the plot showing the model fitting a 5th-degree polynomial perfectly, even with a different noise seed:

![5th Degree Polynomial After Normalization](img/after_normalization.png)


## Additional Aspects
The NumPy-based neural network implementation appears to be quite robust, consistently lowering the loss value in a matter of seconds. However, the Torch-based implementation is noticeably slower and more sensitive to certain factors like the random seed, as well as the initialization of weights and biases. In some cases, this sensitivity even causes the Torch network to fail during training, resulting in NaN loss values.

To improve the robustness of the Torch-based network and mitigate the impact of these issues, I plan to implement batch normalization in the next step. This technique should help stabilize the learning process and make the network less dependent on specific initialization and random seeds.
