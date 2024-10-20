# The Problem

In this section, we'll address critical issues we encountered in neural networks during previous episodes by using normalization/standardization techniques. Specifically, we aim to solve the challanges in following problems:

1. Polynomial curve fitting (regression) [notes](/basics/03_overfitting_regularization/notes.md)
2. Spiral data point classification (classification) [notes](/basics/02_datapoint_classification/notes.md)

Across both problems, I observed the following challenges:

1. Particularly in the classification problem, the training outcome was highly dependent on initial conditions, such as initialization of weights and biases depended on the random seed. Small changes in these could make the gradients unstable (divergent) or cause them to converge too slowly to a minimum or stuck in local minimum. It was crucial to know the right starting point and the appropriate learning rate to ensure proper convergence.
   
2. I could not increase the learning rate beyond 0.1 without encountering unstable (exploding) gradients. This caused the weights and biases to diverge from the minimum loss points, resulting in extremely slow training — achieving the desired network took up to 50k epochs.

3. I struggled to set the true values of dataset on a large scale, particularly in the regression problem, as doing so caused the loss to spike dramatically, making exploding gradients inevitable.

## Unstable Gradients

For effective training in the gradient descent algorithm, the gradients must point towards the minimum value of the loss function, guiding the network's weights and biases to converge to that point. During each step of gradient descent, parameters (weights and biases) are updated proportionally to their corresponding gradients, steering them toward the optimal values that minimize the loss.

Consider a weight $w$. The formula for updating this parameter during gradient descent is:

$$
w_{\text{new}} = w_{\text{old}} - \alpha \left.\frac{\partial L}{\partial w}\right|_{w=w_{\text{old}}}
$$

This derivative represents the **gradient** of the loss with respect to the weight. For a more in-depth explanation of gradient descent, refer to the section [Backward Propagation Explained](/math_concepts/02_backward_propagation_gd.md).

### Vanishing Gradients

The **vanishing gradient problem** occurs when the gradients in certain layers of the network become extremely small, often less than 1. Since these gradients are used to compute further gradients through the chain rule (backpropagation), multiplying small gradients together results in even smaller values. By the time these values reach the earlier layers of the network, they can be close to zero.

Additionally, when updating parameters using gradient descent, the already-small gradient is multiplied by the learning rate, which is typically a small value (such as 0.1 or 0.001). This results in minimal or no change to the weights, effectively "freezing" the learning process. As a result, the weights and biases hardly change, impairing the network’s ability to learn and converge effectively.


### Exploding Gradients

Similar to the vanishing gradient problem, the **exploding gradient problem** occurs when gradients are excessively large. When the gradients for multiple layers are greater than 1, their product becomes even larger, leading to exponential growth as backpropagation moves through the network. 

In the context of our regression problem, since the true values in the dataset are large, the mean squared error (MSE) becomes significantly higher. This inflates the gradients, as they are directly dependent on the loss. Consequently, when the gradient descent algorithm updates the weights, the weight adjustments are disproportionately large. Instead of converging towards the minimum point of the loss function, the weights may overshoot, potentially oscillating with increasing amplitude, much like an unstable control system, or diverging entirely.

This makes the model unable to stabilize and learn effectively, leading to instability and divergence rather than convergence.

These YouTube videos offer a clear explanation of the unstable gradient problem:

- [Vanishing & Exploding Gradient Explained | A problem resulting from backpropagation](https://www.youtube.com/watch?v=qO_NLVjD6zE)
- [The Fundamental Problem with Neural Networks - Vanishing Gradients](https://www.youtube.com/watch?v=ncTHBi8a9uA&t)


