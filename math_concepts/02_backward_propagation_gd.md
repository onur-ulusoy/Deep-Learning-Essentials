# Backward Propagation and Gradient Descent Explained

In this section, we will break down the math behind the backward propagation process and the gradient descent algorithm. These two concepts form the backbone of neural network training and are foundational across all types of deep learning training. Backward propagation implements the gradient descent algorithm, executed in every iteration after forward propagation during training to progressively adjust model parameters and bring the output predictions closer to the correct labels.

## Logic & Calculations
We will perform calculations and explanations using the exact same neural network structure from the [Forward Propagation Explained](01_forward_propagation.md) section. Therefore, the network architecture and parameters will not be reintroduced here.

**Backpropagation** is a fundamental algorithm used to train neural networks by adjusting all the learnable parameters (such as weights and biases) in every layer. After obtaining predictions through **forward propagation**, backpropagation calculates how each parameter contributes to the loss, allowing the network to update these parameters to minimize the loss function.

To understand how a parameter affects the loss, backpropagation computes the **gradients**—the derivatives of the loss with respect to each parameter—using the **chain rule** from calculus. These gradients indicate the direction and magnitude by which each parameter should be adjusted to reduce the loss.

Once the gradients are computed, an **optimization algorithm** like **Gradient Descent** is employed to update the parameters. **Gradient Descent** is an optimization technique that iteratively adjusts the parameters in the opposite direction of the gradients to find a local minimum of the loss function.

In summary, backpropagation consists of two main steps in a simple neural network:
1. **Calculating Gradients:** Using the chain rule, backpropagation efficiently computes the gradients of the loss with respect to each parameter.
2. **Updating Parameters:** An optimization algorithm (e.g., Gradient Descent) uses these gradients to adjust the parameters in a way that minimizes the loss.


### Gradient Descent Details

The graph below, taken from [here](https://vitalflux.com/gradient-descent-explained-simply-with-examples/), illustrates how gradient descent works to minimize the cost (loss) function.

![Gradient Descent](img/gradient_descent.png)

Consider a weight $w$. The formula that expresses how it updates the parameter is:

$$
w_{\text{new}} = w_{\text{old}} - \alpha \left.\frac{\partial L}{\partial w}\right|_{w=w_{\text{old}}}
$$

where:

- **$w_{\text{old}}$**  
  The current value of the weight before the update. It's the starting point for the gradient descent step.

- **$w_{\text{new}}$**  
  The updated value of the weight after applying the gradient descent step.

- **$L$**  
  The cost (loss) function that measures the discrepancy between the predicted values and the actual values.

- **$\frac{\partial L}{\partial w}\bigg|_{w=w_{\text{old}}}$**  
  The derivative of the cost function with respect to the weight $w$, evaluated at $w = w_{\text{old}}$. This derivative represents the **gradient**, slope of that function at the current weight value, indicating the direction and rate at which $L$ changes with respect to $w$.

- **$\alpha$**  
  The learning rate, a hyperparameter that determines the size of the steps taken towards the minimum of the loss function. It controls the update strength, as depicted by the incremental step in the graph. 


