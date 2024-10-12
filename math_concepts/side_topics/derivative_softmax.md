# Derivative of the Softmax Function with Respect to its Inputs

## Overview

In this document, we derive the partial derivative of the softmax output with respect to its input logits. This is a  step good to know, in understanding backpropagation through the softmax layer in neural networks.

## 1. Softmax Function Definition

The **softmax function** transforms logits (raw scores) into probabilities that sum to 1. For a given sample $k$ and class $j$, if we assume that our output layer or layer that we work on is indexed 2, the softmax function is defined as:

$$
Y_{\text{pred}}^{(k,j)} = \frac{e^{Z_2^{(k,j)}}}{\sum_{c=1}^{C} e^{Z_2^{(k,c)}}}
$$

where:
- $C$ is the total number of classes.
- $Z_2^{(k,j)}$ is the input logit for class $j$ of sample $k$.
- $Y_{\text{pred}}^{(k,j)}$ is the predicted probability for class $j$ of sample $k$.

For simplicity, we'll drop the sample index $k$ when considering a single sample:

$$
Y_{\text{pred}}^{(j)} = \frac{e^{Z_2^{(j)}}}{\sum_{c=1}^{C} e^{Z_2^{(c)}}}
$$

## 2. Computing the Partial Derivative

We aim to compute the partial derivative of the softmax output for class $j$ with respect to the input logit $Z_2^{(i)}$:

$$
\frac{\partial Y_{\text{pred}}^{(j)}}{\partial Z_2^{(i)}}
$$

This derivative quantifies how a small change in the input logit $Z_2^{(i)}$ affects the predicted probability $Y_{\text{pred}}^{(j)}$.



### Applying the Quotient Rule

The softmax function is a ratio of the exponential of the logit for class $j$ to the sum of exponentials of all logits. To compute the derivative, we'll use the **quotient rule**:

For functions $f(x)$ and $g(x)$:

$$
\frac{d}{dx} \left( \frac{f(x)}{g(x)} \right) = \frac{f'(x) \cdot g(x) - f(x) \cdot g'(x)}{[g(x)]^2}
$$

Apply this to the softmax function:

Let:
- $f(Z_2^{(j)}) = e^{Z_2^{(j)}}$
- $g(Z_2) = \sum_{c=1}^{C} e^{Z_2^{(c)}}$

Then:

$$
\frac{\partial Y_{\text{pred}}^{(j)}}{\partial Z_2^{(i)}} = \frac{f'(Z_2^{(i)}) \cdot g(Z_2) - f(Z_2^{(j)}) \cdot g'(Z_2^{(i)})}{[g(Z_2)]^2}
$$

Compute the derivatives:
- $f'(Z_2^{(i)}) = \frac{\partial e^{Z_2^{(j)}}}{\partial Z_2^{(i)}} = e^{Z_2^{(j)}} \cdot \delta_{ij}$
- $g'(Z_2^{(i)}) = \frac{\partial}{\partial Z_2^{(i)}} \sum_{c=1}^{C} e^{Z_2^{(c)}} = e^{Z_2^{(i)}}$

Here, $\delta_{ij}$ is the **Kronecker delta**:

$$
\delta_{ij} = 
\begin{cases}
1 & \text{if } i = j \\
0 & \text{if } i \ne j
\end{cases}
$$

Substitute these back into the derivative:

$$
\frac{\partial Y_{\text{pred}}^{(j)}}{\partial Z_2^{(i)}} = \frac{e^{Z_2^{(j)}} \delta_{ij} \cdot g(Z_2) - e^{Z_2^{(j)}} \cdot e^{Z_2^{(i)}}}{[g(Z_2)]^2}
$$

### Simplifying the Expression

Let $S = g(Z_2) = \sum_{c=1}^{C} e^{Z_2^{(c)}}$. Then:

$$
\frac{\partial Y_{\text{pred}}^{(j)}}{\partial Z_2^{(i)}} = \frac{e^{Z_2^{(j)}} \delta_{ij} \cdot S - e^{Z_2^{(j)}} e^{Z_2^{(i)}}}{S^2}
$$

Factor out $e^{Z_2^{(j)}}$:

$$
\frac{\partial Y_{\text{pred}}^{(j)}}{\partial Z_2^{(i)}} = \frac{e^{Z_2^{(j)}}}{S^2} \left( \delta_{ij} S - e^{Z_2^{(i)}} \right)
$$

Recognize that:

$$
Y_{\text{pred}}^{(j)} = \frac{e^{Z_2^{(j)}}}{S}, \quad Y_{\text{pred}}^{(i)} = \frac{e^{Z_2^{(i)}}}{S}
$$

Substitute these into the expression:

$$
\frac{\partial Y_{\text{pred}}^{(j)}}{\partial Z_2^{(i)}} = \frac{Y_{\text{pred}}^{(j)}}{S} \left( \delta_{ij} S - e^{Z_2^{(i)}} \right) = Y_{\text{pred}}^{(j)} \left( \delta_{ij} - Y_{\text{pred}}^{(i)} \right)
$$

### Final Result

Therefore, the partial derivative is:

$$
\frac{\partial Y_{\text{pred}}^{(j)}}{\partial Z_2^{(i)}} = Y_{\text{pred}}^{(j)} \left( \delta_{ij} - Y_{\text{pred}}^{(i)} \right)
$$

This formula shows that the derivative depends on whether $i = j$ or $i \ne j$:

- **When $i = j$**:

  $$
  \frac{\partial Y_{\text{pred}}^{(j)}}{\partial Z_2^{(j)}} = Y_{\text{pred}}^{(j)} (1 - Y_{\text{pred}}^{(j)})
  $$

- **When $i \ne j$**:

  $$
  \frac{\partial Y_{\text{pred}}^{(j)}}{\partial Z_2^{(i)}} = -Y_{\text{pred}}^{(j)} Y_{\text{pred}}^{(i)}
  $$

## 3. Matrix Representation of the Derivative

We can represent the derivatives for all classes using a **Jacobian matrix** $J$, where each element $J_{ji}$ is:

$$
J_{ji} = \frac{\partial Y_{\text{pred}}^{(j)}}{\partial Z_2^{(i)}} = Y_{\text{pred}}^{(j)} \left( \delta_{ij} - Y_{\text{pred}}^{(i)} \right)
$$

