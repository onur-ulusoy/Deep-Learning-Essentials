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