# Batch Normalization Explained

Batch Normalization is a technique used to improve the training of deep neural networks by normalizing the inputs to each layer. This normalization stabilizes the learning process and significantly reduces the number of training epochs required to train deep networks.

## Overview

The main idea behind Batch Normalization is to normalize the inputs of each layer so that they have a mean of zero and a variance of one. This helps in mitigating the problem of **internal covariate shift**, where the distribution of inputs to layers in a neural network changes during training, slowing down the training process.

## Forward Pass


**Given an input batch** $\mathbf{X} = \{ \mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(m)} \}^T$, **where** $m$ **is the batch size (number of samples), and each sample** $\mathbf{x}^{(i)}$ **is a** $1 \times n$ **vector**:

$$
\mathbf{x}^{(i)} = \begin{bmatrix} k_1^{(i)} & k_2^{(i)} & \dots & k_n^{(i)} \end{bmatrix}
$$