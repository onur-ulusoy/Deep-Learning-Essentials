# The Problem

In this section, we'll address critical issues we encountered in neural networks during previous episodes by using normalization/standardization techniques. Specifically, we aim to solve the challanges in following problems:

1. Polynomial curve fitting (regression) [notes](/basics/03_overfitting_regularization/notes.md)
2. Spiral data point classification (classification) [notes](/basics/02_datapoint_classification/notes.md)

Across both problems, I observed the following challenges:

1. Particularly in the classification problem, the training outcome was highly dependent on initial conditions, such as initialization of weights and biases depended on the random seed. Small changes in these could make the gradients unstable (divergent) or cause them to converge too slowly to a minimum or stuck in local minimum. It was crucial to know the right starting point and the appropriate learning rate to ensure proper convergence.
   
2. I could not increase the learning rate beyond 0.1 without encountering unstable (exploding) gradients. This caused the weights and biases to diverge from the minimum loss points, resulting in extremely slow training — achieving the desired network took up to 50k epochs.

3. I struggled to set the true values on a large scale, particularly in the regression problem, as doing so caused the loss to spike dramatically, making exploding gradients inevitable.



