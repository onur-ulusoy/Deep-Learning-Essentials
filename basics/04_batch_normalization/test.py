
import os, pickle, torch
import numpy as np
import matplotlib.pyplot as plt
from basics.dataset_gen_scripts.polynomial_data import PolynomialDataGenerator
from nn import NN
import hyperparams as hp

# Test class prepared with NumPy based NN
class ModelTester:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.network = NN(
            self.input_size, 
            self.hidden1_size, 
            self.hidden2_size, 
            self.output_size,
        )

    