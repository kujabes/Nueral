import numpy as np
from nnfs.datasets import spiral_data
import math
# Implementing layers as objects
np.random.seed(0)

# X = np.array([[1, 2, 3, 2.5],
#               [2, 5, -1, 2],
#               [-1.5, 2.7, 3.3, -0.8]])


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # samples from standard normal distribution (mean = 0, variance = 1)
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        # Matrix where each column is a neuron and each row
        # contains elements from a particular batch. Each element 
        # is the input for the 'activation' of the neuron
        self.output = np.matmul(inputs, self.weights) + self.biases

# Rectified Linear Activation Function
# if input is <= 0, then the input is mapped to 0
# else the input is mapped to itself. 
# i.e. x <= 0 -> 0
#      x > 0 -> x
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Sigmoid:
    def forward(self, inputs):
        f = lambda x:  1/(1 + np.exp(-x))
        self.output = f(inputs)

class Activation_Softmax:
    def forward(self, inputs):
        # shifting range to (-inf, 0] so domain is from (0, 1] for each batch
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # for each batch, we normalize by their sum
        # i.e. each row is summed, then normalized by elem/sum(row)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# spiral_data(n, m) Creates a "spiral" dataset, with shape (n*m, 2)
# Can interpret as m sets of n points, each of which spiral from 
# the origin, forming a tail
X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])




