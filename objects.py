import numpy as np
# Implementing layers as objects

np.random.seed(0)

X = np.array([[1, 2, 3, 2.5],
              [2, 5, -1, 2],
              [-1.5, 2.7, 3.3, -0.8]])

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # samples from standard normal distribution (mean = 0, variance = 1)
        self.weights = np.random.randn(n_inputs, n_neurons)
        pass
    def forward(self):
        pass

print(np)
