import numpy as np
import math
from nnfs.datasets import spiral_data

np.random.seed(0)

# X = np.array([[1, 2, 3, 2.5],
#               [2, 5, -1, 2],
#               [-1.5, 2.7, 3.3, -0.8]])


# Implementing layers as objects
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
        self.output = 1/(1 + np.exp(-inputs))

class Activation_Softmax:
    def forward(self, inputs):
        # shifting range to (-inf, 0] so domain is from (0, 1] for each batch
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # for each batch, we normalize by their sum
        # i.e. each row is summed, then normalized by elem/sum(row)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
        
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = range(len(y_pred))
        # If an input is 0, then loss will evaluate to inf.
        # clipping the inputs helps prevent that issue
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # list of correct labels/index passes in
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[samples, y_true]
        # one hot encoded vectors being passes as a matrix
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood




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

loss_function = Loss_CategoricalCrossEntropy()

loss = loss_function.calculate(activation2.output, y)


print(activation2.output[:5])
print('Loss: ', loss)



