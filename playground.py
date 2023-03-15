import numpy as np
# Playground for testing out forward propagation

# multiple batches of inputs
inputs = np.array([[1, 2, 3, 2.5],
                   [2, 5, -1, 2],
                   [-1.5, 2.7, 3.3, -0.8]])

# Weights for the first hidden layer
weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])

# biases for the first hidden layer
bias = np.array([2, 3, 0.5])

weights2 = np.array([[0.1, -0.14, 0.5],
                     [-0.5, 0.12, -0.33],
                     [-0.44, 0.73, -0.13]])

bias2 = [-1, 2, -0.5]

# Two different ways to get the same output layer 
output2 = np.matmul(inputs, weights.T) + bias
output3 = np.matmul(weights, inputs.T).T + bias.T
# print(output2)
# print(output3)


layer1_outputs = np.matmul(inputs, weights.T) + bias
layer2_outputs = np.matmul(layer1_outputs, weights2.T) + bias2

print(layer2_outputs)



