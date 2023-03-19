import numpy as np
import mnist
import math
import matplotlib.pyplot as plt
np.random.seed(0)


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

class SquaredLoss(Loss):
    def forward(self, y_pred, y_true):
        ...

class Accuracy:
    def calculate(self, y_pred, y_true):
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y_true)
        return accuracy

def main():
    # loading training data
    images = mnist.test_images()
    labels = mnist.test_labels()

    sample_images = images[:32]
    sample_labels = labels[:32]
    num_inputs = sample_images.shape[1]*sample_images.shape[2]
    input = sample_images.reshape(sample_images.shape[0], num_inputs)

    # initializing network with 1 hidden layer
    # and 1 output layer
    layer1 = Layer_Dense(num_inputs, 16)
    layer2 = Layer_Dense(16, 10)
    act1 = Activation_ReLU()
    act2 = Activation_Softmax()
    loss_function = Loss_CategoricalCrossEntropy()

    # forward pass
    layer1.forward(input)
    act1.forward(layer1.output)

    layer2.forward(act1.output)
    act2.forward(layer2.output)

    output = act2.output
    prediction = np.argmax(output, axis=1)
    loss = loss_function.calculate(output, sample_labels)
    
    print(prediction[:10])
    print(sample_labels[:10])
    print(loss)

if __name__ == '__main__':
    main()

