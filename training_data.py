import mnist
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

images = mnist.train_images()
labels = mnist.train_labels()

sample = images[:50]
sample_vector = sample.reshape(sample.shape[0], sample.shape[1] * sample.shape[2])

print(labels.shape)
print(sample_vector.shape)
print(labels[0])
