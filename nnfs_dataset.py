#from nnfs.dataset import spiral_data

import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros(shape=(1, neurons))


    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

nnfs.init()

print(np.random.randn(2,5))
print(np.zeros((2,5)))

inputs = 2
neurons = 4

weights = 0.01 * np.random.randn(inputs, neurons)
biases = np.zeros((1, neurons))

print(weights)
print(biases)

X, y = spiral_data(100, 3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs (each sample has 2 features), 3 outputs

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Let's see output of the first few samples:
print(dense1.output[:5])
