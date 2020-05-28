import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1]
bias = 2

# if np.shape(inputs) == np.shape(weights):
#     print('same shapes')

output = np.dot(weights, inputs) + bias # np.dot(weights, inputs) doesn't matter here because all shapes(4,)
#print(output)

'''
layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)
'''


tab1 = [
    [1,2,3],
    [4,5,6]
]

print(np.shape(tab1))

tab2 = [
    [2,1],
    [8,8],
    [5,6]
]

wynik = np.dot(tab2, tab1)

print(wynik)

inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2,3,0.5]

weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
]

biases2 = [-1.0, 2.0, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases #we have to transpose weights0
#biases dont have to be shape(3,3)
#layer1_outputs
# [[ 4.8    1.21   2.385]
#  [ 8.9   -1.81   0.2  ]
#  [ 1.41   1.051  0.026]]

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)