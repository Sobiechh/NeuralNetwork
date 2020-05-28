import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1]
bias = 2

# if np.shape(inputs) == np.shape(weights):
#     print('same shapes')

output = np.dot(weights, inputs) + bias # np.dot(weights, inputs) doesn't matter here because all shapes(4,)
print(output)

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