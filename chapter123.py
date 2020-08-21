import numpy as np
import matplotlib

np.random.seed(0)

X = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

X2 = [
    [1.1, -1.0, 3.0],
    [1.2, 8.9, -1.0],
    [-7.5, 2.3, 1.3],
    [1.5, -3.5, 2.4]
]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) #tuple
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

'''
np.random.randn(3, 4)
[[ 1.76405235  0.40015721  0.97873798  2.2408932 ]
 [ 1.86755799 -0.97727788  0.95008842 -0.15135721]
 [-0.10321885  0.4105985   0.14404357  1.45427351]]
'''

''' ex1
layer1 = Layer_Dense(4, 5) #(how many features, numb of neurons)
layer2 = Layer_Dense(5, 2) # 5==5

layer1.forward(X)
#print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)
'''


'''
X2 = [
    [1.1, -1.0, 3.0],
    [1.2, 8.9, -1.0],
    [-7.5, 2.3, 1.3],
    [1.5, -3.5, 2.4]
]
'''
layer1 = Layer_Dense(3, 10) #3 couse x2 colmumns(features)
layer2 = Layer_Dense(10, 2) 

layer1.forward(X2)

layer2.forward(layer1.output)
print(layer2.output)