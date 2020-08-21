# import numpy as np

# np.random.seed(0)

# layer_outputs = [4.8, 1.21, 2.385]  

# exp_values = np.exp(layer_outputs)
# print(f'exponentiated values: {exp_values}')

# norm_values = exp_values / np.sum(exp_values)
# print(f'normalized exponentiated values: {norm_values}')
# print(f'sum of normalized values: {np.sum(norm_values)}')


import numpy as np 

layer_outputs = np.array([[4.8 ,   1.21 ,   2.385],
                          [8.9  , -1.81  ,  0.2  ],
                          [1.41 ,  1.051  , 0.026]])


print(f"sum without axis: {np.sum(layer_outputs)}")

print(f"This will be identical to the above since default is None: \n {np.sum(layer_outputs, axis=None)}")

print(f"another way to think of it w/ a matrix == axis 0: columns: \n{np.sum(layer_outputs, axis= 0)}")

print(f"sum axis 1 (rows): \n{np.sum(layer_outputs, axis= 1)}")

print(f"sum axis 1 (rows) but keeping the same dimension as input: \n{np.sum(layer_outputs, axis= 1, keepdims = True)}")