from main import * 
import numpy as np 
from matplotlib import pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
import numpy as np


# Initializing
nnfs.init()

# Creating hyperparameter objects
sigmoid = Sigmoid()
softmax = Softmax()


# Generating dataset
X, y = spiral_data(samples=100, classes=2)


''' print(X)
print(X.shape) '''

dense = Dense(2, 3)
print(dense)
output = dense.forward(X)

output_s = sigmoid.forward(output)


dense_2 = Dense(3, 1)
output_2 = dense_2.forward(output_s)
output_2 = softmax.forward(output_2)

print(output_2.shape)


#loss = Mean_Squared_Error()
loss = Binary_Crossentropy()


#print(loss.forward(X, output_2.T, y))
print(loss.forward(y, y, output_2))

print(y)

plt.plot(output_2, 'g')
plt.plot(X, '.b')
plt.show()

plt.plot(output, 'r')
plt.plot(X, '.b')
plt.show()

