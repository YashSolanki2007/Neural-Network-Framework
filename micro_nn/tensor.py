'''
Framework information 
This is a very basic implementation of something like keras which is a higher level api for tensorflow (a library for tensor 
manipulation and building of neural networks in python). Similarly, this library is built on top of numpy for the same purpose 
as keras. 
'''

'''
TODO: Add support for multiplication with a scalar directly without converting to tensor
'''

import numpy as np


# Base tensor class on top of which all ops will be built
class Tensor:
    def __init__(self, data):
        self.data = data
        self.grad = 0
        self.length = len(self.data) if type(self.data) == list else 1
    
    # Binary Ops
    def __add__(self, other):
        return Tensor(self.data + other.data)
    
    def __mul__(self, other):
        out = []
        if other.length > 1:
            for i in range(other.length):
                out.append(self.data[i] * other.data[i])
        else:
            for i in range(self.length):
                out.append(self.data[i] * other.data[i])

        return Tensor(out)
    
    def __pow__(self, other):
        out = []
        for i in range(len(self.data)):
            out.append(self.data[i] ** other)
        return Tensor(out)
    
    def dot(self, other):
        product = Tensor(sum(i[0] * i[1] for i in zip(self.data, other.data)))
        return product 
    
    @staticmethod
    def sample_random_normal(fan_in, fan_out):
        return Tensor(np.random.randn(fan_in, fan_out))
    
    def __repr__(self):
        return f"Tensor with value: {self.data} and gradient: {self.grad}"
