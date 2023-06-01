'''
Framework information 
This is a very basic implementation of something like keras which is a higher level api for tensorflow (a library for tensor 
manipulation and building of neural networks in python). Similarly, this library is built on top of numpy for the same purpose 
as keras. 
'''

import numpy as np

class Linear:
    def __init__(self, fan_in, fan_out, input):
        self.fan_in = fan_in 
        self.fan_out = fan_out 
        self.weights = 0.001 * np.random.randn(fan_in, fan_out)
        self.bias = 0.001 * np.random.randn(1, fan_out)
        self.output = 0
        self.inputs = input
        
    def forward(self):
        self.output = np.dot(self.inputs, self.weights) + self.bias

    def backward(self, prev_outputs):
        self.dweights = np.dot(self.inputs.T, prev_outputs)
        self.dbiases = np.sum(prev_outputs, axis=0, keepdims=True)
        self.dinputs = np.dot(prev_outputs, self.weights.T)

    def compute_output_shape(self):
        return self.output.shape
    

