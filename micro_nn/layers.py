import numpy as np
from micro_nn.tensor import *

'''
The following layers in this file will be based on the Tensor datatype instead of a numpy array. 
'''



class Linear(Tensor):
    def __init__(self, fan_in, fan_out, inputs):
        self.fan_in = fan_in 
        self.fan_out = fan_out 
        self.weights = Tensor([0.001]) * Tensor.sample_random_normal(self.fan_in, self.fan_out)
        self.bias = Tensor([0.001]) * Tensor.sample_random_normal(self.fan_in, self.fan_out)
        self.inputs = Tensor(inputs)
        self.output = 0
        
    def forward(self):
        self.output = Tensor.dot(self.inputs, self.weights) + self.bias

    def backward(self, prev_outputs):
        self.dweights = np.dot(self.inputs.T, prev_outputs)
        self.dbiases = np.sum(prev_outputs, axis=0, keepdims=True)
        self.dinputs = np.dot(prev_outputs, self.weights.T)

    def compute_output_shape(self):
        return self.output.shape
    
    def get_output(self):
        return self.output

    
