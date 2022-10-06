# Imports
import numpy as np 
from numba import jit


# Setting random seed
np.random.seed(1)


# Creating the preprocess data function 
def preprocess_data(img_list):
    '''
    Takes a list param and loops over and divides each element by 255.0 
    '''
    for i in range(len(img_list)):
        img_list[i] /= 255.0

    return img_list


# Dense layer class 
class Dense:
    def __init__(self, inputs_shape, num_neurons):
        self.weights = 0.01 * np.random.randn(inputs_shape, num_neurons)
        self.biases = np.zeros((1, num_neurons))

    def forward(self, inputs):
        output = np.dot(inputs, self.weights) + self.biases
        return output

    def backward(self, inputs, prev_outputs):
        self.dweights = np.dot(inputs.T, prev_outputs) 
        self.dbiases = np.sum(prev_outputs, axis=0, keepdims=True)

        self.dinputs = np.dot(prev_outputs, self.weights.T)



# Flatten layer
class Flatten:
    def __init__(self, inputs):
        self.inputs = inputs
        self.inputs_copy = inputs

    def forward(self):
        # Collapse inputs into a single dimension
        flattened_outputs = self.inputs.flatten() 
        return flattened_outputs 

    def backward(self):
        return self.inputs_copy




# ReLU activation function 
class ReLU:
    def forward(self, x):
        return np.maximum(x, 0)


    def backward(self, x):
        return x > 0


# Sine activation function
class Sine:
    def forward(self, x):
        return np.sin(x)

    def backward(self, x):
        return np.cos(x)


# Tanh activation function
class Tanh:
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1 - pow(np.tanh(x), 2)


# Sigmoid activation function
class Sigmoid:
    def forward(self, x):
        return (1 / 1 + np.exp(-x))

    def backward(self, x):
        return (1 / 1 + np.exp(-x)) * (1 - (1 / 1 + np.exp(-x)))


# Softmax activation function
class Softmax:
    def forward(self, x):
        return np.exp(x) / np.sum(np.exp(x))


    # Fast execution using jit
    @jit(fastmath=True, forceobj=True)
    def backward(self, s):
        a = np.eye(s.shape[-1])
        temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
        temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                for k in range(s.shape[1]):
                    temp1[i,j,k] = s[i,j]*a[j,k]
                    temp2[i,j,k] = s[i,j]*s[i,k]
     
        return temp1-temp2



# Crossentropy loss 
class Crossentropy:
    def forward(self, y_true, y_pred):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped*y_true,
                axis=1
            )


        negative_log_likelihoods = -np.log(correct_confidences)
        final_loss = np.mean(negative_log_likelihoods)

        return final_loss 

    def backward(self, y_true, prev_outputs):
        # Number of samples
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples




# Binary Crossentropy Loss Function 
class Binary_Crossentropy:

    # Using just in time compilation to increase processing speed
    @jit(fastmath=True, forceobj=True)
    def forward(self, data, actual_values, predicted_probabilities):
        # Redifining the function params into the variables used in the actual equation
        N = len(data)
        p = predicted_probabilities
        y = actual_values

        p = np.clip(p, 1e-7, 1 - 1e-7)
        prob = np.log(predicted_probabilities)
        loss_val = abs(-np.mean((y * (np.log(p + 1e-7) + (1 - y) * (np.log(1 - p + 1e-7))))))


        return loss_val


    @jit(fastmath=True, forceobj=True)
    def backward(self, actual_values, predicted_probabilities):
        # Redifining the function params into the variables used in the actual equation
        p = predicted_probabilities
        y = actual_values

        p = np.clip(p, 1e-7, 1 - 1e-7)

        term_1 = (y / p)
        term_2 = (1 - y) / (1 - p)

        output = -np.mean(term_1 - term_2)
        return output
