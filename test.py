from main import *
import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy

y_true = np.array([1., 1., 1.])
y_pred = np.array([1., 1., 0.7])

bce = BinaryCrossentropy()
loss = bce(y_true, y_pred)
np_loss = loss.numpy()
print(np_loss)


loss = Binary_Crossentropy()
print(loss.forward(y_true, y_true, y_pred))


print(loss.backward(y_true, y_pred))
