from micro_nn.layers import * 
from micro_nn.tensor import Tensor

X = [[1, 3, 4], [4, 4 , 3]]

# Testing of the tensor class 
t = Tensor([1, 2, 3])
q = Tensor([4, 5, 6])


print(t + q)
print(t.__pow__(2))
print(t * q)

print(Tensor([2]) * Tensor([4]))

print(Tensor.sample_random_normal(3, 3))


# Currently the input params are fan_out, fan_in
l1 = Linear(5, 3, X)
l1.forward()
print(l1.output)