import torch
import numpy as np

# INITIALIZING A TENSOR
# From data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# From a Numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From another tensor
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


# Random or constant variables, with a given size of shape
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")


# Attributes of a Tensor
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# Operations on Tensors
# The .to method moves a tensor from the CPU to the GPU for hardware acceleration
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# Standard numpy like indexing and slicing
tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column: ', tensor[..., -1])
tensor[:, 1] = 0
print(tensor)

# .cat joins tensors together along a given dimension
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Arithmetic operations
# Compute matric multiplication beteween two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# Compute element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Single element tensors
# Will aggregate the entire tensor into a single sum
agg= tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In place operations that store the result in the operand are denoted with an _
# Their use is discouraged due to a loss of history when computing derivatives
print (tensor, "\n")
tensor.add_(5)
print(tensor)


# Bride with Numpy
# Tensors on CPU and NumPy arrays can share their underlying memory locations
# such that changing one will change the other
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

print("\nChange in Tensor to numpy array")
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor
print("\nNumpy to Tensor")
n = np.ones(5)
t = torch.from_numpy(n)
print(t)

print("\nChange in Numpy array to Tensor")
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")








