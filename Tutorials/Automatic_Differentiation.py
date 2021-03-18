import torch

x = torch.ones(5)   # input tensor
y = torch.zeros(3)  # expected output
# In order to compute the gradients of the loss function we add set the requires grad function on a tensor
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print('Gradient function for z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

# Compute the partial derivatives of the loss wrt weight and bias matrices
loss.backward()
print(w.grad)
print(b.grad)

z = torch.matmul(x, w)+b
print(z.requires_grad)

# Will disable to gradient function for forward calculation on neural network without call to backpropagation
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# Can also use detach method on the tensor
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)