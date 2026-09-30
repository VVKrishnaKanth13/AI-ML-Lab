import torch
from sympy import false

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

# Forward propagation
z = x * y + y ** 2

print("z:", z)

# Retain gradients for the intermediate tensor z
z.retain_grad()

# Backpropagation
loss = z.sum()
loss.backward()

# Gradients of leaf tensors
print("x.grad:", x.grad)
print("y.grad:", y.grad)

# Gradient of the intermediate tensor
print("z.grad:", z.grad)

# Detach z from the computation graph
z_det = z.detach()
print("z_det.requires_grad:", z_det.requires_grad)
print("z_det.grad:", z_det.grad)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=False)

k = x * y + y ** 2
loss = k.sum()
loss.backward()
