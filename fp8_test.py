import torch

print(torch.finfo(torch.float8))
x = torch.tensor(-10.0, dtype=torch.float8)
#y = 12
y = torch.tensor(1.23, dtype=torch.float)
y = x + y

print(x.item())
print(x.dtype)

print(f"y is: {y.item()}")
print(f"y data type: {y.dtype}")