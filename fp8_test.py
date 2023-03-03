import torch


# it is a test
print(torch.finfo(torch.float8))
x = torch.tensor(128, dtype=torch.float8)
#y = 12
print("2")
y = torch.tensor(128, dtype=torch.float8)
x = x +y
print(x.item())
print(x.dtype)
