import torch

x = torch.tensor(2, dtype=torch.float8)
t = torch.tensor(1, dtype=torch.float16)
t = x - t

print(x)
print(x.dtype)

print(t)
print(t.dtype)