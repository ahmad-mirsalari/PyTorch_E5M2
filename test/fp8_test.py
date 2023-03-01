import torch

x = torch.tensor(-1345.0000485, dtype=torch.float32)
x = x.type(torch.float8)
t = torch.tensor(1, dtype=torch.float16)
t = x - t

print(x)
print(x.dtype)

print(t)
print(t.dtype)