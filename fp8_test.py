import torch


# it is a test
print(torch.finfo(torch.float8))
x = torch.tensor(512, dtype=torch.bfloat16)
#y = 12
print("2")
y = torch.tensor(128, dtype=torch.float8)
add = x + y
sub = x - y
div = x / y
mult = x * y


print (f"x+y: {add.item()}, dtype:{add.dtype}")
print (f"x-y: {sub.item()}, dtype:{sub.dtype}")
print (f"x/y: {div.item()}, dtype:{div.dtype}")
print (f"x*y: {mult.item()}, dtype:{mult.dtype}")
t = torch.tensor(4, dtype = torch.float8)
z = torch.pow(2, t)
print (f"pow: {z.item()}, dtype:{z.dtype}")