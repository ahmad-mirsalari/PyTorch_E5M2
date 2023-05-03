import torch




# print float8_E5M2 information
print(torch.finfo(torch.float8))

#define two fp8 tensors
x = torch.tensor(-23.5, dtype=torch.float8)
print(x)
'''y = torch.tensor(0.156, dtype=torch.float8)


add = x + y
sub = x - y
div = x / y
mul = x * y


print (f"x + y : {add.item()}, dtype:{add.dtype}")
print (f"x - y : {sub.item()}, dtype:{sub.dtype}")
print (f"x / y : {div.item()}, dtype:{div.dtype}")
print (f"x * y : {mul.item()}, dtype:{mul.dtype}")

#check some basic math functions
t = torch.tensor(4, dtype = torch.float8)
z = torch.pow(2, t)
print (f"pow: {z.item()}, dtype:{z.dtype}")
f = - add
print (f"f=-(x+y): {f.item()}, dtype:{f.dtype}, abs(f):{torch.abs(f)}")

#check matrix 
arr = torch.randn((4,3), dtype= torch.float8)
print(f"arr= {arr}")

arr1 = torch.randn((4,3), dtype= torch.float8)
print(f"arr1= {arr1}")

arr_add = arr + arr1
arr_sub = arr - arr1
print(f"arr + arr1= {arr_add}")
print(f"arr - arr1= {arr_sub}")

arr2 = torch.randn((4,3), dtype= torch.float8)
print(f"arr2= {arr2}")

arr_mul = arr * arr2
print(f"arr * arr2= {arr_mul}")'''




