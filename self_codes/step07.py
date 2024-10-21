import numpy as np
from step06 import *


# class Variable:
#     def __init__(self,data) -> None:
#         self.data = data
#         self.grad = None
#         self.creator = None
    
#     def set_creator(self,func):
#         self.creator = func
    
#     def backward(self):
#         funcs = [self.creator]
#         while funcs:
#             f = funcs.pop()
#             x, y = f.input, f.output
#             x.grad = f.backward(y.grad)
            
#             if x.creator is not None:
#                 funcs.append(x.creator)
  

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)