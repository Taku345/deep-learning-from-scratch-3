import numpy as np

class Variable:
    def __init__(self, data) -> None:
        self.data = data
        
    
# step2
class Function:
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self,x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2
    
# x = Variable(np.array(10))
# f = Square()
# y = f(x)
# print(type(y))
# print(y.data)

#step3
class Exp(Function):
    def forward(self, x):
        return np.exp(x)


#step4
def numerical_diff(f,x,eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

# f= Square()
# x = Variable(np.array(3.0))
# dy = numerical_diff(f,x)
# print(dy)

# def f(x):
#     A = Square()
#     B = Exp()
#     C = Square()
#     return C(B(A(x)))

# x = Variable(np.array(0.5))
# dy = numerical_diff(f,x)

# print(dy)
