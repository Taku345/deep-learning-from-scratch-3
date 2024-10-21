if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, Function
from dezero.core import as_variable
import dezero.functions as F

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self,x):
        self.x_shape = x.shape
        y = x.reshepe(self.shape)
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)
    
def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

x = Variable

class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
    
    def backward(self,gy):
        gx = transpose(gy)
        return gx
def transpose(x):
    return Transpose()(x)