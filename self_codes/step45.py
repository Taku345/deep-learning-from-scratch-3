if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import *
import dezero.layers as L
import dezero.functions as F

# class Layer:
#     def __init__(self) -> None:
#         self._params = set()
        
#     def __setattr(self, name, value):
#         if isinstance(value, (Parameter, Layer)):
#             self._params.add(name)
#         super().__setattr__(name,value)
        
#     def params(self):
#         for name in self._params:
#             obj = self.__dict__[name]
            
#             if isinstance(obj, Layer):
#                 yield from obj.params()
#             else:
#                 yield obj

model = Layer()
model.l1 = L.Linear(5)
model.l2 = L.Linear(3)

def predict(model, x):
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y

for p in model.params():
    print(p)
    
model.cleargrads()

class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size) -> None:
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
        
    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y
    
x = Variable(np.random.randn(5,10), name='x')
model = TwoLayerNet(100, 10)
model.plot(x)