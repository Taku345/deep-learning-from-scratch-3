if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import Model
from dezero.functions import *
from dezero.layers import *

class RNNSelf(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)
        self.h = None
        
    def reset_state(self):
        self.h = None
        
    def forward(self, x):
        if self.h is None:
            h_new = tanh(self.x2h)
        else:
            h_new = tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new
    
# rnn = RNNSelf(10)
# x = np.random.rand(1,1)
# h =  rnn(x)
# print(h.shape)
# なんかエラーでてる