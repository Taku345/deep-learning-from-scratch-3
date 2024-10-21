if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import optimizers
from dezero.models import MLP
from dezero.functions import *

class Optimizer:
    def __init__(self) -> None:
        self.target = None
        self.hooks = []
        
    def setup(self, target):
        self.target = target
        return self

    def updata(self):
        params = [p for p in self.target.params() if p.grad is not None]
        
        for f in self.hooks:
            f(params)
            
        for param in params:
            self.updata_one(param)
        
    def update_one(self, parma):
            raise NotImplementedError()
        
    def add_hook(self, f):
        self.hooks.append(f)
        
class SGD(Optimizer):
    def __init__(self, lr=0.01) -> None:
        super().__init__()
        self.lr = lr
        
    def updata_one(self, param):
        param.data -=self.lr * param.grad.data
        

np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)

lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr)
optimizer.setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)
    
    model.cleargrads()
    loss.backward()
    
    optimizer.update()
    if i % 1000 == 0:
        print(loss)
        
class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9) -> None:
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}
        
    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)
        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v
        
def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    
    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y