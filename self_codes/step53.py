import sys, os
import numpy as np
import dezero.datasets
import dezero.functions as F
import dezero
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP

max_epoch = 3
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000,10))
optimizer = optimizers.SGD().setup(model)

# if os.path.exists('my_mlp.npz'):
#     model.load_weights('my_mlp.npz')

# for epoch in range(max_epoch):
#     sum_loss = 0
    
#     for x, t in train_loader:
    
    
def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)
    
    if dezero.Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x


# class Layer:
#     # 省略
    
#     def _flatten_params(self, params_dict, parent_key=""):
#         for name in self._params:
#             obj = self.__dict__[name]
#             key = parent_key + '/' + name if parent_key else name
            
#             if isinstance(obj, Layer):
#                 obj._flatten_params(params_dict, key)
#             else:
#                 params_dict[key] = obj
    
#     def save_weights(self, path):
#         self.to_cpu()
        
#         params_dict = {}
#         self._flatten_params(params_dict)
#         array_dict = {key: param.data for key, param in params_dict.items() if param is not None}
        
#         try:
#             np.savez_compressed(path, **array_dict)
#         except (Exception, KeyboardInterrupt) as e:
#             if os.path.exists(path):
#                 os.remove(path)
#             raise
        
#     def load_weights(self, path):
#         npz = np.load(path)
#         params_dict = {}
#         self._flatten_params(params_dict)
#         for key, param in params_dict.items():
#             param.data = npz[key]
            
