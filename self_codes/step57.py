if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import *
import dezero.functions as F
from dezero.utils import get_conv_outsize
from dezero.functions_conv import im2col
from dezero.functions import linear
from dezero.layers import cuda

x1 = np.random.rand(1, 3, 7, 7)
col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)

def pair(x):
    if isinstance(x, int):
        return (x,x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError

def conv2d_simple(x, W, b=None, stride=1, pad=0):
    x, W = as_variable(x), as_variable(W)
    
    Weight = W
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    
    col = im2col(x, (KH, KW), stride, pad, to_matrix=True)
    Weight = Weight.reshape(OC, -1).transpose()
    t = linear(col, Weight, b)
    y = t.reshape(N, OH, OW, OC).transpose(0,3,1,2)
    return y

class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype
        
        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')
        
    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data
        
    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)
    
        y = F.conv2d_simple(x, self.W, self.b, self.stride, self.pad)
        return y
    
    def pooling_simple(x, kernel_size, stride=1, pad=0):
        x = as_variable(x)
        
        N, C, H, W = x.shape
        KH, KW = pair(kernel_size)
        PH, PW = pair(pad)
        SH, SW = pair(stride)
        OH = get_conv_outsize(H, KH, SH, PH)
        OW = get_conv_outsize(W, KW, SW, PW)
        
        col = im2col(x, kernel_size, stride, pad, to_matrix=True)
        col = col.reshape(-1, KH * KW)
        y = col.max(axis=1)
        y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
        return y