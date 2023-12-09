import math
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm_notebook as tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

from functools import reduce
from spectral_source import kernels, integrators 
from spectral_source.utils import to_np
import random
import pickle
import time



class F_NN(nn.Module):
    def __init__(self,in_dim,in_dim_t,out_dim,shapes,NL=nn.ELU):
        super(F_NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.first_t = nn.Linear(in_dim_t,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.layers_t = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.last_t = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y_in = y
        
        y = y_in.flatten(-2,-1)
        
        y = self.NL(self.first.forward(y))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y = self.last.forward(y)
        
        y = y.view(y_in.shape[0],y_in.shape[1],y_in.shape[2]-1,y_in.shape[3])
        y = y.permute(0,2,1,3)
        
        y_in = y
        y = y_in.flatten(-2,-1)
        y = self.NL(self.first_t.forward(y))
        for layer_t in self.layers_t:
            y = self.NL(layer_t.forward(y))   
        y = self.last_t.forward(y)
        
        y = y.view(y_in.shape[0],y_in.shape[1],y_in.shape[2],y_in.shape[3])
        y_out = y.permute(0,2,1,3)
        
        return y_out


class F_func(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(F_func, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        
        y = self.NL(self.first.forward(y))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y = self.last.forward(y)
        
        return y
