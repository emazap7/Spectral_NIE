from torch import nn
import torch
import numpy as np

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)




class F_NN(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(F_NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y_in = y
        y = y_in.flatten(-2,-1)
        y = self.NL(self.first.forward(y))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y = self.last.forward(y)
        
        return y.view(y_in.shape[0],y_in.shape[1],y_in.shape[2]-1,y_in.shape[3])
