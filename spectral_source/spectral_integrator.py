import numpy as np
import torch
import torch_dct as dct

from spectral_source.integrators import MonteCarlo 
mc = MonteCarlo()


class spectral_integrator:
    def __init__(self,shapes=[8,16,16],device="cuda:0"):
        
        self.shapes = shapes
        self.signs = torch.Tensor([(-1)**(i+1) for i in range(shapes[-1]-1)]).to(device)
        self.norm = torch.Tensor([2*i for i in range(shapes[-1])]).to(device)
        self.device = device
    
    def integrate(self,a):
        dd = torch.div(a.roll(1,dims=-1)[...,1:-1] - a.roll(-1,dims=-1)[...,1:-1],self.norm[1:-1])
        dd = torch.cat([dd,a.roll(1,dims=-1)[...,-1:]/self.norm[-1]],-1)
        d0 = (self.signs*dd).sum(-1)
        dd = torch.cat([d0.unsqueeze(-1),dd],-1)
        
        d = dd[...,1::2]
        half_integral = d.sum(-1)
        
        return 2*half_integral
    
    def sum_prime(self,x):
        divisor = torch.ones_like(x).to(self.device)
        
        divisor[...,0] = 2
        
        return torch.div(x,divisor).sum(-1)
      
class Chebyshev:
    def __init__(self,max_deg=10,N_mc=10000,device="cuda:0"):
        
        self.max_deg = max_deg
        self.coeff = torch.linspace(0,max_deg-1,max_deg).to(device)
        self.cheb_poly = lambda a, x: (a.to(device)*\
                                       (torch.cos(self.coeff[1:]*torch.arccos(x.to(device)))\
                                       .repeat(a.shape[0],a.shape[1],1))).sum()
        self.N_mc = N_mc
        self.device = device
        
    def expand(self,f):
        
        func = lambda x: f(torch.cos(x.to(self.device)))\
                         *torch.cos(self.coeff*x.to(self.device))
        
        a = mc.integrate(func,
                         dim=1,
                         out_dim=0,
                         N=self.N_mc,
                         integration_domain=[[0,torch.pi]])
        return a*2/torch.pi
    
    def inverse(self,a):
        
        return lambda x: a[...,0]/2+self.cheb_poly(a[...,1:],x).sum()
        

class spectral_integrator_nchannels:
    def __init__(self,shapes=[8,16,16,2],device="cuda:0"):
        
        self.shapes = shapes
        self.signs = torch.Tensor([(-1)**(i+1) for i in range(shapes[-2]-1)])\
                     .view(1,1,shapes[-2]-1,1).to(device)
        self.norm = torch.Tensor([2*i for i in range(shapes[-2])]).to(device)
        self.device = device
    
    def integrate(self,a):
        
        dd = torch.div(a.roll(1,dims=-2)[...,1:-1,:] - a.roll(-1,dims=-2)[...,1:-1,:],\
                       self.norm[1:-1].view(1,1,a.shape[-2]-2,1))
        dd = torch.cat([dd,a.roll(1,dims=-2)[...,-1:,:]/self.norm[-1]],-2)
        d0 = (self.signs*dd).sum(-2)
        dd = torch.cat([d0.unsqueeze(-2),dd],-2)
        
        d = dd[...,1::2,:]
        half_integral = d.sum(-2)
        
        return 2*half_integral
    
    def sum_prime(self,x):
        divisor = torch.ones_like(x).to(self.device)
        
        divisor[...,0,:] = 2
        
        return torch.div(x,divisor).sum(-2)


class Chebyshev_nchannel:
    def __init__(self,max_deg=10,N_mc=10000,channels=2,n_batch=8,device="cuda:0"):
        
        self.max_deg = max_deg
        self.coeff = torch.linspace(0,max_deg-1,max_deg).to(device)
        self.cheb_poly = lambda a, x: (a.to(device).view(n_batch,max_deg-1,1,channels)*\
                                       (torch.cos(\
                                       (self.coeff[1:].view(self.max_deg-1,1).repeat(1,x.shape[0])\
                                        *torch.arccos(x.to(device))).view(1,self.max_deg-1,x.shape[0],1)\
                                       .repeat(n_batch,1,1,self.channels))
                                       )).sum(-3)
        
        self.N_mc = N_mc
        self.channels = channels
        self.n_batch = n_batch
        self.device = device
        
    def expand(self,f):
        
        func = lambda x: f(torch.cos(x.to(self.device)))\
                         .view(self.n_batch,self.N_mc,1,self.channels)\
                         .repeat(1,1,self.max_deg,1)\
                         *torch.cos((self.coeff*x.to(self.device))\
                         .view(1,x.shape[0],self.max_deg,1)\
                         .repeat(self.n_batch,1,1,self.channels))
        
        a = mc.integrate(func,
                         dim=1,
                         out_dim=-3,
                         N=self.N_mc,
                         integration_domain=[[0,torch.pi]])
        return a*2/torch.pi
    
    def inverse(self,a):
        
        return lambda x: a[...,:1,:]/2+self.cheb_poly(a[...,1:,:],x)
    
    
