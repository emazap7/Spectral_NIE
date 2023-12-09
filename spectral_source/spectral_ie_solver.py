import logging
import warnings
from typing import Callable, Optional, Union

import numpy as np
import torch

from scipy import integrate

logger = logging.getLogger("idesolver")
logger.setLevel(logging.WARNING)

import matplotlib.pyplot as plt


import torchcubicspline

from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
                             
from torchdiffeq import odeint

    
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

from functools import reduce
from spectral_source import kernels, integrators 
from spectral_source.utils import to_np
import random

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F
from torch.autograd import Variable
from joblib import Parallel, delayed

use_cuda = torch.cuda.is_available()


if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"
    
    
    

def global_error(y1: torch.Tensor, y2: torch.Tensor) -> float:

    diff = y1 - y2
    return torch.sqrt(torch.dot(diff.flatten(), diff.flatten()))




class Spectral_IE_solver:
    
    def __init__(
        self,
        x: torch.Tensor,
        y_0: Union[float, np.float64, complex, np.complex128, np.ndarray, list,torch.Tensor,torch.tensor],
        c: Optional[Callable] = None,
        d: Optional[Callable] = None,
        model:Optional[Callable] = None,
        domain_dim: int = 1,
        channels: int = 1,
        chebyshev: bool = False,
        chebyshev_coeff: Optional[Callable] = None,
        max_deg = 20,
        spectral_integrator: Optional[Callable] = None,
        global_error_tolerance: float = 1e-6,
        max_iterations: Optional[int] = None,
        int_atol: float = 1e-5,
        int_rtol: float = 1e-5,
        smoothing_factor: float = 0.5,
        store_intermediate_y: bool = False,
        global_error_function: Callable = global_error,
        output_support_tensors = False,
        return_function = False,
        accumulate_grads = False,
    ):
        
        
        self.y_0 = y_0.to(device)
        
        self.x = x.to(device)
        
        self.accumulate_grads=accumulate_grads
        
        self.dim = self.y_0.shape[-1]
        
        self.n_batch = self.y_0.shape[0]
        
        self.domain_dim = domain_dim
        
        self.channels = channels
            
        if chebyshev_coeff is None and chebyshev is False:
            #Perform cosine transform (depending on dimension)
            if domain_dim == 1:
                self.cosine_transform = dct.dct
                self.inv_cosine_transform = dct.idct
            elif domain_dim == 2:
                self.cosine_transform = dct.dct_2d
                self.inv_cosine_transform = dct.idct_2d
            elif domain_dim == 3:
                self.cosine_transform = dct.dct_3d
                self.inv_cosine_transform = dct.idct_3d
            else:
                pass #Need to add higher dimensional versions (not implemented in dct)
        
            self.spec_y_0 = self.cosine_transform(self.y_0)
        
        elif chebyshev_coeff is None and chebyshev is True:
            #Chebyshev transformation class
            self.chebyshev = chebyshev_coeff
            self.spec_y_0 = self.chebyshev.expand(self.y_0)
        else:
            self.chebyshev = chebyshev_coeff
            self.spec_y_0 = self.y_0
            
        self.spectral_integrator = spectral_integrator
        
        self.max_deg = max_deg
        
        self.support_tensors=x.to(device)
            
        self.return_function=return_function

        if c is None:
            c = lambda x: self.y_0.repeat(1,self.support_tensors.shape[0],1).to(device)
        if d is None:
            d = lambda x: torch.Tensor([1]).to(device)
            
            
        self.c = lambda x: c(x)
        self.d = lambda x: d(x)
        
              
        self.model = model.to(device)
        
        
        if global_error_tolerance == 0 and max_iterations is None:
            raise exceptions.InvalidParameter(
                "global_error_tolerance cannot be 0 if max_iterations is None"
            )
        if global_error_tolerance < 0:
            raise exceptions.InvalidParameter("global_error_tolerance cannot be negative")
        self.global_error_tolerance = global_error_tolerance
        self.global_error_function = global_error_function

        if not 0 < smoothing_factor < 1:
            raise exceptions.InvalidParameter("Smoothing factor must be between 0 and 1")
        self.smoothing_factor = smoothing_factor

        
        
        if max_iterations is not None and max_iterations <= 0:
            raise exceptions.InvalidParameter("If given, max iterations must be greater than 0")
        
        
        self.max_iterations = max_iterations


        self.int_atol = int_atol
        self.int_rtol = int_rtol

        self.store_intermediate = store_intermediate_y
        if self.store_intermediate:
            self.y_intermediate = []

        self.iteration = None
        self.y = None
        self.global_error = None
        
    
    
    def solve(self, callback: Optional[Callable] = None) -> torch.Tensor:
            

        y_current = self._initial_y()
        y_guess = self._solve_rhs_with_known_y(y_current) 

        error_current = self._global_error(y_current, y_guess)

        self.iteration = 0

        logger.debug(
            f"Advanced to iteration {self.iteration}. Current error: {error_current}."
        )

        if callback is not None:
            logger.debug(f"Calling {callback} after iteration {self.iteration}")
            callback(self, y_guess, error_current)

        while error_current > self.global_error_tolerance:

            new_current = self._next_y(y_current, y_guess)
            new_guess = self._solve_rhs_with_known_y(new_current)
            new_error = self._global_error(new_current, new_guess)
            if new_error > error_current:
                warnings.warn(
                    f"Error increased on iteration {self.iteration}",
                )

            y_current, y_guess, error_current = (
                new_current,
                new_guess,
                new_error,
                )

            if self.store_intermediate:
                self.y_intermediate.append(y_current)

            self.iteration += 1


            logger.debug(
            f"Advanced to iteration {self.iteration}. Current error: {error_current}."
            )

            if callback is not None:
                logger.debug(f"Calling {callback} after iteration {self.iteration}")
                callback(self, y_guess, error_current)

            if self.max_iterations is not None and self.iteration >= self.max_iterations:
            
                break


        
        self.y = y_guess
        self.global_error = error_current
        
            
        if self.return_function is True and self.chebyshev is True:
            return self.chebyshev.inverse(self.y)
        
        else:
            
            return self.y
    
    
    def _initial_y(self) -> torch.Tensor:
        
        y_initial = self.c(self.chebyshev) 
            
        return y_initial

    
    
    
    def _next_y(self, curr: torch.Tensor, guess: torch.Tensor) -> torch.Tensor:
        """Calculate the next guess at the solution by merging two guesses."""
        return (self.smoothing_factor * curr) + ((1 - self.smoothing_factor) * guess)

    
    def _global_error(self, y1: torch.Tensor, y2: torch.Tensor) -> float:
        
        
        return self.global_error_function(y1, y2)

    
    
    
    def _solve_rhs_with_known_y(self, y: torch.Tensor) -> torch.Tensor:
        
        def integral(x):
            
            time = self.support_tensors.view(1,self.x.shape[0],1,1)\
                   .repeat(self.n_batch,1,1,self.channels)
            y_in = y
            
            y_in = y_in.view(self.n_batch,1,y.shape[-2],y.shape[-1])\
                    .repeat(1,self.x.shape[0],1,1)
            
            y_in = torch.cat([y_in,time],-2)
            
            model = self.model
            
            if self.accumulate_grads is False:
                y_in = y_in.detach().requires_grad_(True)
            else:
                y_in = y_in.requires_grad_(True)
               
            
            T = model.forward(y_in)
            
            z = self.spectral_integrator.integrate(T)
            
            del y_in
            del time
            
            return z
        

        def rhs(x):
            
            return self.c(self.chebyshev) + (self.d(x)*integral(x))
            

        return self.solve_IE(rhs)

    
    def solve_IE(self, rhs: Callable) -> torch.Tensor:
        
        func = rhs 
        
        times = self.support_tensors
        
        sol = rhs(times)
        
        return sol
