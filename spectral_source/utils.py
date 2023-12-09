#General libraries
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import scipy
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

import torchcubicspline

from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

from torch.utils.data import SubsetRandomSampler

from spectral_source.spectral_integrator import Chebyshev_nchannel, spectral_integrator_nchannels, spectral_integrator, Chebyshev

#Torch libraries
import torch
from torch.nn import functional as F

if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"
        
def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
class Select_times_function():
    def __init__(self,times,max_index):
        self.max_index = max_index
        self.times = times
    
    
def to_np(x):
    return x.detach().cpu().numpy()

class EarlyStopping():

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                

class SaveBestModel:

    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

    
    def __call__(self, path, current_valid_loss, epoch, model, model_func, f_func):
        if current_valid_loss < self.best_valid_loss:
            
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch}\n")
            
            model_state = {'state_dict': model_func.state_dict()}
            torch.save(model, os.path.join(path,'model.pt'))
            
            if f_func is not None: 
                f_func_state = {'state_dict': f_func.state_dict()}
                torch.save(f_func_state, os.path.join(path,'f_func.pt'))
            
            
            
def load_checkpoint(path, model, optimizer, scheduler, f_func):
    print('Loading ', os.path.join(path))
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
     
    checkpoint = torch.load(os.path.join(path, 'model.pt'), map_location=map_location)
    start_epoch = checkpoint['epoch']
    offset = start_epoch
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    checkpoint = torch.load(os.path.join(path, 'model.pt'), map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])

    if f_func is not None: 
        checkpoint = torch.load(os.path.join(path, 'f_func.pt'), map_location=map_location)
        f_func.load_state_dict(checkpoint['state_dict'])
    
    return model, optimizer, scheduler, f_func
    
class Test_Dynamics_Dataset(torch.utils.data.Dataset):
    def __init__(self, Data, times):

        self.times = times.float()
        self.Data = Data.float()

    def __len__(self):
        return len(self.times)

    def __getitem__(self,index):

        ID = index 
        obs = self.Data[ID]
        t = self.times[ID]

        return obs, t, ID
    
    
class Dynamics_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, Data, times,n_batch=1):
        'Initialization'
        self.times = times.float()
        self.Data = Data.float()
        self.n_batch = n_batch

    def __getitem__(self, index):
        
        ID = index 
        obs = self.Data[:,ID,:]
        t = self.times[ID]

        return obs, t, ID
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.times)

    
class Train_val_split:
    def __init__(self, IDs,val_size_fraction):
        
        
        IDs = np.random.permutation(IDs)
        
        self.IDs = IDs
        self.val_size = int(val_size_fraction*len(IDs))
    
    def train_IDs(self):
        train = sorted(self.IDs[:len(self.IDs)-self.val_size])
        
        return train
    
    def val_IDs(self):
        val = sorted(self.IDs[len(self.IDs)-self.val_size:])
        
        return val
    
    
class spectral_dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, Data, spectral_coeff):
        'Initialization'
        self.Data = Data.float()
        self.spectral_coeff = spectral_coeff.float()

    def __getitem__(self, index):
        
        ID = index 
        obs = self.Data[ID,...]
        obs_spec = self.spectral_coeff[ID,...]

        return obs, obs_spec
    def __len__(self):
        return self.Data.shape[0]
    
    
def plot_dim_vs_time(obs_to_print, time_to_print, z_to_print, dummy_times_to_print, z_all_to_print, frames_to_drop, path_to_save_plots, name, epoch, args):
    
    verbose=False
    
    if verbose: 
        print('[plot_dim_vs_time] obs_to_print.shape: ',obs_to_print.shape)
        print('[plot_dim_vs_time] time_to_print.shape: ',time_to_print.shape)
        print('[plot_dim_vs_time] args.num_dim_plot: ',args.num_dim_plot)
        print('[plot_dim_vs_time] dummy_times_to_print.shape: ',dummy_times_to_print.shape)
        print('[plot_dim_vs_time] z_all_to_print.shape: ',z_all_to_print.shape)
        
        
    n_plots_x = int(np.ceil(np.sqrt(args.num_dim_plot)))
    n_plots_y = int(np.floor(np.sqrt(args.num_dim_plot)))
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=(10, 10), sharex=True, dpi=100, facecolor='w', edgecolor='k')
    ax=ax.ravel()
    for idx in range(args.num_dim_plot):
        
        ax[idx].plot(dummy_times_to_print,z_all_to_print[:,idx],c='r', label='model')
        
        if frames_to_drop is not None and frames_to_drop>0:
            ax[idx].scatter(time_to_print[:-frames_to_drop],obs_to_print[:-frames_to_drop,idx],label='Data',c='blue', alpha=0.5)
            ax[idx].scatter(time_to_print[-frames_to_drop:],obs_to_print[-frames_to_drop:,idx],label='Hidden',c='green', alpha=0.5)
        else:
            ax[idx].scatter(time_to_print[:],obs_to_print[:,idx],label='Data',c='blue', alpha=0.5)
        ax[idx].set_xlabel("Time")
        ax[idx].set_ylabel("dim"+str(idx))
        
        ax[idx].legend()
        
    fig.tight_layout()

    if args.mode=='train' or path_to_save_plots is not None:
        plt.savefig(os.path.join(path_to_save_plots, name + str(epoch)))
        plt.close('all')
    else: plt.show()
    
    del obs_to_print, time_to_print, z_to_print, frames_to_drop
    
    
     
class spectral_dataset_generator(Dataset):
    def __init__(self, Data, times, segment_len, segment_window_factor, frames_to_drop):
        self.times = times.float()
        self.Data = Data.float()
        self.segment_len=segment_len
        self.segment_window_factor = segment_window_factor
        self.frames_to_drop = frames_to_drop
        

    def __getitem__(self, index):
        
        max_index = index+self.segment_window_factor*self.segment_len
        if max_index>=len(self.Data): max_index=len(self.Data)-self.segment_len 
        
        
        if int(max_index)==index:
            first_id = index
        else: first_id = np.random.randint(index, int(max_index))
        
        IDs = torch.arange(first_id,first_id+self.segment_len)
        
        
        
        obs = self.Data[IDs]
        
        
        t = self.times 

        return obs

    def __len__(self):
        return len(self.times)
