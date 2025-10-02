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
from scipy.ndimage import gaussian_filter
import matplotlib.gridspec as gridspec

import torchcubicspline

from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

#Custom libraries
from spectral_source.utils import Select_times_function, EarlyStopping, SaveBestModel, to_np, load_checkpoint, Train_val_split, Dynamics_Dataset, Test_Dynamics_Dataset
from torch.utils.data import SubsetRandomSampler
#from spectral_source.solver import IESolver_monoidal
from spectral_source.Attentional_IE_solver import Integral_attention_solver, Integral_spatial_attention_solver, Integral_spatial_attention_solver_multbatch
from spectral_source.kernels import RunningAverageMeter, log_normal_pdf, normal_kl
#from spectral_source.utils import plot_reconstruction
from spectral_source.spectral_integrator import Chebyshev_nchannel, spectral_integrator_nchannels, spectral_integrator, Chebyshev
from spectral_source.spectral_ie_solver import Spectral_IE_solver
from spectral_source.utils import spectral_dataset, plot_dim_vs_time

#Torch libraries
import torch
from torch.nn import functional as F

if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"

    

def spectral_experiment(model, Data, spectral_coefficients, interpolator, time_seq, args, chebyshev_transform=None, cosine_transform=None, spectral_integrator=None, f_func = None):
    
    
    if args.model=='spectral': 
        str_model_name = "spectral"
    elif args.model=='nie': 
        str_model_name = "nie"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)

    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)

        num_experiments= len(txt)
        print("run:", num_experiments)
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)
            
        #with open(os.path.join(writer.log_dir,'commandline_args.txt'), 'w') as f:
        #    for key, value in args.__dict__.items(): 
        #        f.write('%s:%s\n' % (key, value))



    obs = Data
    spectral_coefficients = spectral_coefficients
    times = time_seq
    
    
    if chebyshev_transform is not None:
        transform = chebyshev_transform
    elif cosine_transform is not None:
        transform = cosine_transform
    
    if spectral_integrator is None:
        spectral_integrator = \
        spectral_integrator_nchannels(
            [spectral_coefficients.shape[i] for i in range(len(spectral_coefficients.shape))]
            )
    else:
        spectral_integrator = spectral_integrator
    
    All_parameters = model.parameters()
    
    if f_func is not None and args.f_nn is True:
        All_parameters = list(All_parameters) + list(f_func.parameters())
    if spectral_integrator is not None and args.integrator_nn is True:
        All_parameters = list(All_parameters) + list(spectral_integrator.parameters())
     
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        model, optimizer, scheduler, f_func = load_checkpoint(path, model, optimizer, scheduler, f_func)
        
    
    
    if args.mode=='train':
        #lr_scheduler = LRScheduler(optimizer,patience = 50,min_lr=1e-5,factor=0.1)
        early_stopping = EarlyStopping(patience=args.patience,min_delta=0)

        # Loss_print = []
        # Val_Loss = []
        all_train_loss=[]
        all_val_loss=[]
        
            
#         Data_splitting_indices = Train_val_split(np.copy(index_np),0)
#         Train_Data_indices = Data_splitting_indices.train_IDs()
#         Val_Data_indices = Data_splitting_indices.val_IDs()
#         print('\nlen(Train_Data_indices): ',len(Train_Data_indices))
#         print('Train_Data_indices: ',Train_Data_indices)
#         print('\nlen(Val_Data_indices): ',len(Val_Data_indices))
#         print('Val_Data_indices: ',Val_Data_indices)
        
        # Train Neural IDE
#         get_times = Select_times_function(times,extrapolation_points)

        save_best_model = SaveBestModel()
        
        
        split_size = int(args.training_split*obs.size(0))

        if args.mode == 'train':
            Dataset_train = spectral_dataset(Data[:obs.shape[0]-split_size,...],spectral_coefficients[:obs.shape[0]-split_size,...])
            Dataset_valid = spectral_dataset(Data[obs.shape[0]-split_size:,...],spectral_coefficients[obs.shape[0]-split_size:,...])
            train_loader = torch.utils.data.DataLoader(Dataset_train,batch_size=args.n_batch,shuffle=True)
            valid_loader = torch.utils.data.DataLoader(Dataset_valid,batch_size=args.n_batch,shuffle=False)
            
        
        start = time.time()
        for i in range(args.epochs):
            

            
            model.train()
            if f_func is not None:
                f_func.train()
            
            start_i = time.time()
            print('Epoch:',i)
            # GPUtil.showUtilization()
            counter=0
            train_loss = 0.0
            
#             perm = torch.randperm(obs.shape[0]).to(device)
#             perm2 = perm.clone()
#             spectral_shuffle, obs_shuffle = spectral_train[perm,...], obs_train[perm2,...]
            #spectral_shuffle, obs_shuffle = spectral_train[...], obs_train[...]
                
            #for j in tqdm(range(0,spectral_coefficients.size(0)-split_size,args.n_batch)):
            for  obs_, coeff_ in tqdm(train_loader): 
                
#                 coeff_ = spectral_shuffle[j:j+args.n_batch,...]
#                 obs_ = obs_shuffle[j:j+args.n_batch,...]
                nCheb = Chebyshev_nchannel(max_deg=args.max_deg,n_batch=obs_.shape[0],N_mc=args.N_MC,channels=args.dim,device=args.device)
                transform = nCheb
                
                coeff_ = coeff_.to(args.device)
                obs_ = obs_.to(args.device)
                    
                if args.perturbation_to_obs0 is not None:
                       perturb = torch.normal(mean=torch.zeros(obs_.shape[1]).to(args.device),
                                              std=args.std_noise)#args.perturbation_to_obs0*obs_[:3,:].std(dim=0))
                else:
                    perturb = torch.zeros_like(coeff_[0]).to(args.device)
                
                
                
                c= lambda x: f_func.forward(x.to(args.device).requires_grad_(True))

                
                    
#                 if args.ts_integration is not None:
#                     times_integration = args.ts_integration
#                 else:
#                     times_integration = torch.linspace(0,1,args.time_points)
                
                
                y_0 = c(coeff_)
                
                #start_ = time.time()
                a_ = Spectral_IE_solver(
                            x = torch.linspace(0,1,args.max_deg).to(device),
                            y_0 = y_0,
                            c = c,
                            model = model,
                            domain_dim = 1,
                            channels = args.dim,
                            max_deg = args.max_deg,
                            chebyshev = args.chebyshev,
                            chebyshev_coeff = coeff_,
                            spectral_integrator = spectral_integrator,
                            max_iterations = args.max_iterations,
                            smoothing_factor = 0.5,
                            store_intermediate_y = False,
                            return_function = False,
                            accumulate_grads = True,        
                            ).solve()
                #end_ = time.time()
                #print("Solving time: ", end_-start_)

                
                a_ = a_.view(a_.shape[0],args.max_deg,args.dim)
                #_start_ = time.time()
                if chebyshev_transform is not None:
                    a_func = transform.inverse(a_.requires_grad_(True))
                    z_ = a_func(times.requires_grad_(True))
                elif cosine_transform is not None:
                    z_ = transform.inverse(a_) 
                #_end_ = time.time()
                #print("Decoding: ",_end_-_start_)
                
                loss = F.mse_loss(z_, obs_)
                
                
                #_start = time.time()
                optimizer.zero_grad()
                loss.backward()#(retain_graph=True)
                optimizer.step()
                #_end = time.time()
                #print("Backward: ",_end-_start)
                
                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  split_size==0 and args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, obs_, z_, #ts_, ids_

            ## Validating
                
            model.eval()
            if f_func is not None:
                f_func.eval()
                
            with torch.no_grad():

                    #Only do this if there is a validation dataset
                
                val_loss = 0.0
                counter = 0
                if split_size>0:
                    
                    #for j in tqdm(range(obs.size(0)-split_size,obs.size(0),args.n_batch)):
                    for  obs_val, coeff_val in tqdm(valid_loader):  
                        
                        nCheb = Chebyshev_nchannel(max_deg=args.max_deg,n_batch=obs_val.shape[0],N_mc=args.N_MC,channels=args.dim,device=args.device)
                        transform = nCheb

                        coeff_val = coeff_val.to(args.device)
                        obs_val = obs_val.to(args.device)


                        if args.perturbation_to_obs0 is not None:
                               perturb = torch.normal(mean=torch.zeros(obs_val.shape[1]).to(args.device),
                                                      std=args.std_noise)#args.perturbation_to_obs0*obs_[:3,:].std(dim=0))
                        else:
                            perturb = torch.zeros_like(coeff_val[0]).to(args.device)



                        c= lambda x: f_func.forward(x.to(args.device))



#                         if args.ts_integration is not None:
#                             times_integration = args.ts_integration
#                         else:
#                             times_integration = torch.linspace(0,1,args.time_points)

                        
                        y_0 = c(coeff_val)

                        #start_ = time.time()
                        a_val = Spectral_IE_solver(
                                    x = torch.linspace(0,1,args.max_deg).to(device),
                                    y_0 = y_0,
                                    c = c,
                                    model = model,
                                    domain_dim = 1,
                                    channels = args.dim,
                                    max_deg = args.max_deg,
                                    chebyshev = args.chebyshev,
                                    chebyshev_coeff = coeff_val,
                                    spectral_integrator = spectral_integrator,
                                    max_iterations = args.max_iterations,
                                    smoothing_factor = 0.5,
                                    store_intermediate_y = False,
                                    return_function = False,
                                    accumulate_grads = False,        
                                    ).solve()
                        #end_ = time.time()
                        #print("Solving time: ", end_-start_)
                        

                        a_val = a_val.view(a_val.shape[0],args.max_deg,args.dim)
                        #_start_ = time.time()
                        if chebyshev_transform is not None:
                            a_func = transform.inverse(a_val)
                            z_val = a_func(times)
                        elif cosine_transform is not None:
                            z_val = transform.inverse(a_val) 
                        #_end_ = time.time()
                        #print("Decoding: ",_end_-_start_)

                        loss_validation = F.mse_loss(z_val, obs_val)
                        
                        del obs_val, z_val, a_val, coeff_val

                        counter += 1
                        val_loss += loss_validation.item()
                        
                        del loss_validation

                        #LRScheduler(loss_validation)
                        if args.lr_scheduler == 'ReduceLROnPlateau':
                            scheduler.step(val_loss)
                
                
                else: counter += 1

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            #writer.add_scalar('train_loss', all_train_loss[-1], global_step=i)
            #if len(all_val_loss)>0:
            #    writer.add_scalar('val_loss', all_val_loss[-1], global_step=i)
            #if args.lr_scheduler == 'ReduceLROnPlateau':
            #    writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], global_step=i)
            #elif args.lr_scheduler == 'CosineAnnealingLR':
            #    writer.add_scalar('Epoch/learning_rate', scheduler.get_last_lr()[0], global_step=i)

            
            with torch.no_grad():
                
                model.eval()
                if f_func is not None:
                    f_func.eval()
                
                
                if i % args.plot_freq == 0:
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
                    # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
                        
                    plt.plot(np.log10(all_train_loss),label='Train loss')
                    if split_size>0:
                        plt.plot(np.log10(all_val_loss),label='Val loss')
                    plt.xlabel("Epoch")
                    plt.ylabel("MSE Loss")
                    # timestr = time.strftime("%Y%m%d-%H%M%S")
                    #plt.show()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))

#                     for j in tqdm(range(0,obs.size(0),args.n_batch)):
                        
                        
                    coeff_test = spectral_coefficients[:args.n_batch,...]
                    obs_test = obs[:args.n_batch,...]
                
                    nCheb = Chebyshev_nchannel(max_deg=args.max_deg,n_batch=args.n_batch,N_mc=args.N_MC,channels=args.dim,device=args.device)
                    transform = nCheb

                    coeff_test = coeff_test.to(args.device)
                    obs_test = obs_test.to(args.device)


                    if args.perturbation_to_obs0 is not None:
                           perturb = torch.normal(mean=torch.zeros(obs_test.shape[1]).to(args.device),
                                                  std=args.std_noise)#args.perturbation_to_obs0*obs_[:3,:].std(dim=0))
                    else:
                        perturb = torch.zeros_like(coeff_test[0]).to(args.device)



                    c= lambda x: f_func.forward(x.to(args.device))



#                         if args.ts_integration is not None:
#                             times_integration = args.ts_integration
#                         else:
#                             times_integration = torch.linspace(0,1,args.time_points)

                    
                    y_0 = c(coeff_test)

                    #start_ = time.time()
                    a_test = Spectral_IE_solver(
                                x = torch.linspace(0,1,args.max_deg).to(device),
                                y_0 = y_0,
                                c = c,
                                model = model,
                                domain_dim = 1,
                                channels = args.dim,
                                max_deg = args.max_deg,
                                chebyshev = args.chebyshev,
                                chebyshev_coeff = coeff_test,
                                spectral_integrator = spectral_integrator,
                                max_iterations = args.max_iterations,
                                smoothing_factor = 0.5,
                                store_intermediate_y = False,
                                return_function = False,
                                accumulate_grads = False,        
                                ).solve()
                    #end_ = time.time()
                    #print("Solving time: ", end_-start_)


                    a_test = a_test.view(a_test.shape[0],args.max_deg,args.dim)
                    #_start_ = time.time()
                    if chebyshev_transform is not None:
                        a_func = transform.inverse(a_test)
                        z_test = a_func(times)
                    elif cosine_transform is not None:
                        z_test = transform.inverse(a_test) 
                    #_end_ = time.time()
                    #print("Decoding: ",_end_-_start_)




                    z_p = z_test

                    z_p = to_np(z_p)

                    obs_print = to_np(obs_test[0,...])

                    if args.dataset_name == 'fMRI' is False:
                        #plot_reconstruction(obs_print, z_p, None, path_to_save_plots, 'plot_epoch_', i, args)
                        plt.figure(1, figsize=(8,8),facecolor='w')

                        plt.scatter(obs_print[:,0],obs_print[:,1],label='Data')
                        plt.plot(z_p[0,:,0],z_p[0,:,1],label='Model')


                        plt.savefig(os.path.join(path_to_save_plots,'plot_'+str(i)))


                        plt.close('all')
                        del z_p, z_test, obs_print, a_test, y_0
                    else:
                        plot_dim_vs_time(obs_print,to_np(times),z_p[0,...],to_np(times),z_p[0,...],None,path_to_save_plots,'plot',i,args)
                        

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            if split_size>0:
                save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, f_func)
            else:
                save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, model, f_func)


            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break

        if args.support_tensors is True or args.support_test is True:
                del dummy_times
                
        end = time.time()
        
        tot_time = end - start
        f = open('time_results.txt','w')
        f.write(str(tot_time) + '\n')
        f.close()

        return tot_time
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')

        Dataset_test = spectral_dataset(Data,spectral_coefficients)
        # test_loader = torch.utils.data.DataLoader(Dataset_test,batch_size=args.n_batch,shuffle=False)
        test_loader = torch.utils.data.DataLoader(Dataset_test,batch_size=1,shuffle=False)
        
        ## Validating
        model.eval()
        if f_func is not None:
            f_func.eval()

        test_loss = 0.0
        all_test_loss=[]
        counter = 0
        
        for  obs_test, coeff_test in tqdm(test_loader): 
        
            #nCheb = Chebyshev_nchannel(max_deg=args.max_deg,n_batch=args.n_batch,N_mc=10000,device=args.device)
            nCheb = Chebyshev_nchannel(max_deg=args.max_deg,n_batch=1,N_mc=args.N_MC,channels=args.dim,device=args.device)
            transform = nCheb
    
            coeff_test = coeff_test.to(args.device)
            obs_test = obs_test.to(args.device)
    
    
            if args.perturbation_to_obs0 is not None:
                   perturb = torch.normal(mean=torch.zeros(obs_test.shape[1]).to(args.device),
                                          std=args.std_noise)#args.perturbation_to_obs0*obs_[:3,:].std(dim=0))
            else:
                perturb = torch.zeros_like(coeff_test[0]).to(args.device)
    
    
    
            c= lambda x: f_func.forward(x.to(args.device))
    
    
            y_0 = c(coeff_test)
    
            
            a_test = Spectral_IE_solver(
                        x = torch.linspace(0,1,args.max_deg).to(device),
                        y_0 = y_0,
                        c = c,
                        model = model,
                        domain_dim = 1,
                        channels = args.dim,
                        max_deg = args.max_deg,
                        chebyshev = args.chebyshev,
                        chebyshev_coeff = coeff_test,
                        spectral_integrator = spectral_integrator,
                        max_iterations = args.max_iterations,
                        smoothing_factor = 0.5,
                        store_intermediate_y = False,
                        return_function = False,
                        accumulate_grads = False,        
                        ).solve()
    
    
            a_test = a_test.view(a_test.shape[0],args.max_deg,args.dim)
            #_start_ = time.time()
            if chebyshev_transform is not None:
                a_func = transform.inverse(a_test)
                z_test = a_func(times)
            elif cosine_transform is not None:
                z_test = transform.inverse(a_test) 
            

            loss_test = F.mse_loss(z_test, obs_test)
            
            
            if args.dataset_name == 'fMRI':
                z_p = to_np(z_test[0,...])
                z_p = gaussian_filter(z_p,sigma=3)
                obs_p = to_np(obs_test[0,...])
                obs_p = gaussian_filter(obs_p,sigma=3)
                diff_p = np.abs(z_p-obs_p)**2
                
                save_dir = "Eval_plots"
                os.makedirs(save_dir, exist_ok=True)
                fig, axs = plt.subplots(3, 2, figsize=(8, 8), gridspec_kw={'width_ratios': [20, 1]})
                # First image (full width)
                axs[0, 0].imshow(z_p)
                axs[0, 1].axis("off")  # empty slot
                # Second image (full width)
                axs[1, 0].imshow(obs_p)
                axs[1, 1].axis("off")
                # Third image with colorbar
                im = axs[2, 0].imshow(diff_p, cmap="inferno")
                axs[2, 0].set_title("Error Map")
                cbar = fig.colorbar(im, cax=axs[2, 1],shrink=0.75) 
                cbar.set_label("Error intensity")
                plt.tight_layout()
                # Save in Eval_plots
                save_path = os.path.join(save_dir, f"output{counter}.png")
                plt.savefig(save_path, dpi=300)
                plt.show()
            
            del obs_test, z_test, a_test, coeff_test

            counter += 1
            test_loss += loss_test.item()
            all_test_loss.append(loss_test.item())
            
            del loss_test
        

        test_loss /= counter

        return test_loss, torch.tensor(all_test_loss).std()
        
