import copy

from tqdm import tqdm
import numpy as np

import torch
import torch.utils.data as utils


def supervised_IVIM(X_infer, bvalues, ivim_path, arg):
    from .deep import checkarg, Net, isnan

    arg = checkarg(arg)
    n_bval = len(bvalues)
    n_samples = len(X_infer)

    if arg.verbose:
        print(f'The number of samples are: {n_samples}')  
    
    # The b-values that get into the model need to be torch Tensor type.
    bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)
    X_infer = torch.from_numpy(X_infer).to(arg.train_pars.device)

    #load the pretrained network
    ivim_model = Net(bvalues, arg, arg.net_pars).to(arg.train_pars.device)
    if torch.cuda.is_available():
        ivim_model.load_state_dict(torch.load(ivim_path))
    else:
        ivim_model.load_state_dict(torch.load(ivim_path, map_location=torch.device('cpu')))        
    ivim_model.eval()
    
    ## normalise the signal to b=0 and remove data with nans
    if arg.key == 'phantom':
        S0 = np.mean(X_infer[:, bvalues == 100], axis=1).astype('<f')
        X_infer = X_infer / S0[:, None]
        nan_idx =  isnan(np.mean(X_infer, axis=1))
        X_infer = np.delete(X_infer, nan_idx , axis=0)
        print('phantom training')
    else:
        # print(torch.mean(X_infer[: , bvalues == 0], axis=1))
        if X_infer.ndim == 1:
            S0 = X_infer[bvalues == 0]
        else:
            S0 = torch.mean(X_infer[:, bvalues == 0], axis=1)      
        X_infer = X_infer / S0[:, None]
        nan_idx =  isnan(torch.mean(X_infer, axis=1))

        X_infer = X_infer.cpu().numpy()
        nan_idx = nan_idx.cpu().numpy()
        X_infer = np.delete(X_infer, nan_idx , axis=0)

    # Limiting the percentile threshold
    if arg.key == 'phantom':
        b_less_300_idx = np.percentile(X_infer[:, bvalues < 500], 95, axis=1) < 1.3 # X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
        b_greater_300_idx = np.percentile(X_infer[:, bvalues > 500], 95, axis=1) < 1.2 #  X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
        b_greater_700_idx = np.percentile(X_infer[:, bvalues > 1000], 95, axis=1) < 1 # X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
        thresh_idx = b_less_300_idx & b_greater_300_idx & b_greater_700_idx
    else: 
        bvalues = bvalues.cpu().numpy()
        b_less_50_idx = np.percentile(X_infer[:, bvalues < 50], 95, axis=1) < 1.3 # X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
        b_greater_50_idx = np.percentile(X_infer[:, bvalues > 50], 95, axis=1) < 1.2 #  X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
        b_greater_150_idx = np.percentile(X_infer[:, bvalues > 150], 95, axis=1) < 1 # X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
        thresh_idx = b_less_50_idx & b_greater_50_idx & b_greater_150_idx
    
    if arg.key== 'sim_boot': 
        size_boot= arg.sim.size_boot_low + arg.sim.size_boot_high + 1
        thresh_idx= np.broadcast_to(thresh_idx,( arg.sim.n_subsamples)) 
        indices= np.array(range(n_bval))
        subsamples= np.zeros( (arg.sim.n_subsamples, n_bval), dtype= 'uint8')   
        X_infer= np.broadcast_to( X_infer, (arg.sim.n_subsamples, n_bval)) 
        for i in range(arg.sim.n_subsamples): 
            subsample_low = np.random.choice(indices[1: (n_bval//2)+1], arg.sim.size_boot_low, replace= False)
            subsample_high = np.random.choice(indices[(n_bval//2)+1:], arg.sim.size_boot_high, replace= False)
            subsample= np.concatenate((subsample_low, subsample_high))  
            subsample= np.insert(subsample, 0, indices[0])
            subsamples[i, subsample]= 1    
        mask = subsamples.astype(bool)
        X_infer = X_infer[mask].reshape(-1, size_boot)
        bvalues_new= np.broadcast_to(bvalues,( arg.sim.n_subsamples, n_bval))
        bvalues_new = bvalues_new[mask].reshape(-1, size_boot)
        X_infer= np.concatenate((X_infer,bvalues_new), axis=1)
    elif arg.key== 'sim_bvalue':  
        if X_infer.ndim == 1:
            X_infer= np.concatenate((X_infer, bvalues))
        else:
            bvalues_new= np.broadcast_to(bvalues, (X_infer.shape[0], len(bvalues)))
            X_infer= np.concatenate((X_infer, bvalues_new), axis=1)  

    # ang.a: create fake labels so that the shape in the output is the same. note that we don't really use them
    labels = np.zeros((X_infer.shape[0], 1))
    suprevised_data = np.append(X_infer[thresh_idx, ], labels[thresh_idx, ], axis = 1)

    # initialise parameters and data
    Dp_infer = np.array([])
    Dt_infer = np.array([])
    Fp_infer = np.array([])
    S0_infer = np.array([])
    # Dp_orig = np.array([])
    # Dt_orig = np.array([])
    # Fp_orig = np.array([])
    # S0_orig = np.array([1])
    # initialise dataloader. Batch size can be way larger as we are still training.
    inferloader = utils.DataLoader(torch.from_numpy(suprevised_data.astype(np.float32)),
                                   batch_size=1, # previously was 2056,
                                   shuffle=False,
                                   drop_last=False)

    # start predicting
    with torch.no_grad():
        for i, suprevised_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            
            suprevised_batch = suprevised_batch.to(arg.train_pars.device)
            if arg.key== 'sim_boot': 
                X_batch = suprevised_batch[: ,:size_boot*2]
            elif arg.key== 'sim_bvalue':
                X_batch = suprevised_batch[: ,:n_bval*2]    
            else:    
                X_batch = suprevised_batch[:, :n_bval]
         
            # Dp_batch = suprevised_batch[: ,-1] 
            # Fp_batch = suprevised_batch[: ,-2] 
            # Dt_batch = suprevised_batch[: ,-3]
            
            # Dp_orig = np.append(Dp_orig, (Dp_batch.cpu()).numpy())
            # Dt_orig = np.append(Dt_orig, (Dt_batch.cpu()).numpy())
            # Fp_orig = np.append(Fp_orig, (Fp_batch.cpu()).numpy())
            
            # here the signal is predicted. Note that we now are interested in the parameters and no longer in the predicted signal decay.
            _, Dtt, Fpt, Dpt, S0t = ivim_model(X_batch, train_mode=False)
            # Quick and dirty solution to deal with networks not predicting S0
            try:
                S0 = np.append(S0, (S0t.cpu()).numpy())
            except:
                S0 = np.append(S0.cpu().numpy(), S0t.cpu().numpy())
            
            Dp_infer = np.append(Dp_infer, (Dpt.cpu()).numpy())
            Dt_infer = np.append(Dt_infer, (Dtt.cpu()).numpy())
            Fp_infer = np.append(Fp_infer, (Fpt.cpu()).numpy())
            S0_infer = np.append(S0_infer, (S0t.cpu()).numpy())
            # Error in precent -> devide the absolute error by the original value
  
    # swithc between Dt & Dp if the preduction is wrong
    if np.mean(Dp_infer) < np.mean(Dt_infer):
        Dp22 = copy.deepcopy(Dt_infer)
        Dt_infer = Dp_infer
        Dp_infer= Dp22
        Fp_infer = 1 - Fp_infer

    if arg.key== 'sim_boot': 
        Dp_infer= np.mean(Dp_infer, keepdims= True)
        Dt_infer= np.mean(Dt_infer, keepdims= True) 
        Fp_infer= np.mean(Fp_infer, keepdims= True)
        S0_infer= np.mean(S0_infer, keepdims= True)   
    #print(f'Dp_orig {Dp_orig} /n Dp_infer {Dp_infer} ')
    #print(f'Dp_orig-Dp_infer {Dp_orig-Dp_infer}')
    
    # e_calc_type = 'NRSE'
    # if e_calc_type == 'NRSE':
    #     Dp_norm_error = np.sqrt(np.square(Dp_orig-Dp_infer))/Dp_orig
    #     Dt_norm_error = np.sqrt(np.square(Dt_orig-Dt_infer))/Dt_orig
    #     Fp_norm_error = np.sqrt(np.square(Fp_orig-Fp_infer))/Fp_orig
    #     S0_norm_error = np.sqrt(np.square(S0_orig-S0_infer))
    # else: # NRMSE
    #     Dp_norm_error = np.sqrt(np.square(Dp_orig-Dp_infer)/Dp_orig)
    #     Dt_norm_error = np.sqrt(np.square(Dt_orig-Dt_infer)/Dt_orig)
    #     Fp_norm_error = np.sqrt(np.square(Fp_orig-Fp_infer)/Fp_orig)
    #     S0_norm_error = np.sqrt(np.square(S0_orig-S0_infer))
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    
    return [Dp_infer, Dt_infer, Fp_infer, S0_infer]
