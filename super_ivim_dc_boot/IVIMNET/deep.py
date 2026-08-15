"""
Built on code by Sebastiano Barbieri: https://github.com/sebbarb/deep_ivim
"""
import os
import copy
import warnings

from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from torch.utils.tensorboard import SummaryWriter

from . import fitting_algorithms as fit


# Define the neural network.
class Net(nn.Module):
    def __init__(self, bvalues,arg, net_pars, supervised = False):
        """
        this defines the Net class which is the network we want to train.
        :param bvalues: a 1D array with the b-values
        :param net_pars: an object with network design options, as explained in the publication, with attributes:
        fitS0 --> Boolean determining whether S0 is fixed to 1 (False) or fitted (True)
        times len(bvalues), with data sorted per voxel. This option was not explored in the publication
        dropout --> Number between 0 and 1 indicating the amount of dropout regularisation
        batch_norm --> Boolean determining whether to use batch normalisation
        parallel --> Boolean determining whether to use separate networks for estimating the different IVIM parameters
        (True), or have them all estimated by a single network (False)
        con --> string which determines what type of constraint is used for the parameters. Options are:
        'sigmoid' allowing a sigmoid constraint
        'abs' having the absolute of the estimated values to constrain parameters to be positive
        'none' giving no constraints
        cons_min --> 1D array, if sigmoid is the constraint, these values give [Dmin, fmin, D*min, S0min]
        cons_max --> 1D array, if sigmoid is the constraint, these values give [Dmax, fmax, D*max, S0max]
        depth --> integer giving the network depth (number of layers)
        """
        super(Net, self).__init__()
        self.supervised = supervised
        self.bvalues = bvalues
        self.arg= arg
        self.net_pars = net_pars
        torch.manual_seed(self.arg.sim.seed)
        if arg.key== 'sim_boot':
            size_boot= arg.sim.size_boot_low + arg.sim.size_boot_high + 1
            self.net_pars.width = 2*size_boot
        elif arg.key== 'sim_bvalue':
            size = arg.sim.size_low + arg.sim.size_high + 1
            self.net_pars.width = 2*size   
        else: 
            self.net_pars.width = len(bvalues)    
        # define number of parameters being estimated
        self.est_pars = 3
        if self.net_pars.fitS0:
            self.est_pars += 1
        # define number of outputs, if neighbours are taken along, we expect 9 outputs, otherwise 1
        self.outs = 1
        # define module lists. If network is not parallel, we can do with 1 list, otherwise we need a list per parameter
        self.fc_layers = nn.ModuleList()
        if self.net_pars.parallel:
            self.fc_layers2 = nn.ModuleList()
            self.fc_layers3 = nn.ModuleList()
            self.fc_layers4 = nn.ModuleList()
       
        # loop over the layers
        if arg.key== 'sim_boot':
            width = 2*size_boot
        elif arg.key== 'sim_bvalue':
            width = 2*size    
        else: 
            width = len(bvalues)
        for i in range(self.net_pars.depth):
            # extend with a fully-connected linear layer
            self.linear_layer = nn.Linear(width, self.net_pars.width)
            nn.init.kaiming_normal_(self.linear_layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.linear_layer.bias)
            self.fc_layers.extend([self.linear_layer])
            if self.net_pars.parallel:
                self.linear_layer = nn.Linear(width, self.net_pars.width)
                nn.init.kaiming_normal_(self.linear_layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(self.linear_layer.bias)
                self.fc_layers2.extend([self.linear_layer])
                self.linear_layer = nn.Linear(width, self.net_pars.width)
                nn.init.kaiming_normal_(self.linear_layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(self.linear_layer.bias)
                self.fc_layers3.extend([self.linear_layer])
                self.linear_layer = nn.Linear(width, self.net_pars.width)
                nn.init.kaiming_normal_(self.linear_layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(self.linear_layer.bias)
                self.fc_layers4.extend([self.linear_layer])
            width = self.net_pars.width
            # if desired, add batch normalisation
            if self.net_pars.batch_norm:
                self.fc_layers.extend([nn.BatchNorm1d(self.net_pars.width)])
                if self.net_pars.parallel:
                    self.fc_layers2.extend([nn.BatchNorm1d(self.net_pars.width)])
                    self.fc_layers3.extend([nn.BatchNorm1d(self.net_pars.width)])
                    self.fc_layers4.extend([nn.BatchNorm1d(self.net_pars.width)])
            # add ELU units for non-linearity          
            self.fc_layers.extend([nn.ELU()])
            if self.net_pars.parallel:
                self.fc_layers2.extend([nn.ELU()])
                self.fc_layers3.extend([nn.ELU()])
                self.fc_layers4.extend([nn.ELU()])
            # if dropout is desired, add dropout regularisation
            if self.net_pars.dropout != 0:
                self.dropout_layer = nn.Dropout(self.net_pars.dropout)
                self.fc_layers.extend([self.dropout_layer])
                if self.net_pars.parallel:
                    self.fc_layers2.extend([self.dropout_layer])
                    self.fc_layers3.extend([self.dropout_layer])
                    self.fc_layers4.extend([self.dropout_layer])
          
        # Final layer yielding output, with either 3 (fix S0) or 4 outputs of a single network, or 1 output
        # per network in case of parallel networks.
        if self.net_pars.parallel:
            self.linear_layer_out = nn.Linear(self.net_pars.width, self.outs)
            nn.init.kaiming_normal_(self.linear_layer_out.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.linear_layer_out.bias)
            self.encoder = nn.Sequential(*self.fc_layers, self.linear_layer_out)
            self.linear_layer_out = nn.Linear(self.net_pars.width, self.outs)
            nn.init.kaiming_normal_(self.linear_layer_out.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.linear_layer_out.bias)           
            self.encoder2 = nn.Sequential(*self.fc_layers2, self.linear_layer_out)
            self.linear_layer_out = nn.Linear(self.net_pars.width, self.outs)
            nn.init.kaiming_normal_(self.linear_layer_out.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.linear_layer_out.bias)
            self.encoder3 = nn.Sequential(*self.fc_layers3, self.linear_layer_out)          
            if self.net_pars.fitS0:
                self.linear_layer_out = nn.Linear(self.net_pars.width, self.outs)
                nn.init.kaiming_normal_(self.linear_layer_out.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(self.linear_layer_out.bias)
                self.encoder4 = nn.Sequential(*self.fc_layers4, self.linear_layer_out)
        else:
            self.linear_layer_out = nn.Linear(self.net_pars.width, self.est_pars * self.outs)
            nn.init.kaiming_normal_(self.linear_layer_out.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.linear_layer_out.bias)
            self.encoder = nn.Sequential(*self.fc_layers, self.linear_layer_out)


    def forward(self, X, train_mode = True):
        # select constraint method
        if self.net_pars.con == 'sigmoid':
            # define constraints
            Dmin = self.net_pars.cons_min[0]
            Dmax = self.net_pars.cons_max[0]
            fmin = self.net_pars.cons_min[1]
            fmax = self.net_pars.cons_max[1]
            Dpmin = self.net_pars.cons_min[2]
            Dpmax = self.net_pars.cons_max[2]
            S0min = self.net_pars.cons_min[3]
            S0max = self.net_pars.cons_max[3]
            # this network constrains the estimated parameters between two values by taking the sigmoid.
            # Advantage is that the parameters are constrained and that the mapping is unique.
            # Disadvantage is that the gradients go to zero close to the prameter bounds.
            params1 = self.encoder(X)
            # if parallel again use each param comes from a different output
            if self.net_pars.parallel:
                params2 = self.encoder2(X)
                params3 = self.encoder3(X)
                if self.net_pars.fitS0:
                    params4 = self.encoder4(X)
        elif self.net_pars.con == 'none' or self.net_pars.con == 'abs':
            if self.net_pars.con == 'abs':
                # this network constrains the estimated parameters to be positive by taking the absolute.
                # Advantage is that the parameters are constrained and that the derrivative of the function remains
                # constant. Disadvantage is that -x=x, so could become unstable.
                params1 = torch.abs(self.encoder(X))
                if self.net_pars.parallel:
                    params2 = torch.abs(self.encoder2(X))
                    params3 = torch.abs(self.encoder3(X))
                    if self.net_pars.fitS0:
                        params4 = torch.abs(self.encoder4(X))
            else:
                # this network is not constraint
                params1 = self.encoder(X)
                if self.net_pars.parallel:
                    params2 = self.encoder2(X)
                    params3 = self.encoder3(X)
                    if self.net_pars.fitS0:
                        params4 = self.encoder4(X)
        else:
            raise Exception('the chose parameter constraint is not implemented. Try ''sigmoid'', ''none'' or ''abs''')
        X_temp=[]
        for aa in range(self.outs):
            if self.net_pars.con == 'sigmoid':
                # applying constraints
                if self.net_pars.parallel:
                    Dp = Dpmin + torch.sigmoid(params1[:, aa].unsqueeze(1)) * (Dpmax - Dpmin)
                    Dt = Dmin + torch.sigmoid(params2[:, aa].unsqueeze(1)) * (Dmax - Dmin)
                    Fp = fmin + torch.sigmoid(params3[:, aa].unsqueeze(1)) * (fmax - fmin)
                    if self.net_pars.fitS0:
                        S0 = S0min + torch.sigmoid(params4[:, aa].unsqueeze(1)) * (S0max - S0min)
                else:
                    Dp = Dpmin + torch.sigmoid(params1[:, aa * self.est_pars + 0].unsqueeze(1)) * (Dpmax - Dpmin)
                    Dt = Dmin + torch.sigmoid(params1[:, aa * self.est_pars + 1].unsqueeze(1)) * (Dmax - Dmin)
                    Fp = fmin + torch.sigmoid(params1[:, aa * self.est_pars + 2].unsqueeze(1)) * (fmax - fmin)
                    if self.net_pars.fitS0:
                        S0 = S0min + torch.sigmoid(params1[:, aa * self.est_pars + 3].unsqueeze(1)) * (S0max - S0min)
            elif self.net_pars.con == 'none' or self.net_pars.con == 'abs':
                if self.net_pars.parallel:
                    Dp = params1[:, aa].unsqueeze(1)
                    Dt = params2[:, aa].unsqueeze(1)
                    Fp = params3[:, aa].unsqueeze(1)
                    if self.net_pars.fitS0:
                        S0 = params4[:, aa].unsqueeze(1)
                else:
                    Dp = params1[:, aa * self.est_pars + 0].unsqueeze(1)
                    Dt = params1[:, aa * self.est_pars + 1].unsqueeze(1)
                    Fp = params1[:, aa * self.est_pars + 2].unsqueeze(1)
                    if self.net_pars.fitS0:
                        S0 = params1[:, aa * self.est_pars + 3].unsqueeze(1)
            # the central voxel will give the estimates of D, f and D*. In all other cases a is always 0.
            if aa == 0:
                if self.supervised == False:
                    Dpout = copy.copy(Dp)
                    Dtout = copy.copy(Dt)
                    Fpout = copy.copy(Fp)
                    if self.net_pars.fitS0:
                        S0out = copy.copy(S0)
                else:
                    Dpout = Dp
                    Dtout = Dt
                    Fpout = Fp
                    if self.net_pars.fitS0:
                        S0out = S0        

            # here we estimate X, the signal as function of b-values given the predicted IVIM parameters. Although
            # this parameter is not interesting for prediction, it is used in the loss function
            # in this a>0 case, we fill up the predicted signal of the neighbouring voxels too, as these are used in
            # the loss function.
            if self.net_pars.fitS0:
                X_temp.append(S0 * (Fp * torch.exp(-self.bvalues * (Dt + Dp)) + (1 - Fp) * torch.exp(-self.bvalues * Dt)))
            else:
                X_temp.append((Fp * torch.exp(-self.bvalues * (Dt + Dp)) + (1 - Fp) * torch.exp(-self.bvalues * Dt)))   
        X = torch.cat(X_temp,dim=1)
        if self.arg.key == 'sim_boot':
            size_boot= self.arg.sim.size_boot_low + self.arg.sim.size_boot_high + 1
            indices= torch.tensor((range(len(self.bvalues))))
            subsamples= torch.zeros_like( X , dtype= torch.uint8)
            np.random.seed(0)
            for i in range( X.shape[0]):  
                subsample_low = np.random.choice(indices[1: (len(self.bvalues)//2)+1], self.arg.sim.size_boot_low, replace= False)
                subsample_high = np.random.choice(indices[(len(self.bvalues)//2)+1:], self.arg.sim.size_boot_high, replace= False)
                subsample= np.concatenate((subsample_low, subsample_high)) 
                subsample= np.insert(subsample, 0, indices[0])
                subsamples[i, subsample]= 1 
            mask = subsamples.bool()
            X = X[mask].reshape(-1, size_boot) 
        elif self.arg.key== 'sim_bvalue' and train_mode:
            size = self.arg.sim.size_low + self.arg.sim.size_high + 1
            indices= torch.tensor((range(len(self.bvalues))))
            subsamples= torch.zeros_like( X , dtype= torch.uint8)
            np.random.seed(0)
            for i in range( X.shape[0]): 
                subsample_low = np.random.choice(indices[1:10], self.arg.sim.size_low, replace= False)
                subsample_high = np.random.choice(indices[10:], self.arg.sim.size_high, replace= False)
                subsample= np.concatenate((subsample_low, subsample_high))        
                subsample= np.insert(subsample, 0, indices[0])
                subsamples[i, subsample]= 1    
            mask = subsamples.bool()
            X = X[mask].reshape(-1, size)
        else:
            X = X  
        if self.net_pars.fitS0:
            return X, Dtout, Fpout, Dpout, S0out
        else:
            return X, Dtout, Fpout, Dpout, torch.ones(len(Dtout))

def learn_IVIM(X_train, bvalues, arg, sf, mode, work_dir, filename, net=None):
    """
    This program builds a IVIM-NET network and trains it.
    :param X_train: 2D array of IVIM data we use for training. First axis are the voxels and second axis are the b-values
    :param bvalues: a 1D array with the b-values
    :param arg: an object with network design options, as explained in the publication check hyperparameters.py for
    options
    :param net: an optional input pre-trained network with initialized weights for e.g. transfer learning or warm start
    :return net: returns a trained network
    """
    if arg.verbose:
        print(X_train.shape[0], 'noam')
    torch.backends.cudnn.benchmark = True
    arg = checkarg(arg)

    ## normalise the signal to b=0 and remove data with nans
    if arg.key == 'phantom':
        S0 = np.mean(X_train[:, bvalues == 100], axis=1).astype('<f')
        X_train = X_train / S0[:, None]
        np.delete(X_train, isnan(np.mean(X_train, axis=1)), axis=0)
        print('phantom training')
    else:
        S0 = np.mean(X_train[:, bvalues == 0], axis=1).astype('<f')
        X_train = X_train / S0[:, None]
        np.delete(X_train, isnan(np.mean(X_train, axis=1)), axis=0)
            

    # removing non-IVIM-like data; this often gets through when background data is not correctly masked
    # Estimating IVIM parameters in these data is meaningless anyways.
    if arg.key == 'phantom':
        X_train = X_train[np.percentile(X_train[:, bvalues < 500], 95, axis=1) < 1.3]
        X_train = X_train[np.percentile(X_train[:, bvalues > 500], 95, axis=1) < 1.2]
        X_train = X_train[np.percentile(X_train[:, bvalues > 1000], 95, axis=1) < 1.0]
        
    else: 
        X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
        X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
        X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
    X_train[X_train > 1.5] = 1.5

    # initialising the network of choice using the input argument arg
    if net is None:
        print('debug')
        bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)
        net = Net(bvalues, arg ,arg.net_pars).to(arg.train_pars.device)
    #NOAM else:
        # if a network was used as input parameter, work with that network instead (transfer learning/warm start).
        #net.to(arg.train_pars.device)

    # defining the loss function; not explored in the publication
    if arg.train_pars.loss_fun == 'rms':
        criterion = nn.MSELoss(reduction='mean').to(arg.train_pars.device)
    elif arg.train_pars.loss_fun == 'L1':
        criterion = nn.L1Loss(reduction='mean').to(arg.train_pars.device)

    # splitting data into learning and validation set; subsequently initialising the Dataloaders
    split = int(np.floor(len(X_train) * arg.train_pars.split))
    train_set, val_set = torch.utils.data.random_split(torch.from_numpy(X_train.astype(np.float32)),
                                                       [split, len(X_train) - split])
    # train loader loads the trianing data. We want to shuffle to make sure data order is modified each epoch and different data is selected each epoch.
    trainloader = utils.DataLoader(train_set,
                                   batch_size=arg.train_pars.batch_size,
                                   shuffle=True,
                                   drop_last=True)
    # validation data is loaded here. By not shuffling, we make sure the same data is loaded for validation every time. We can use substantially more data per batch as we are not training.
    inferloader = utils.DataLoader(val_set,
                                   batch_size=32 * arg.train_pars.batch_size,
                                   shuffle=False,
                                   drop_last=True)

    # defining the number of training and validation batches for normalisation later
    totalit = np.min([arg.train_pars.maxit, np.floor(split // arg.train_pars.batch_size)])
    batch_norm2 = np.floor(len(val_set) // (32 * arg.train_pars.batch_size))

    # defining optimiser
    if arg.train_pars.scheduler:
        optimizer, scheduler = load_optimizer(net, arg)
    else:
        optimizer = load_optimizer(net, arg)

    # Initialising parameters
    best = 1e16
    num_bad_epochs = 0
    loss_train = []
    loss_val = []
    prev_lr = 0
    # get_ipython().run_line_magic('matplotlib', 'inline')
    final_model = copy.deepcopy(net.state_dict())

    ## Train
    for epoch in range(1000):
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        # initialising and resetting parameters
        net.train()
        running_loss_train = 0.
        running_loss_val = 0.
        #losstotcon = 0.
        maxloss = 0.
        for i, X_batch in enumerate(tqdm(trainloader, position=0, leave=True, total=totalit), 0):
            if i > totalit:
                # have a maximum number of batches per epoch to ensure regular updates of whether we are improving
                break
            # zero the parameter gradients
            optimizer.zero_grad()
            # put batch on GPU if pressent
            X_batch = X_batch.to(arg.train_pars.device)
            ## forward + backward + optimize
            X_pred, Dt_pred, Fp_pred, Dp_pred, S0pred = net(X_batch)
            # removing nans and too high/low predictions to prevent overshooting
            X_pred[isnan(X_pred)] = 0
            X_pred[X_pred < 0] = 0
            X_pred[X_pred > 3] = 3
            # determine loss for batch; note that the loss is determined by the difference between the predicted signal and the actual signal. The loss does not look at Dt, Dp or Fp.
            loss = criterion(X_pred, X_batch)
            # updating network
            loss.backward()
            optimizer.step()
            # total loss and determine max loss over all batches
            running_loss_train += loss.item()
            if loss.item() > maxloss:
                maxloss = loss.item()
        # show some figures if desired, to show whether there is a correlation between Dp and f
        if arg.fig:
            plt.figure(3)
            plt.clf()
            plt.plot(Dp_pred.tolist(), Fp_pred.tolist(), 'rx', markersize=5)
            plt.ion()
            plt.show()
        # after training, do validation in unseen data without updating gradients
        print('\n validation \n')
        net.eval()
        # validation is always done over all validation data
        for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            optimizer.zero_grad()
            X_batch = X_batch.to(arg.train_pars.device)
            # do prediction, only look at predicted IVIM signal
            X_pred, _, _, _, _ = net(X_batch)
            X_pred[isnan(X_pred)] = 0
            X_pred[X_pred < 0] = 0
            X_pred[X_pred > 3] = 3
            # validation loss
            loss = criterion(X_pred, X_batch)
            running_loss_val += loss.item()
        # scale losses
        running_loss_train = running_loss_train / totalit
        running_loss_val = running_loss_val / batch_norm2
        # save loss history for plot
        loss_train.append(running_loss_train)
        loss_val.append(running_loss_val)
        # as discussed in the article, LR is important. This approach allows to reduce the LR if we think it is too
        # high, and return to the network state before it went poorly
        if arg.train_pars.scheduler:
            scheduler.step(running_loss_val)
            if optimizer.param_groups[0]['lr'] < prev_lr:
                net.load_state_dict(final_model)
            prev_lr = optimizer.param_groups[0]['lr']
        # print stuff
        print("\nLoss: {loss}, validation_loss: {val_loss}, lr: {lr}".format(loss=running_loss_train,
                                                                             val_loss=running_loss_val,
                                                                             lr=optimizer.param_groups[0]['lr']))
        # early stopping criteria
        if running_loss_val < best:
            print("\n############### Saving good model ###############################")
            final_model = copy.deepcopy(net.state_dict())
            best = running_loss_val
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == arg.train_pars.patience:
                print("\nDone, best val loss: {}".format(best))
                break
        # plot loss and plot 4 fitted curves
        #if epoch > 0:  TODO remove the comment
            # plot progress and intermediate results (if enabled) # TODO remove the comment
            # plot_progress(X_batch.cpu(), X_pred.cpu(), bvalues.cpu(), loss_train, loss_val, arg)
    print("Done")
    # save final fits
    if arg.fig:
        if not os.path.isdir('plots'):
            os.makedirs('plots')
        plt.figure(1)
        plt.gcf()
        plt.savefig('plots/fig_fit.png')
        plt.figure(2)
        plt.gcf()
        plt.savefig('plots/fig_train.png')
        plt.close('all')
    # Restore best model
    if arg.train_pars.select_best:
        net.load_state_dict(final_model)
    del trainloader
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    #save the model (Elad addition)
    save_state_model = True
    if save_state_model:
        final_weights = net.state_dict()
        torch.save(final_weights, f'{work_dir}/{filename}.pt')
    return net

def load_optimizer(net, arg):
    if arg.net_pars.parallel:
        if arg.net_pars.fitS0:
            par_list = [{'params': net.encoder.parameters(), 'lr': arg.train_pars.lr},
                        {'params': net.encoder2.parameters()}, {'params': net.encoder3.parameters()},
                        {'params': net.encoder4.parameters()}]
        else:
            par_list = [{'params': net.encoder.parameters(), 'lr': arg.train_pars.lr},
                        {'params': net.encoder2.parameters()}, {'params': net.encoder3.parameters()}]
    else:
        par_list = [{'params': net.encoder.parameters()}]
    if arg.train_pars.optim == 'adam':
        optimizer = optim.Adam(par_list, lr=arg.train_pars.lr, weight_decay=1e-4)
    elif arg.train_pars.optim == 'sgd':
        optimizer = optim.SGD(par_list, lr=arg.train_pars.lr, momentum=0.9, weight_decay=1e-4)
    elif arg.train_pars.optim == 'adagrad':
        optimizer = torch.optim.Adagrad(par_list, lr=arg.train_pars.lr, weight_decay=1e-4)
    if arg.train_pars.scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2,
                                                         patience=round(arg.train_pars.patience / 2))
        return optimizer, scheduler
    else:
        return optimizer


def predict_IVIM(data, bvalues, net, arg):
    """
    This program takes a trained network and predicts the IVIM parameters from it.
    :param data: 2D array of IVIM data we want to predict the IVIM parameters from. First axis are the voxels and second axis are the b-values
    :param bvalues: a 1D array with the b-values
    :param net: the trained IVIM-NET network
    :param arg: an object with network design options, as explained in the publication check hyperparameters.py for
    options
    :return param: returns the predicted parameters
    """
    arg = checkarg(arg)
    
    if arg.key == 'phantom':
        S0 = np.mean(data[:, bvalues == 100], axis=1).astype('<f')
        data = data / S0[:, None]
        np.delete(data, isnan(np.mean(data, axis=1)), axis=0)
        print('phantom training')

    else:
    ## normalise the signal to b=0 and remove data with nans
        S0 = np.mean(data[:, bvalues == 0], axis=1).astype('<f')
        data = data / S0[:, None]
        np.delete(data, isnan(np.mean(data, axis=1)), axis=0)
    # skip nans.
    mylist = isnan(np.mean(data, axis=1))
    sels = [not i for i in mylist]
    # remove data with non-IVIM-like behaviour. Estimating IVIM parameters in these data is meaningless anyways.
    if arg.key == 'phantom':
        sels = sels & (np.percentile(data[:, bvalues < 500], 0.95, axis=1) < 1.3) & (
                    np.percentile(data[:, bvalues > 500], 0.95, axis=1) < 1.2) & (
                           np.percentile(data[:, bvalues > 1000], 0.95, axis=1) < 1.0)
    else: 
        sels = sels & (np.percentile(data[:, bvalues < 50], 0.95, axis=1) < 1.3) & (
                    np.percentile(data[:, bvalues > 50], 0.95, axis=1) < 1.2) & (
                           np.percentile(data[:, bvalues > 150], 0.95, axis=1) < 1.0)
    # we need this for later
    lend = len(data)
    data = data[sels]

    # tell net it is used for evaluation
    net.eval()
    # initialise parameters and data
    Dp = np.array([])
    Dt = np.array([])
    Fp = np.array([])
    S0 = np.array([])
    # initialise dataloader. Batch size can be way larger as we are still training.
    inferloader = utils.DataLoader(torch.from_numpy(data.astype(np.float32)),
                                   batch_size=2056,
                                   shuffle=False,
                                   drop_last=False)
    # start predicting
    with torch.no_grad():
        for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            X_batch = X_batch.to(arg.train_pars.device)
            # here the signal is predicted. Note that we now are interested in the parameters and no longer in the predicted signal decay.
            _, Dtt, Fpt, Dpt, S0t = net(X_batch)
            # Quick and dirty solution to deal with networks not predicting S0
            try:
                S0 = np.append(S0, (S0t.cpu()).numpy())
            except:
                S0 = np.append(S0, S0t)
            Dp = np.append(Dp, (Dpt.cpu()).numpy())
            Dt = np.append(Dt, (Dtt.cpu()).numpy())
            Fp = np.append(Fp, (Fpt.cpu()).numpy())
    # The 'abs' and 'none' constraint networks have no way of figuring out what is D and D* a-priori. However, they do
    # tend to pick one output parameter for D or D* consistently within the network. If the network has swapped D and
    # D*, we swap them back here.
    if np.mean(Dp) < np.mean(Dt):
        Dp22 = copy.deepcopy(Dt)
        Dt = Dp
        Dp = Dp22
        Fp = 1 - Fp
    # here we correct for the data that initially was removed as it did not have IVIM behaviour, by returning zero
    # estimates
    Dptrue = np.zeros(lend)
    Dttrue = np.zeros(lend)
    Fptrue = np.zeros(lend)
    S0true = np.zeros(lend)
    Dptrue[sels] = Dp
    Dttrue[sels] = Dt
    Fptrue[sels] = Fp
    S0true[sels] = S0
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    return [Dttrue, Fptrue, Dptrue, S0true]


def isnan(x):
    # this program indicates what are NaNs 
    return x != x


def plot_progress(X_batch, X_pred, bvalues, loss_train, loss_val, arg):
    # this program plots the progress of the training. It will plot the loss and validatin loss, as well as 4 IVIM curve
    # fits to 4 data points from the input
    inds1 = np.argsort(bvalues)
    X_batch = X_batch[:, inds1]
    X_pred = X_pred[:, inds1]
    bvalues = bvalues[inds1]
    if arg.fig:
        plt.close('all')
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(bvalues, X_batch.data[0], 'o')
        axs[0, 0].plot(bvalues, X_pred.data[0])
        axs[0, 0].set_ylim(min(X_batch.data[0]) - 0.3, 1.2 * max(X_batch.data[0]))
        axs[1, 0].plot(bvalues, X_batch.data[1], 'o')
        axs[1, 0].plot(bvalues, X_pred.data[1])
        axs[1, 0].set_ylim(min(X_batch.data[1]) - 0.3, 1.2 * max(X_batch.data[1]))
        axs[0, 1].plot(bvalues, X_batch.data[2], 'o')
        axs[0, 1].plot(bvalues, X_pred.data[2])
        axs[0, 1].set_ylim(min(X_batch.data[2]) - 0.3, 1.2 * max(X_batch.data[2]))
        axs[1, 1].plot(bvalues, X_batch.data[3], 'o')
        axs[1, 1].plot(bvalues, X_pred.data[3])
        axs[1, 1].set_ylim(min(X_batch.data[3]) - 0.3, 1.2 * max(X_batch.data[3]))
        for ax in axs.flat:
            ax.set(xlabel='b-value (s/mm2)', ylabel='signal (a.u.)')
        for ax in axs.flat:
            ax.label_outer()
        plt.ion()
        plt.show()
        plt.pause(0.001)
        plt.figure(2)
        plt.clf()
        plt.plot(loss_train)
        plt.plot(loss_val)
        plt.yscale("log")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.ion()
        plt.show()
        plt.pause(0.001)


def make_data_complete(dw_data,bvalues,fraction_threshold=0.2):
    """
    This function is specific to missing data. For example, due to motion, after image registration our dataset
    contained gaps of information in some patients. As the Neural Network might get confused by empty slots,
    this program was desigend to fill up these slots with more realistic data estimates.

    :param bvalues: Array with the b-values
    :param dw_data: 1D Array with diffusion-weighted signal at different b-values
    :param fraction_threshold: an optional parameter determining the maximum fraction of missing data allowed.
    if more data is missing, the algorithm will not correct to prrvent too unrealistic (noiseless) data.

    :return dw_data: corrected dataset
    """
    if len(np.shape(dw_data)) == 4:
        sx, sy, sz, n_b_values = dw_data.shape
        dw_data = np.reshape(dw_data, (sx * sy * sz, n_b_values))
        reshape = True
    dw_data[isnan(dw_data)] = 0
    zeros = (dw_data == 0)
    locs = np.mean(zeros,axis=1)
    sels = (locs > 0) & (locs < fraction_threshold)
    data_to_correct = dw_data[sels,:]
    print('correcting {} datapoints'.format(len(data_to_correct)))
    def parfun(i):
        datatemp = data_to_correct[i,:]
        nonzeros = datatemp > 0
        bvaltemp = bvalues[nonzeros]
        datatempf=datatemp[nonzeros]
        norm=np.nanmean(datatempf)
        datatemp = datatemp / norm
        datatempf = datatempf / norm
        [Dt,Fp,Dp,S0]=fit.fit_least_squares(bvaltemp, datatempf, S0_output=True, fitS0=True, bounds=([0, 0, 0, 0.8], [0.005, 0.7, 0.3, 3]))
        datatemp[~nonzeros] = fit.ivim(bvalues,Dt,Fp,Dp,S0)[~nonzeros]
        return datatemp * norm
    data_to_correct = Parallel(n_jobs=4,batch_size=64)(delayed(parfun)(i) for i in tqdm(range(len(data_to_correct)), position=0,
                                                                    leave=True))
    dw_data[sels, :] = data_to_correct
    if reshape:
        dw_data = np.reshape(dw_data, (sx, sy, sz, n_b_values))
    return dw_data


def checkarg_train_pars(arg):
    if not hasattr(arg,'optim'):
        warnings.warn('arg.train.optim not defined. Using default ''adam''')
        arg.optim = 'adam'  # these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
    if not hasattr(arg,'lr'):
        warnings.warn('arg.train.lr not defined. Using default value 0.0001')
        arg.lr = 0.0001  # this is the learning rate. adam needs order of 0.001; others order of 0.05? sgdr can do 0.5
    if not hasattr(arg, 'patience'):
        warnings.warn('arg.train.patience not defined. Using default value 10')
        arg.patience = 10  # this is the number of epochs without improvement that the network waits untill determining it found its optimum
    if not hasattr(arg,'batch_size'):
        warnings.warn('arg.train.batch_size not defined. Using default value 128')
        arg.batch_size = 128  # number of datasets taken along per iteration
    if not hasattr(arg,'maxit'):
        warnings.warn('arg.train.maxit not defined. Using default value 500')
        arg.maxit = 500  # max iterations per epoch
    if not hasattr(arg,'split'):
        warnings.warn('arg.train.split not defined. Using default value 0.9')
        arg.split = 0.9  # split of test and validation data
    if not hasattr(arg,'load_nn'):
        warnings.warn('arg.train.load_nn not defined. Using default of False')
        arg.load_nn = False
    if not hasattr(arg,'loss_fun'):
        warnings.warn('arg.train.loss_fun not defined. Using default of ''rms''')
        arg.loss_fun = 'rms'  # what is the loss used for the model. rms is root mean square (linear regression-like); L1 is L1 normalisation (less focus on outliers)
    if not hasattr(arg,'skip_net'):
        warnings.warn('arg.train.skip_net not defined. Using default of False')
        arg.skip_net = False
    if not hasattr(arg,'use_cuda'):
        arg.use_cuda = torch.cuda.is_available()
    if not hasattr(arg, 'device'):
        arg.device = torch.device("cuda:0" if arg.use_cuda else "cpu")
    return arg


def checkarg_net_pars(arg):
    if not hasattr(arg,'dropout'):
        warnings.warn('arg.net_pars.dropout not defined. Using default value of 0.1')
        arg.dropout = 0.1  # 0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
    if not hasattr(arg,'batch_norm'):
        warnings.warn('arg.net_pars.batch_norm not defined. Using default of True')
        arg.batch_norm = True  # False/True turns on batch normalistion
    if not hasattr(arg,'parallel'):
        warnings.warn('arg.net_pars.parallel not defined. Using default of True')
        arg.parallel = True  # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
    if not hasattr(arg,'con'):
        warnings.warn('arg.net_pars.con not defined. Using default of ''sigmoid''')
        arg.con = 'sigmoid'  # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
    if not hasattr(arg,'cons_min'):
        warnings.warn('arg.net_pars.cons_min not defined. Using default values of  [-0.0001, -0.05, -0.05, 0.7]')
        arg.cons_min = [-0.0001, -0.05, -0.05, 0.7, -0.05, 0.06]  # Dt, Fp, Ds, S0 F2p, D2*
    if not hasattr(arg,'cons_max'):
        warnings.warn('arg.net_pars.cons_max not defined. Using default values of  [-0.0001, -0.05, -0.05, 0.7]')
        arg.cons_max = [0.005, 0.7, 0.3, 1.3, 0.3, 0.3]  # Dt, Fp, Ds, S0
    if not hasattr(arg,'fitS0'):
        warnings.warn('arg.net_pars.parallel not defined. Using default of False')
        arg.fitS0 = False  # indicates whether to fix S0 to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
    if not hasattr(arg,'depth'):
        warnings.warn('arg.net_pars.depth not defined. Using default value of 4')
        arg.depth = 4  # number of layers
    if not hasattr(arg, 'width'):
        warnings.warn('arg.net_pars.width not defined. Using default of number of b-values')
        arg.width = 0
    return arg


def checkarg_sim(arg):
    if not hasattr(arg, 'bvalues'):
        warnings.warn('arg.sim.bvalues not defined. Using default value of [0, 5, 10, 20, 30, 40, 60, 150, 300, 500, 700]')
        arg.bvalues = [0, 5, 10, 20, 30, 40, 60, 150, 300, 500, 700]
    if not hasattr(arg, 'repeats'):
        warnings.warn('arg.sim.repeats not defined. Using default value of 1')
        arg.repeats = 1  # this is the number of repeats for simulations
    if not hasattr(arg, 'rician'):
        warnings.warn('arg.sim.rician not defined. Using default of False')
        arg.rician = False
    if not hasattr(arg, 'SNR'):
        warnings.warn('arg.sim.SNR not defined. Using default of [20]')
        arg.SNR = [20]
    if not hasattr(arg, 'sims'):
        warnings.warn('arg.sim.sims not defined. Using default of 100000')
        arg.sims = 100000
    if not hasattr(arg, 'num_samples_eval'):
        warnings.warn('arg.sim.num_samples_eval not defined. Using default of 100000')
        arg.num_samples_eval = 100000
    if not hasattr(arg, 'range'):
        warnings.warn('arg.sim.range not defined. Using default of ([0.0005, 0.05, 0.01],[0.003, 0.4, 0.1])')
        arg.range = ([0.0005, 0.05, 0.01],
                  [0.003, 0.4, 0.1])
    return arg


def checkarg(arg):
    if not hasattr(arg, 'fig'):
    #     arg.fig = False
        warnings.warn('arg.fig not defined. Using default of False')
    if not hasattr(arg, 'save_name'):
        warnings.warn('arg.save_name not defined. Using default of ''default''')
        arg.save_name = 'default'
    if not hasattr(arg,'net_pars'):
        warnings.warn('arg no net_pars. Using default initialisation')
        arg.net_pars=net_pars()
    if not hasattr(arg, 'train_pars'):
        warnings.warn('arg no train_pars. Using default initialisation')
        arg.train_pars = train_pars()
    if not hasattr(arg, 'sim'):
        warnings.warn('arg no sim. Using default initialisation')
        arg.sim = sim()
    if not hasattr(arg, 'fit'):
        warnings.warn('arg no lsq. Using default initialisation')
        arg.fit = lsqfit()
    if not hasattr(arg, 'verbose'):
        arg.verbose = True

    arg.net_pars=checkarg_net_pars(arg.net_pars)
    arg.train_pars = checkarg_train_pars(arg.train_pars)
    arg.sim = checkarg_sim(arg.sim)
    arg.fit = fit.checkarg_lsq(arg.fit)
    return arg


class train_pars:
    def __init__(self):
        self.optim='adam' #these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
        self.lr = 0.0001 # this is the learning rate.
        self.patience= 10 # this is the number of epochs without improvement that the network waits untill determining it found its optimum
        self.batch_size= 128 # number of datasets taken along per iteration
        self.maxit = 500 # max iterations per epoch
        self.split = 0.9 # split of test and validation data
        self.load_nn= False # load the neural network instead of retraining
        self.loss_fun = 'rms' # what is the loss used for the model. rms is root mean square (linear regression-like); L1 is L1 normalisation (less focus on outliers)
        self.skip_net = False # skip the network training and evaluation
        self.scheduler = False # as discussed in the article, LR is important. This approach allows to reduce the LR itteratively when there is no improvement throughout an 5 consecutive epochs
        # use GPU if available
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.select_best = True
        self.grid_search= True

'''
class net_pars:
    def __init__(self):
        # select a network setting
        # the optimized network settings
        self.dropout = 0.1 #0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
        self.batch_norm = True # False/True turns on batch normalistion
        self.parallel = True # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
        self.con = 'sigmoid' # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
        #### only if sigmoid constraint is used!
        self.cons_min = [-0.0001, -0.05, -0.05, 0.7]  # Dt, Fp, Ds, S0 [0.0003, 0.001, 0.009, 0.99]#
        self.cons_max = [0.005, 0.7, 0.3, 1.3]  # Dt, Fp, Ds, S0 [0.01, 0.5,0.04, 1]#
        ####
        self.fitS0=True # indicates whether to fit S0 (True) or fix it to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
        self.depth = 4 # number of layers
        self.width = 0 # new option that determines network width. Putting to 0 makes it as wide as the number of b-values
'''

class lsqfit:
    def __init__(self):
        self.method = 'lsq' #seg, bayes or lsq
        self.do_fit = True # skip lsq fitting
        self.load_lsq = False # load the last results for lsq fit
        self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 in the least squares fit.
        self.jobs = 2 # number of parallel jobs. If set to 1, no parallel computing is used
        self.bounds = ([0, 0, 0.005, 0.7],[0.005, 0.7, 0.3, 1.3]) #Dt, Fp, Ds, S0

class sim:
    def __init__(self):
        self.bvalues = np.array([0, 2, 5, 10, 20, 30, 40, 60, 150, 300, 500, 700]) # array of b-values
        self.SNR = [15, 20, 30, 40, 50] # the SNRs to simulate at
        self.sims = 1000000 # number of simulations to run
        self.num_samples_eval = 100000 # number of simualtiosn te evaluate. This can be lower than the number run. Particularly to save time when fitting. More simulations help with generating sufficient data for the neural network
        self.repeats = 1 # this is the number of repeats for simulations
        self.rician = False # add rician noise to simulations; if false, gaussian noise is added instead
        self.range = ([0.0005, 0.05, 0.01],
                  [0.003, 0.55, 0.1])
   

def learn_supervised_IVIM(X_train, labels, bvalues ,arg,  sf, mode, work_dir, filename, net=None):
    """
    This program builds supervised IVIM-NET network and trains it.
    :param suprevised_data: 2D array of IVIM data we use for training. First axis are the voxels and second axis are the b-values
    :param labels: 
    :param bvalues: a 1D array with the b-values
    :param arg: an object with network design options, as explained in the publication check hyperparameters.py for
    options
    :param Dstar:  2D array of IVIM perfusion parameter use as training label
    :param D:  2D array of IVIM diffusion parameter use as training label
    :param Fp:  2D array of IVIM diffusion parameter use as training label
    :param net: an optional input pre-trained network with initialized weights for e.g. transfer learning or warm start
    :return net: returns a trained network
    """
    # ivim_combine = False
    torch.backends.cudnn.benchmark = True
    arg = checkarg(arg)
    n_bval = len(bvalues)
    ivim_combine = arg.train_pars.ivim_combine
    # print(f'\n \n ivim combine flag is {ivim_combine}\n \n ')

    ## normalise the signal to b=0 and remove data with nans
    if arg.key == 'phantom':
        S0 = np.mean(X_train[:, bvalues == 100], axis=1).astype('<f')
        X_train = X_train / S0[:, None]
        nan_idx =  isnan(np.mean(X_train, axis=1))
        X_train = np.delete(X_train, nan_idx , axis=0)
        labels = np.delete(labels, nan_idx , axis=0) # Dt, f, Dp
        print('phantom training')
    else:
        S0 = np.mean(X_train[:, bvalues == 0], axis=1).astype('<f')
        X_train = X_train / S0[:, None]
        nan_idx =  isnan(np.mean(X_train, axis=1))
        X_train = np.delete(X_train, nan_idx , axis=0)
        labels = np.delete(labels, nan_idx , axis=0) # Dt, f, Dp 
    # Limiting the percentile threshold
    if arg.key == 'phantom':
        b_less_300_idx = np.percentile(X_train[:, bvalues < 500], 95, axis=1) < 1.3 # X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
        b_greater_300_idx = np.percentile(X_train[:, bvalues > 500], 95, axis=1) < 1.2 #  X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
        b_greater_700_idx = np.percentile(X_train[:, bvalues > 1000], 95, axis=1) < 1 # X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
        thresh_idx = b_less_300_idx & b_greater_300_idx & b_greater_700_idx
    elif arg.key == 'clinic': #22 bval 
        print('DEBUG')
        b_less_25_idx = np.percentile(X_train[:, bvalues < 25], 95, axis=1) < 1.3 # X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
        b_greater_25_idx = np.percentile(X_train[:, bvalues > 25], 95, axis=1) < 1.2 #  X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
        b_greater_100_idx = np.percentile(X_train[:, bvalues > 100], 95, axis=1) < 1 # X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
        thresh_idx = b_less_25_idx & b_greater_25_idx & b_greater_100_idx         
    else: 
        b_less_50_idx = np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3 # X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
        b_greater_50_idx = np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2 #  X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
        b_greater_150_idx = np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1 # X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
        thresh_idx = b_less_50_idx & b_greater_50_idx & b_greater_150_idx

    if arg.key == 'sim_boot':
        size_boot= arg.sim.size_boot_low + arg.sim.size_boot_high + 1
        indices= np.array(range(n_bval))
        subsamples= np.zeros_like( X_train , dtype= 'uint8')
        np.random.seed(0)
        for i in range( X_train.shape[0]): 
            subsample_low = np.random.choice(indices[1: (n_bval//2)+1], arg.sim.size_boot_low, replace= False)
            subsample_high = np.random.choice(indices[(n_bval//2)+1:], arg.sim.size_boot_high, replace= False)
            subsample= np.concatenate((subsample_low, subsample_high))        
            subsample= np.insert(subsample, 0, indices[0])
            subsamples[i, subsample]= 1    
        mask = subsamples.astype(bool)
        X_train = X_train[mask].reshape(-1, size_boot)
        bvalues_new= np.broadcast_to(bvalues,( X_train.shape[0], n_bval))
        bvalues_new = bvalues_new[mask].reshape(-1, size_boot)
        X_train= np.concatenate((X_train,bvalues_new), axis=1) 
    elif arg.key == 'sim_bvalue':
        size = arg.sim.size_low + arg.sim.size_high + 1
        indices= np.array(range(n_bval))
        subsamples= np.zeros_like( X_train , dtype= 'uint8')
        np.random.seed(0)
        for i in range( X_train.shape[0]): 
            subsample_low = np.random.choice(indices[1:10], arg.sim.size_low, replace= False)
            subsample_high = np.random.choice(indices[10:], arg.sim.size_high, replace= False)
            subsample= np.concatenate((subsample_low, subsample_high))        
            subsample= np.insert(subsample, 0, indices[0])
            subsamples[i, subsample]= 1    
        mask = subsamples.astype(bool)
        X_train = X_train[mask].reshape(-1, size)
        bvalues_new= np.broadcast_to(bvalues,( X_train.shape[0], n_bval))
        bvalues_new = bvalues_new[mask].reshape(-1, size)
        X_train= np.concatenate((X_train,bvalues_new), axis=1)
    else:
        print('standard training') 

    suprevised_data = np.append(X_train[thresh_idx, ], labels[thresh_idx, ] , axis = 1)    # combine the labels and the X_train data for supervised learning

    # initialising the network of choice using the input argument arg
    if net is None:
        bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)
        net = Net(bvalues,arg, arg.net_pars, supervised = True).to(arg.train_pars.device)
    else:
        # if a network was used as input parameter, work with that network instead (transfer learning/warm start).
        net.to(arg.train_pars.device)

    # defining the loss function; not explored in the publication
    criterion_Dt = nn.MSELoss(reduction='mean').to(arg.train_pars.device)
    criterion_Fp = nn.MSELoss(reduction='mean').to(arg.train_pars.device)
    criterion_Dp = nn.MSELoss(reduction='mean').to(arg.train_pars.device)
    if ivim_combine:
        criterion_ivim = nn.MSELoss(reduction='mean').to(arg.train_pars.device)

    # splitting data into learning and validation set; subsequently initialising the Dataloaders
    split = int(np.floor(len(X_train) * arg.train_pars.split))
    train_set, val_set = torch.utils.data.random_split(torch.from_numpy(suprevised_data.astype(np.float32)),
                                                       [split, len(suprevised_data) - split])
    
    # train loader loads the trianing data. We want to shuffle to make sure data order is modified each epoch and different data is selected each epoch.
    trainloader = utils.DataLoader(train_set,
                                   batch_size=arg.train_pars.batch_size,
                                   shuffle=True,
                                   drop_last=True)
    # validation data is loaded here. By not shuffling, we make sure the same data is loaded for validation every time. We can use substantially more data per batch as we are not training.
    inferloader = utils.DataLoader(val_set,
                                   batch_size=32 * arg.train_pars.batch_size,
                                   shuffle=False,
                                   drop_last=True)

    # defining the number of training and validation batches for normalisation later
    totalit = np.min([arg.train_pars.maxit, np.floor(split // arg.train_pars.batch_size)])
    batch_norm2 = np.floor(len(val_set) // (32 * arg.train_pars.batch_size))

    # defining optimiser
    if arg.train_pars.scheduler:
        optimizer, scheduler = load_optimizer(net, arg)
    else:
        optimizer = load_optimizer(net, arg)

    # Initialising parameters
    best = 1e16
    num_bad_epochs = 0
    loss_train = []
    loss_val = []
    acum_loss_Dp = []
    acum_loss_Dt = []
    acum_loss_Fp = []
    val_loss_Dp = []
    val_loss_Dt = []
    val_loss_Fp = []
    if ivim_combine:
        acum_loss_recon = []
        val_loss_recon = []
    prev_lr = 0
    # get_ipython().run_line_magic('matplotlib', 'inline')
    final_model = copy.deepcopy(net.state_dict())
    if arg.key == 'sim_boot': 
        writer = SummaryWriter(f'{work_dir}/loss_{arg.sim.size_boot_low}_{arg.sim.size_boot_high}_{arg.sim.seed}_boot') 
    elif arg.key== 'sim_bvalue': 
        writer = SummaryWriter(f'{work_dir}/loss_{arg.sim.seed}_bvalue')    
    else:
        writer = SummaryWriter(f'{work_dir}/loss_{arg.sim.seed}_super_ivim_dc')        
    ## Train
    for epoch in range(1000):
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        # initialising and resetting parameters
        net.train()
        running_loss_train = 0.
        running_loss_Dp = 0.
        running_loss_Dt = 0.
        running_loss_Fp = 0.

        running_loss_val = 0.
        running_loss_Dp_val = 0.
        running_loss_Dt_val = 0.
        running_loss_Fp_val = 0.
        
        if ivim_combine:
            running_loss_recon = 0.
            running_loss_recon_val =0.
        #losstotcon = 0.
        maxloss = 0.
        for i, suprevised_batch in enumerate(tqdm(trainloader, position=0, leave=True, total=totalit), 0):
            if i > totalit:
                # have a maximum number of batches per epoch to ensure regular updates of whether we are improving
                break
            # zero the parameter gradients
            optimizer.zero_grad()
            # put batch on GPU if pressent
            suprevised_batch = suprevised_batch.to(arg.train_pars.device)
            # forward + backward + optimize
            if arg.key == 'sim_boot':
                X_batch = suprevised_batch[: ,:2*size_boot]
            elif arg.key== 'sim_bvalue':
                X_batch = suprevised_batch[: ,:2*size]
            else:
                X_batch = suprevised_batch[: ,:n_bval]
            Dp_batch = suprevised_batch[: ,-1] 
            Fp_batch = suprevised_batch[: ,-2] 
            Dt_batch = suprevised_batch[: ,-3] 
            # farward path
            X_pred, Dt_pred, Fp_pred, Dp_pred, S0pred= net(X_batch)
            # removing nans and too high/low predictions to prevent overshooting
            X_pred[isnan(X_pred)] = 0
            X_pred[X_pred < 0] = 0
            X_pred[X_pred > 3] = 3
            
            # determine loss for batch; note that the loss is determined by the difference between the predicted signal and the actual signal. The loss does not look at Dt, Dp or Fp.
            Dp_loss = criterion_Dp(Dp_pred, Dp_batch.unsqueeze(1)).to(arg.train_pars.device)
            Fp_loss = criterion_Fp(Fp_pred, Fp_batch.unsqueeze(1)).to(arg.train_pars.device)
            Dt_loss = criterion_Dt(Dt_pred, Dt_batch.unsqueeze(1)).to(arg.train_pars.device)
            #S0_loss = criterion_Dt(S0_pred, S0_batch)
            if ivim_combine:
                if arg.key == 'sim_boot':
                    ivim_loss = criterion_ivim(X_pred, X_batch[:, :size_boot])
                elif arg.key == 'sim_bvalue':
                    ivim_loss = criterion_ivim(X_pred, X_batch[:, :size])
                else:
                    ivim_loss = criterion_ivim(X_pred, X_batch)
            #loss coefficiants
            Dp_coeff = (arg.loss_coef_Dp).to(arg.train_pars.device)
            Dt_coeff = (arg.loss_coef_Dt).to(arg.train_pars.device)
            Fp_coeff = (arg.loss_coef_Fp).to(arg.train_pars.device)
            if ivim_combine:
                ivim_coeff = torch.FloatTensor([arg.loss_coef_ivim]).to(arg.train_pars.device)
                loss = Dp_coeff*Dp_loss + Dt_coeff*Dt_loss + Fp_coeff*Fp_loss + ivim_coeff*ivim_loss
            else:
                loss = Dp_coeff*Dp_loss + Dt_coeff*Dt_loss + Fp_coeff*Fp_loss
            
            # updating network
            loss.backward()
            optimizer.step()
            #parameters loss
            running_loss_Dp += Dp_loss.item()
            running_loss_Dt += Dt_loss.item()
            running_loss_Fp += Fp_loss.item()
            if ivim_combine:
                running_loss_recon += ivim_loss.item()
            # total loss and determine max loss over all batches
            running_loss_train += loss.item()
            if loss.item() > maxloss:
                maxloss = loss.item()
        # show some figures if desired, to show whether there is a correlation between Dp and f
        if arg.fig:
            plt.figure()
            plt.clf()
            plt.plot(Dp_pred.tolist(), Fp_pred.tolist(), 'rx', markersize=5)
            plt.ion()
            plt.show()
        # after training, do validation in unseen data without updating gradients
        print('\n validation \n')
        net.eval()
        # validation is always done over all validation data
        for i, suprevised_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            optimizer.zero_grad()
            suprevised_batch = suprevised_batch.to(arg.train_pars.device)
            if arg.key == 'sim_boot':
                X_batch = suprevised_batch[: ,:2*size_boot]
            elif arg.key== 'sim_bvalue': 
                X_batch = suprevised_batch[: ,:2*size]   
            else:
                X_batch = suprevised_batch[:, :n_bval]      
            ## forward + backward + optimize
            Dp_batch = suprevised_batch[: ,-1]
            Fp_batch = suprevised_batch[: ,-2]
            Dt_batch = suprevised_batch[: ,-3]
            
            X_pred, Dt_pred, Fp_pred, Dp_pred, S0pred = net(X_batch)
      
            X_pred[isnan(X_pred)] = 0
            X_pred[X_pred < 0] = 0
            X_pred[X_pred > 3] = 3
            # validation loss
            Dp_loss_val = criterion_Dp(Dp_pred, Dp_batch.unsqueeze(1)).to(arg.train_pars.device)
            Fp_loss_val = criterion_Fp(Fp_pred, Fp_batch.unsqueeze(1)).to(arg.train_pars.device)
            Dt_loss_val = criterion_Dt(Dt_pred, Dt_batch.unsqueeze(1)).to(arg.train_pars.device)
            if ivim_combine:
                if arg.key == 'sim_boot':
                    ivim_loss_val = criterion_ivim(X_pred, X_batch[:, :size_boot]) 
                elif arg.key== 'sim_bvalue': 
                    ivim_loss_val = criterion_ivim(X_pred, X_batch[:, :size]) 
                else:
                    ivim_loss_val = criterion_ivim(X_pred, X_batch)
                loss = Dp_coeff*Dp_loss_val + Dt_coeff*Dt_loss_val + Fp_coeff*Fp_loss_val + ivim_coeff*ivim_loss_val
            else:
                #S0 coefficiant is not covered
                loss = Dp_coeff*Dp_loss_val + Dt_coeff*Dt_loss_val + Fp_coeff*Fp_loss_val #add the wieghts as configurable value
            running_loss_val += loss.item()
            running_loss_Dp_val += Dp_loss_val.item()
            running_loss_Dt_val += Dt_loss_val.item()
            running_loss_Fp_val += Fp_loss_val.item()
            if ivim_combine:
                running_loss_recon_val += ivim_loss_val.item()
        # scale losses
        running_loss_train = running_loss_train / totalit
        running_loss_Dp = running_loss_Dp / totalit
        running_loss_Dt = running_loss_Dt / totalit
        running_loss_Fp = running_loss_Fp / totalit
        running_loss_val = running_loss_val / batch_norm2
        running_loss_Dp_val = running_loss_Dp / batch_norm2
        running_loss_Dt_val = running_loss_Dt / batch_norm2
        running_loss_Fp_val = running_loss_Fp / batch_norm2
        
        if ivim_combine:
                running_loss_recon += running_loss_recon / totalit
                running_loss_recon_val += running_loss_recon_val / batch_norm2
        # save loss history for plot
        loss_train.append(running_loss_train)
        loss_val.append(running_loss_val)
        acum_loss_Dp.append(running_loss_Dp)
        acum_loss_Dt.append(running_loss_Dt)
        acum_loss_Fp.append(running_loss_Fp)
        val_loss_Dp.append(running_loss_Dp_val)
        val_loss_Dt.append(running_loss_Dt_val)
        val_loss_Fp.append(running_loss_Fp_val)
        
        if ivim_combine:
            acum_loss_recon.append(running_loss_recon)
            val_loss_recon.append(running_loss_recon_val)

            if arg.fig:
                loss_plot_supervised(acum_loss_Dp, acum_loss_Dt, acum_loss_Fp, loss_train, loss_val, 
                                val_loss_Dp, val_loss_Dt, val_loss_Fp, acum_loss_recon, val_loss_recon)
        else:
            if arg.fig:
                loss_plot_supervised(acum_loss_Dp, acum_loss_Dt, acum_loss_Fp, loss_train, loss_val, 
                                val_loss_Dp, val_loss_Dt, val_loss_Fp)
        # as discussed in the article, LR is important. This approach allows to reduce the LR if we think it is too
        # high, and return to the network state before it went poorly       
        if arg.train_pars.scheduler:
            scheduler.step(running_loss_val)
            if optimizer.param_groups[0]['lr'] < prev_lr:
                net.load_state_dict(final_model)
            prev_lr = optimizer.param_groups[0]['lr']   
        scalar_dict = {'Loss/Training': running_loss_train, 'Loss/Validation': running_loss_val}
        for tag, value in scalar_dict.items():
            writer.add_scalar(tag, value, epoch)      
        # print stuff
        print("\n IVIM Loss: {loss}, IVIM validation_loss: {val_loss}, lr: {lr}".format(loss=running_loss_train,
                                                                             val_loss=running_loss_val,
                                                                             lr=optimizer.param_groups[0]['lr']))
        # print(f'\n D* Loss: {running_loss_Dp}')
        # print(f'\n D Loss: {running_loss_Dt}')
        # print(f'\n Fp Loss: {running_loss_Fp}')
        # early stopping criteria
        if running_loss_val < best: # TODO change it to total loss
            print("\n############### Saving good model ###############################")
            final_model = copy.deepcopy(net.state_dict())
            best = running_loss_val
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == arg.train_pars.patience:
                print("\nDone, best val loss: {}".format(best))
                break
        # plot loss and plot 4 fitted curves
        if epoch > 0:
            # plot progress and intermediate results (if enabled)
            if arg.key == 'sim_boot' or arg.key == 'sim_bvalue':
                pass
            else: 
                plot_progress(X_batch[: ,:n_bval].cpu(), X_pred.cpu(), bvalues.cpu(), loss_train, loss_val, arg)   
            
    print("Done")
    # save final fits
    if arg.fig:
        if not os.path.isdir('plots'):
            os.makedirs('plots')
        plt.figure(1)
        plt.gcf()
        plt.savefig('plots/fig_fit.png')
        plt.figure(2)
        plt.gcf()
        plt.savefig('plots/fig_train.png')
        plt.close('all')
    # Restore best model
    if arg.train_pars.select_best:
        net.load_state_dict(final_model)
    del trainloader
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
        
    #save the model (Elad addition)
    save_state_model = True
    if save_state_model:
        final_weights = net.state_dict()
        torch.save(final_weights, f'{work_dir}/{filename}.pt')#save the model trained_model.state_dict()
    writer.flush()    
    return net

def predict_supervised_IVIM(X_train, labels, bvalues, net, arg):
    """
    This program takes a trained network and predicts the IVIM parameters from it.
    :param data: 2D array of IVIM data we want to predict the IVIM parameters from. First axis are the voxels and second axis are the b-values
    :param bvalues: a 1D array with the b-values
    :param net: the trained IVIM-NET network
    :param arg: an object with network design options, as explained in the publication check hyperparameters.py for
    options
    :return param: returns the predicted parameters
    """
    arg = checkarg(arg)
    n_bval = len(bvalues)
    #lend = len(X_train)
    ## normalise the signal to b=0 and remove data with nans
    ## normalise the signal to b=0 and remove data with nans
    if arg.key == 'phantom':
        S0 = np.mean(X_train[:, bvalues == 100], axis=1).astype('<f')
        X_train = X_train / S0[:, None]
        nan_idx =  isnan(np.mean(X_train, axis=1))
        X_train = np.delete(X_train, nan_idx , axis=0)
        labels = np.delete(labels, nan_idx , axis=0) # Dt, f, Dp
        print('phantom training')
    else:
        S0 = np.mean(X_train[:, bvalues == 0], axis=1).astype('<f')
        X_train = X_train / S0[:, None]
        nan_idx =  isnan(np.mean(X_train, axis=1))
        X_train = np.delete(X_train, nan_idx , axis=0)
        labels = np.delete(labels, nan_idx , axis=0) # Dt, f, Dp
  
    # Limiting the percentile threshold
    if arg.key == 'phantom':
        b_less_300_idx = np.percentile(X_train[:, bvalues < 500], 95, axis=1) < 1.3 # X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
        b_greater_300_idx = np.percentile(X_train[:, bvalues > 500], 95, axis=1) < 1.2 #  X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
        b_greater_700_idx = np.percentile(X_train[:, bvalues > 1000], 95, axis=1) < 1 # X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
        thresh_idx = b_less_300_idx & b_greater_300_idx & b_greater_700_idx
    else: 
        b_less_50_idx = np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3 # X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
        b_greater_50_idx = np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2 #  X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
        b_greater_150_idx = np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1 # X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
        thresh_idx = b_less_50_idx & b_greater_50_idx & b_greater_150_idx
    
    if arg.key == 'sim_boot':
        if thresh_idx[0]==False:
            Dptrue= 0
            Dttrue= 0
            Fptrue= 0
            S0true= 0

            return [Dttrue, Fptrue, Dptrue, S0true]
        else:
            size_boot= arg.sim.size_boot_low + arg.sim.size_boot_high + 1
            indices= np.array(range(n_bval))
            subsamples= np.zeros( (arg.sim.n_subsamples, n_bval), dtype= 'uint8')
            thresh_idx= np.broadcast_to(thresh_idx,( arg.sim.n_subsamples)) 
            labels= np.broadcast_to(labels,( arg.sim.n_subsamples, 3))
            np.random.seed(100)
            X_train= np.broadcast_to( X_train, (arg.sim.n_subsamples, n_bval)) 
            for i in range(arg.sim.n_subsamples): 
                subsample_low = np.random.choice(indices[1: (n_bval//2)+1], arg.sim.size_boot_low, replace= False)
                subsample_high = np.random.choice(indices[(n_bval//2)+1:], arg.sim.size_boot_high, replace= False)
                subsample= np.concatenate((subsample_low, subsample_high))  
                subsample= np.insert(subsample, 0, indices[0])
                subsamples[i, subsample]= 1    
            mask = subsamples.astype(bool)
            X_train = X_train[mask].reshape(-1, size_boot)
            bvalues_new= np.broadcast_to(bvalues,( arg.sim.n_subsamples, n_bval))
            bvalues_new = bvalues_new[mask].reshape(-1, size_boot)
            X_train= np.concatenate((X_train,bvalues_new), axis=1)
    if arg.key== 'sim_bvalue':
        bvalues_new= np.broadcast_to(bvalues, (X_train.shape[0], len(bvalues)))
        X_train= np.concatenate((X_train, bvalues_new), axis=1)      
    lend = len(X_train)
    suprevised_data = np.append(X_train[thresh_idx, ], labels[thresh_idx, ], axis = 1)    # combine the labels and the X_train data for supervised learning

     # tell net it is used for evaluation
    net.eval()
    # initialise parameters and data
    Dp = np.array([])
    Dt = np.array([])
    Fp = np.array([])
    S0 = np.array([])
    # initialise dataloader. Batch size can be way larger as we are still training.
    inferloader = utils.DataLoader(torch.from_numpy(suprevised_data.astype(np.float32)),
                                batch_size=1000,
                                shuffle=False,
                                drop_last=False)
    # start predicting
    with torch.no_grad():
        for i, suprevised_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            
            suprevised_batch = suprevised_batch.to(arg.train_pars.device)
            if arg.key == 'sim_boot':
                X_batch = suprevised_batch[: ,:2*size_boot]
            elif arg.key== 'sim_bvalue':
                X_batch = suprevised_batch[: ,:2*n_bval]        
            else:
                X_batch = suprevised_batch[: ,:n_bval] 
            
            # here the signal is predicted. Note that we now are interested in the parameters and no longer in the predicted signal decay.
            _, Dtt, Fpt, Dpt, S0t = net(X_batch, train_mode=False)
            # Quick and dirty solution to deal with networks not predicting S0
            try:
                S0 = np.append(S0, (S0t.cpu()).numpy())
            except:
                S0 = np.append(S0, S0t)
            Dp = np.append(Dp, (Dpt.cpu()).numpy())
            Dt = np.append(Dt, (Dtt.cpu()).numpy())
            Fp = np.append(Fp, (Fpt.cpu()).numpy())
    # The 'abs' and 'none' constraint networks have no way of figuring out what is D and D* a-priori. However, they do
    # tend to pick one output parameter for D or D* consistently within the network. If the network has swapped D and
    # D*, we swap them back here.
    if np.mean(Dp) < np.mean(Dt):
        Dp22 = copy.deepcopy(Dt)
        Dt = Dp
        Dp = Dp22
        Fp = 1 - Fp
    
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    #print(lend)
    #print(thresh_idx)
        
    Dptrue = np.zeros(lend)
    Dttrue = np.zeros(lend)
    Fptrue = np.zeros(lend)
    S0true = np.zeros(lend)
    Dptrue[thresh_idx] = Dp
    Dttrue[thresh_idx] = Dt
    Fptrue[thresh_idx] = Fp
    S0true[thresh_idx] = S0
    
    if arg.key== 'sim_boot': 
        Dptrue= np.mean(Dptrue)
        Dttrue= np.mean(Dttrue) 
        Fptrue= np.mean(Fptrue)
        S0true= np.mean(S0true) 
                
    
    return [Dttrue, Fptrue, Dptrue, S0true]


def infer_supervised_IVIM(X_infer, labels, bvalues, ivim_path, arg): # fits only for super-ivim-dc

    arg = checkarg(arg)
    n_bval = len(bvalues)
    n_samples = len(X_infer)
    print(f'The number of samples are: {n_samples}')  
    
    # The b-values that get into the model need to be torch Tensor type.
    bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)
    X_infer = torch.from_numpy(X_infer).to(arg.train_pars.device)

    #load the pretrained network
    ivim_model = Net(bvalues,arg, arg.net_pars).to(arg.train_pars.device)
    ivim_model.load_state_dict(torch.load(ivim_path))
    ivim_model.eval()
    
    ## normalise the signal to b=0 and remove data with nans
    if arg.key == 'phantom':
        S0 = np.mean(X_infer[:, bvalues == 100], axis=1).astype('<f')
        X_infer = X_infer / S0[:, None]
        nan_idx =  isnan(np.mean(X_infer, axis=1))
        X_infer = np.delete(X_infer, nan_idx , axis=0)
        labels = np.delete(labels, nan_idx , axis=0) # Dt, f, Dp
        print('phantom training')
    else:
        # print(torch.mean(X_infer[: , bvalues == 0], axis=1))
        S0 = torch.mean(X_infer[: , bvalues == 0], axis=1)
        X_infer = X_infer / S0[:, None]
        nan_idx =  isnan(torch.mean(X_infer, axis=1))

        X_infer = X_infer.cpu().numpy()
        nan_idx = nan_idx.cpu().numpy()
        X_infer = np.delete(X_infer, nan_idx , axis=0)
        labels = np.delete(labels, nan_idx , axis=0) # Dt, f, Dp
    
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
    
    suprevised_data = np.append(X_infer[thresh_idx, ], labels[thresh_idx, ], axis = 1)    # combine the labels and the X_train data for supervised learning

    # initialise parameters and data
    Dp_infer = np.array([])
    Dt_infer = np.array([])
    Fp_infer = np.array([])
    S0_infer = np.array([])
    Dp_orig = np.array([])
    Dt_orig = np.array([])
    Fp_orig = np.array([])
    S0_orig = np.array([1])
    # initialise dataloader. Batch size can be way larger as we are still training.
    inferloader = utils.DataLoader(torch.from_numpy(suprevised_data.astype(np.float32)),
                                   batch_size=1, # previously was 2056,
                                   shuffle=False,
                                   drop_last=False)

    # start predicting
    with torch.no_grad():
        for i, suprevised_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            
            suprevised_batch = suprevised_batch.to(arg.train_pars.device)
            X_batch = suprevised_batch[: ,:n_bval] 

            Dp_batch = suprevised_batch[: ,-1] 
            Fp_batch = suprevised_batch[: ,-2] 
            Dt_batch = suprevised_batch[: ,-3]
            
            Dp_orig = np.append(Dp_orig, (Dp_batch.cpu()).numpy())
            Dt_orig = np.append(Dt_orig, (Dt_batch.cpu()).numpy())
            Fp_orig = np.append(Fp_orig, (Fp_batch.cpu()).numpy())
            
            # here the signal is predicted. Note that we now are interested in the parameters and no longer in the predicted signal decay.
            _, Dtt, Fpt, Dpt, S0t = ivim_model(X_batch)
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
        
    #print(f'Dp_orig {Dp_orig} /n Dp_infer {Dp_infer} ')
    #print(f'Dp_orig-Dp_infer {Dp_orig-Dp_infer}')
    
    e_calc_type = 'NRSE'
    if e_calc_type == 'NRSE':
        Dp_norm_error = np.sqrt(np.square(Dp_orig-Dp_infer))/Dp_orig
        Dt_norm_error = np.sqrt(np.square(Dt_orig-Dt_infer))/Dt_orig
        Fp_norm_error = np.sqrt(np.square(Fp_orig-Fp_infer))/Fp_orig
        S0_norm_error = np.sqrt(np.square(S0_orig-S0_infer))
    # else: # NRMSE
    #     Dp_norm_error = np.sqrt(np.square(Dp_orig-Dp_infer)/Dp_orig)
    #     Dt_norm_error = np.sqrt(np.square(Dt_orig-Dt_infer)/Dt_orig)
    #     Fp_norm_error = np.sqrt(np.square(Fp_orig-Fp_infer)/Fp_orig)
    #     S0_norm_error = np.sqrt(np.square(S0_orig-S0_infer))
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    
    return [Dp_norm_error, Dt_norm_error, Fp_norm_error, S0_norm_error]

def infer_clinical_supervised_IVIM(X_infer, bvalues, ivim_path, arg): # fits only for super-ivim-dc

    arg = checkarg(arg)
      
    # The b-values that get into the model need to be torch Tensor type.
    bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)
    #load the pretrained network
    ivim_model = Net(bvalues,arg, arg.net_pars).to(arg.train_pars.device)
    ivim_model.load_state_dict(torch.load(ivim_path, map_location=torch.device('cpu')))
    ivim_model.eval()
    
    ## normalise the signal to b=0 and remove data with nans
    
    
    if (X_infer.ndim < 2):
         if arg.key == 'phantom':
            selsb = np.array(bvalues) == 100
            S0 = np.nanmean(X_infer[:, selsb], axis=1)
            S0[S0 != S0] = 0
            S0 = np.squeeze(S0)
            valid_id = (S0 > (0.5 * np.median(S0[S0 > 0]))) # Boolean parameter with indication for True/False prediction value.
            datatot = X_infer[valid_id, :]
         else:
            selsb = (bvalues == 0) #(np.array(bvalues == 0))
            #noam
            selsb = selsb.cpu().numpy()
            S0 = np.nanmean(X_infer[selsb])
            #S0[S0 != S0] = 0
            S0 = np.squeeze(S0)
            valid_id = (S0 > (0.5 * np.median(S0[S0 > 0]))) # Boolean parameter with indication for True/False prediction value.
            datatot = X_infer[valid_id] 
            
            #TODO remove mylist and sels 
        
            mylist = isnan(np.mean(X_infer))
            #sels = [not i for i in mylist]
            #print(f'sels size is {len(sels)}')
            
            # normalise data
            S0 = (datatot[0]).astype('<f')
            datatot = datatot / S0
            print('Clinical patient data loaded\n')
            
    else:
        if arg.key == 'phantom':
            selsb = np.array(bvalues) == 100
            S0 = np.nanmean(X_infer[:, selsb], axis=1)
            S0[S0 != S0] = 0
            S0 = np.squeeze(S0)
            valid_id = (S0 > (0.5 * np.median(S0[S0 > 0]))) # Boolean parameter with indication for True/False prediction value.
            datatot = X_infer[valid_id, :]
        else:
            selsb = (bvalues == 0) #(np.array(bvalues == 0))
            #noam
            selsb = selsb.cpu().numpy()
            S0 = np.nanmean(X_infer[:, selsb], axis=1)
            S0[S0 != S0] = 0
            S0 = np.squeeze(S0)
            valid_id = (S0 > (0.5 * np.median(S0[S0 > 0]))) # Boolean parameter with indication for True/False prediction value.
            datatot = X_infer[valid_id, :]
            #TODO remove mylist and sels 
        
            mylist = isnan(np.mean(X_infer, axis=1))
            sels = [not i for i in mylist]
            print(f'sels size is {len(sels)}')
            
            # normalise data
            S0 = np.nanmean(datatot[:, selsb], axis=1).astype('<f')
            datatot = datatot / S0[:, None]
            print('Clinical patient data loaded\n')
            
    # Limiting the percentile threshold

    # initialise parameters and data
    Dp_infer = np.array([])
    Dt_infer = np.array([])
    Fp_infer = np.array([])
    S0_infer = np.array([])
    recon_error = np.array([])

    # initialise dataloader. Batch size can be way larger as we are still training.
    inferloader = utils.DataLoader(torch.from_numpy(datatot.astype(np.float32)),
                                   batch_size=1,
                                   shuffle=False,
                                   drop_last=False)

    # start predicting
    with torch.no_grad():
        for i, clinical_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            
            clinical_batch = clinical_batch.to(arg.train_pars.device)
            print(type(clinical_batch))
            # here the signal is predicted. Note that we now are interested in the parameters and no longer in the predicted signal decay.
            clinical_infer, Dtt, Fpt, Dpt, S0t = ivim_model(clinical_batch)
            # Quick and dirty solution to deal with networks not predicting S0
            try:
                S0 = np.append(S0, (S0t.cpu()).numpy())
            except:
                S0 = np.append(S0, S0t)
            
            Dp_infer = np.append(Dp_infer, (Dpt.cpu()).numpy())
            Dt_infer = np.append(Dt_infer, (Dtt.cpu()).numpy())
            Fp_infer = np.append(Fp_infer, (Fpt.cpu()).numpy())
            S0_infer = np.append(S0_infer, (S0t.cpu()).numpy())
            #this is an absulote error
            SR_error = (torch.sqrt(torch.square(clinical_infer-clinical_batch))/clinical_batch)
            recon_error = np.append(recon_error, (SR_error.cpu()).numpy())
            # Error in precent -> devide the absolute error by the original value
  
    # swithc between Dt & Dp if the preduction is wrong
    if np.mean(Dp_infer) < np.mean(Dt_infer):
        Dp22 = copy.deepcopy(Dt_infer)
        Dt_infer = Dp_infer
        Dp_infer= Dp22
        Fp_infer = 1 - Fp_infer
    
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    
    # here we correct for the data that initially was removed as it did not have IVIM behaviour, by returning zero
    # estimates
    if(X_infer.ndim >= 2):
        Dp_out = np.zeros(len(valid_id))
        Dt_out = np.zeros(len(valid_id))
        Fp_out = np.zeros(len(valid_id))
        S0_out = np.zeros(len(valid_id))
        Dp_out[valid_id] = Dp_infer
        Dt_out[valid_id] = Dt_infer
        Fp_out[valid_id] = Fp_infer
        S0_out[valid_id] = S0_infer
    else:
        Dp_out = Dp_infer
        Dt_out = Dt_infer
        Fp_out = Fp_infer
        S0_out = S0_infer
    
    #return [recon_error, Dp_infer, Dt_infer, Fp_infer, S0_infer]
    return [recon_error, Dp_out, Dt_out, Fp_out, S0_out]

def ivimN_noS0_lsq(bvalues, Dt, Fp, Dp):
    # IVIM function in which we try to have equal variance in the different IVIM parameters and S0=1
    return (Fp / 10 * np.exp(-bvalues * Dp / 10) + (1 - Fp / 10) * np.exp(-bvalues * Dt / 1000))


def ivimN_lsq(bvalues, Dt, Fp, Dp, S0):
    # IVIM function in which we try to have equal variance in the different IVIM parameters; equal variance helps with certain fitting algorithms
    return S0 * (Fp / 10 * np.exp(-bvalues * Dp / 10) + (1 - Fp / 10) * np.exp(-bvalues * Dt / 1000))


def infer_leastsquares_IVIM(X_infer, labels, bvalues, arg):

    arg = checkarg(arg)
    n_samples = len(X_infer)
    print(f'The number of samples are: {n_samples}')  
       
    ## normalise the signal to b=0 and remove data with nans
    if arg.key == 'phantom':
        S0 = np.mean(X_infer[:, bvalues == 100], axis=1).astype('<f')
        X_infer = X_infer / S0[:, None]
        nan_idx =  isnan(np.mean(X_infer, axis=1))
        X_infer = np.delete(X_infer, nan_idx , axis=0)
        labels = np.delete(labels, nan_idx , axis=0) # Dt, f, Dp
        print('phantom lsq')
    else:
        S0 = np.mean(X_infer[:, bvalues == 0], axis=1).astype('<f')
        X_infer = X_infer / S0[:, None]
        nan_idx =  isnan(np.mean(X_infer, axis=1))
        X_infer = np.delete(X_infer, nan_idx , axis=0)
        labels = np.delete(labels, nan_idx , axis=0) # Dt, f, Dp
    
    # Limiting the percentile threshold
    if arg.key == 'phantom':
        b_less_500_idx = np.percentile(X_infer[:, bvalues < 500], 95, axis=1) < 1.3 # X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
        b_greater_500_idx = np.percentile(X_infer[:, bvalues > 500], 95, axis=1) < 1.2 #  X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
        b_greater_1000_idx = np.percentile(X_infer[:, bvalues > 1000], 95, axis=1) < 1 # X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
        thresh_idx = b_less_500_idx & b_greater_500_idx & b_greater_1000_idx
    else: 
        b_less_50_idx = np.percentile(X_infer[:, bvalues < 50], 95, axis=1) < 1.3 # X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
        b_greater_50_idx = np.percentile(X_infer[:, bvalues > 50], 95, axis=1) < 1.2 #  X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
        b_greater_150_idx = np.percentile(X_infer[:, bvalues > 150], 95, axis=1) < 1 # X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
        thresh_idx = b_less_50_idx & b_greater_50_idx & b_greater_150_idx
    
    suprevised_data = np.append(X_infer[thresh_idx, ], labels[thresh_idx, ], axis = 1)    # combine the labels and the X_train data for supervised learning

    # initialise parameters and data
    Dp_lsq_NRMSE = np.array([])
    Dt_lsq_NRMSE = np.array([])
    Fp_lsq_NRMSE = np.array([])
    S0_lsq_NRMSE = np.array([])
    # bounds are rescaled such that each parameter changes at roughly the same rate to help fitting.
    bound_flag = False
    if bound_flag:
        bounds=([0, 0, 0.005, 0.7],[0.005, 0.7, 0.3, 1.3])
        # bounds = ([bounds[0][0] * 1000, bounds[0][1] * 10, bounds[0][2] * 10],
        #           [bounds[1][0] * 1000, bounds[1][1] * 10, bounds[1][2] * 10])
        # bounds are rescaled such that each parameter changes at roughly the same rate to help fitting.
        bounds = ([bounds[0][0] * 1000, bounds[0][1] * 10, bounds[0][2] * 10, bounds[0][3]],
                  [bounds[1][0] * 1000, bounds[1][1] * 10, bounds[1][2] * 10, bounds[1][3]])
        # calculate least square
        for dw_data, dw_label in zip(X_infer[thresh_idx, ], labels[thresh_idx, ]):
    
            #params, _ = curve_fit(ivimN_noS0_lsq, bvalues, dw_data, p0=[1, 1, 0.1], bounds=bounds)
            params, _ = curve_fit(ivimN_lsq, bvalues, dw_data, p0=[1, 1, 0.1, 1], bounds=bounds)
            S0_lsq = params[3]
            # correct for the rescaling of parameters
            Dt_lsq, Fp_lsq, Dp_lsq = params[0] / 1000, params[1] / 10, params[2] / 10
        else:
            params, _ = curve_fit(ivimN_lsq, bvalues, dw_data, p0=[1, 1, 0.1, 1])
            S0_lsq = params[3]
            # correct for the rescaling of parameters
            Dt_lsq, Fp_lsq, Dp_lsq = params[0] , params[1] , params[2]
        
        Dt_orig = dw_label[0] 
        Fp_orig = dw_label[1]
        Dp_orig = dw_label[2] 

        #calculate normelized mean square error
        Dp_norm_error = np.sqrt(np.square(Dp_orig-Dp_lsq))/Dp_orig
        Dt_norm_error = np.sqrt(np.square(Dt_orig-Dt_lsq))/Dt_orig
        Fp_norm_error = np.sqrt(np.square(Fp_orig-Fp_lsq))/Fp_orig
        S0_norm_error = np.sqrt(np.square(1-S0_lsq))
        #appened all reaults
        Dp_lsq_NRMSE = np.append(Dp_lsq_NRMSE, Dp_norm_error)
        Dt_lsq_NRMSE = np.append(Dt_lsq_NRMSE, Dt_norm_error)
        Fp_lsq_NRMSE = np.append(Fp_lsq_NRMSE, Fp_norm_error)
        S0_lsq_NRMSE = np.append(S0_lsq_NRMSE, S0_norm_error)

    
    return [ Dp_lsq_NRMSE, Dt_lsq_NRMSE, Fp_lsq_NRMSE, S0_lsq_NRMSE]




def boxplot_ivim(all_data, title, save_to=None):
    from pathlib import Path
    
    labels = ['D*_NET', 'D*_SUPER','Dt_NET', 'Dt_SUPER','Fp_NET','Fp_SUPER']
    fig, ax = plt.subplots()
    
    # rectangular box plot
    bplot = ax.boxplot(all_data,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    ax.set_title(title)
        
    # fill with colors
    colors = ['lightgreen', 'lightblue', 'lightgreen', 'lightblue', 'lightgreen', 'lightblue']

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    # adding horizontal grid lines
    ax.yaxis.grid(True)
    ax.set_xlabel('IVIM Parameters')
    ax.set_ylabel('Observed values')
    ax.set_ylim(0, 1.5)

    if save_to is None:
        plt.show()
    else:
        plt.savefig(Path(save_to) / "boxplot.png")

def loss_plot_supervised(loss_Dp, loss_Dt, loss_Fp, loss_train,
                         loss_val, val_loss_Dp, val_loss_Dt, val_loss_Fp, 
                         loss_recon = [], val_loss_recon = []):
    plt.figure(427)
    plt.clf()
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ion()
    plt.legend(["Train", "Validation"])
    plt.title("Total Loss (Reconstruction + IVIM Parameters)")
    plt.show()
    
    plt.figure(5848)
    plt.clf()
    plt.plot(loss_Dp, 'k')
    plt.plot(val_loss_Dp, 'y')

    plt.yscale("log")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ion()
    plt.legend(["Train loss D*", "Valid loss D*"])
    plt.title("D* Error")
    plt.show()
    
    plt.figure(146)
    plt.clf()
    plt.plot(loss_Dt, 'c')
    plt.plot(val_loss_Dt, 'm')
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ion()
    plt.legend(["Train loss Dt", "Valid loss Dt"])
    plt.title("D Error")
    plt.show()
    
    plt.figure(319)
    plt.clf()
    plt.plot(loss_Fp, 'g')
    plt.plot(val_loss_Fp, 'r')
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ion()
    plt.legend(["Train loss Fp", "Valid loss Fp"])
    plt.title("Fp Error")
    plt.show()

    if val_loss_recon: # check if the variable is not empty
        plt.figure(213)
        plt.clf()
        plt.plot(loss_recon, 'k')
        plt.plot(val_loss_recon, 'g')
        plt.yscale("log")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.ion()
        plt.legend(["Train loss Recon", "Valid loss Recon"])
        plt.title("Reconstruction Error")
        plt.show()
    
if __name__ == "__main__":
    from ..hyperparams import hyperparams as hp
    import simulations
    from os import listdir
    from os.path import isfile, join
    SNR = 10
    arg = hp()
    arg = checkarg(arg)
    bvalues = arg.sim.bvalues
    sample_size  = [10, 50, 100, 250, 500, 1000, 2000]
    for n_samples in sample_size:
        #n_samples = 100
        IVIM_signal_noisy, D, f, Dp = simulations.sim_signal(SNR, bvalues, n_samples, Dmin=arg.sim.range[0][0],
                                                  Dmax=arg.sim.range[1][0], fmin=arg.sim.range[0][1],
                                                  fmax=arg.sim.range[1][1], Dsmin=arg.sim.range[0][2],
                                                  Dsmax=arg.sim.range[1][2], rician=arg.sim.rician)
        
        labels = np.stack((D, f, Dp), axis=1).squeeze()  
        
        #IVIM_clinical, sb, sx, sy, b_val = simulations.clinical_signal()
        ### select only relevant values, delete background and noise, and normalise data
        
        # mypath_IVIMNET = '/tcmldrive/Noam/Elad_Net/IVIMNET_saved_models/SNR_20_IVIMNET.pt'
        # mypath_IVIMSUPER = '/tcmldrive/Noam/Elad_Net/IVIMSUPER_saved_models/SNR20_IVIMSUPER.pt'
        
        mypath_IVIMNET = r'C:\Users\ang.a\Documents\SUPER-IVIM-DC\checkpoints\exp1_simulations\20231002-073626\IVIMNET_SNR_10_sf_1.pt'
        mypath_IVIMSUPER = r'C:\Users\ang.a\Documents\SUPER-IVIM-DC\checkpoints\exp1_simulations\20231002-073626\SUPER-IVIM-DC_SNR_10_sf_1.pt'

        #ivim_error, Dp_infer, Dt_infer, Fp_infer, S0_infer = infer_clinical_supervised_IVIM(IVIM_clinical, b_val, mypath_IVIMSUPER, arg)
    
        DtNET_error, FpNET_error, DpNET_error, S0NET_error = infer_supervised_IVIM(IVIM_signal_noisy, labels, bvalues, mypath_IVIMNET, arg)
        DtSUPER_error, FpSUPER_error, DpSUPER_error, S0SUPER_error = infer_supervised_IVIM(IVIM_signal_noisy, labels, bvalues, mypath_IVIMSUPER, arg)
    
        errors_np_array = np.stack([DpNET_error,  DpSUPER_error, DtNET_error, DtSUPER_error,
                                    FpNET_error, FpSUPER_error,], axis=1)
        bp_title = "IVIMNET VS IVIMSUPER parameters error SNR=10"
        
        boxplot_ivim(errors_np_array, bp_title)
