import os
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.stats as scipy

from . import deep
from . import fitting_algorithms as fit


def sim(arg, supervised, sf, mode, work_dir, filename):#, params, case, dist): bvalues, arg, sf, snr, mode
    """ This function defines how well the different fit approaches perform on simulated data. Data is simulated by
    randomly selecting a value of D, f and D* from within the predefined range. The script calculates the random,
    systematic, root-mean-squared error (RMSE) and Spearman Rank correlation coefficient for each of the IVIM parameters.
    Furthermore, it calculates the stability of the neural network (when trained multiple times).

    input:
    :param SNR: SNR of the simulated data. If SNR is set to 0, no noise is added
    :param arg: an object with simulation options. hyperparams.py gives most details on the object (and defines it),
    Relevant attributes are:
    arg.sim.sims = number of simulations to be performed (need a large amount for training)
    arg.sim.num_samples_eval = number of samples to evaluate (save time for lsq fitting)
    arg.sim.repeats = number of times to repeat the training and evaluation of the network (to assess stability)
    arg.sim.bvalues: 1D Array of b-values used
    arg.fit contains the parameters regarding lsq fitting
    arg.train_pars and arg.net_pars contain the parameters regarding the neural network
    arg.sim.range gives the simulated range of D, f and D* in a 2D array

    :return matlsq: 2D array containing the performance of the lsq fit (if enabled). The rows indicate D, f (Fp), D*
    (Dp), whereas the colums give the mean input value, the random error and the systematic error
    :return matNN: 2D array containing the performance of the NN. The rows indicate D, f (Fp), D*
    (Dp), whereas the colums give the mean input value, the random error and the systematic error
    :return stability: a 1D array with the stability of D, f and D* as a fraction of their mean value.
    Stability is only relevant for neural networks and is calculated from the repeated network training.
    """
    arg = deep.checkarg(arg)
    IVIM_signal_noisy_lst= []
    D_lst=[]
    f_lst=[]
    Dp_lst= []
    for snr in arg.sim.SNR:
        IVIM_signal_noisy, D, f, Dp = sim_signal(
        SNR=snr, 
        bvalues = arg.sim.bvalues, 
        sims=arg.sim.sims, 
        Dmin=arg.sim.range[0][0], Dmax=arg.sim.range[1][0], 
        fmin=arg.sim.range[0][1], fmax=arg.sim.range[1][1], 
        Dsmin=arg.sim.range[0][2], Dsmax=arg.sim.range[1][2], 
        rician=arg.sim.rician,
        key = arg.key,
        state= 123
        )
        IVIM_signal_noisy_lst.append(IVIM_signal_noisy)
        D_lst.append(D)
        f_lst.append(f)
        Dp_lst.append(Dp)    
    IVIM_signal_noisy= np.vstack(IVIM_signal_noisy_lst) 
    D= np.vstack(D_lst)
    f= np.vstack(f_lst) 
    Dp= np.vstack(Dp_lst)  
       
    if arg.sim.repeats > 1:
        paramsNN = np.zeros([arg.sim.repeats, 4, arg.sim.num_samples_eval])
    else:
        paramsNN = np.zeros([len(arg.sim.SNR), 4, arg.sim.num_samples_eval])

    if not arg.train_pars.skip_net:
        # loop over repeats
        for aa in range(arg.sim.repeats):
            start_time = time.time()
            # train network

            if arg.verbose:
                print('\nRepeat: {repeat}\n'.format(repeat=aa))
            # supervised addition
            if supervised:
                # combine the ouput and the IVIM parameters as labels here
                if arg.verbose:
                    print('Supervised Training')
                labels = np.stack((D, f, Dp), axis=1).squeeze() 

                # add the labels to this function
                # net = deep.learn_supervised_IVIM(IVIM_signal_noisy, labels, arg.sim.bvalues, arg, sf, SNR, mode, work_dir, filename)#, case) 
                net = deep.learn_supervised_IVIM(              
                    X_train=IVIM_signal_noisy, 
                    labels=labels, 
                    bvalues= arg.sim.bvalues,
                    arg=arg, 
                    sf=sf,  
                    mode=mode, 
                    work_dir=work_dir, 
                    filename=filename
                    )     

            else:
                # net = deep.learn_IVIM(IVIM_signal_noisy, arg.sim.bvalues, arg, sf, SNR, mode, work_dir)
                net = deep.learn_IVIM(
                    X_train=IVIM_signal_noisy, 
                    bvalues=arg.sim.bvalues, 
                    arg=arg, 
                    sf=sf, 
                    mode=mode, 
                    work_dir=work_dir,
                    filename=filename
                    )
            elapsed_time = time.time() - start_time
    
            if arg.verbose:
                print('\ntime elapsed for training: {}\n'.format(elapsed_time))
            start_time = time.time()
            # predict parameters
            if arg.sim.repeats > 1:
                paramsNN[aa] = deep.predict_IVIM(IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.sim.bvalues, net,
                                                 arg) # size?
            else:
                if supervised:
                    if arg.verbose:
                        print('Supervised Prediction')
                    matNN = np.zeros([len(arg.sim.SNR), 3, 5]) 
                    for idx , snr in enumerate(arg.sim.SNR):
                        if arg.key=='sim_bvalue':
                            IVIM_signal_noisy_eval, D_eval, f_eval, Dp_eval = sim_signal(
                            snr, 
                            arg.sim.bvalues_eval, 
                            sims= arg.sim.num_samples_eval, 
                            Dmin=arg.sim.range[0][0], Dmax=arg.sim.range[1][0], 
                            fmin=arg.sim.range[0][1], fmax=arg.sim.range[1][1], 
                            Dsmin=arg.sim.range[0][2], Dsmax=arg.sim.range[1][2], 
                            rician=arg.sim.rician,
                            key = arg.key,
                            state= 100
                            )
                        else:
                            IVIM_signal_noisy_eval, D_eval, f_eval, Dp_eval = sim_signal(
                            snr, 
                            arg.sim.bvalues, 
                            sims= arg.sim.num_samples_eval, 
                            Dmin=arg.sim.range[0][0], Dmax=arg.sim.range[1][0], 
                            fmin=arg.sim.range[0][1], fmax=arg.sim.range[1][1], 
                            Dsmin=arg.sim.range[0][2], Dsmax=arg.sim.range[1][2], 
                            rician=arg.sim.rician,
                            key = arg.key,
                            state= 100
                            ) 
                        labels_eval = np.stack(( D_eval, f_eval, Dp_eval), axis=1).squeeze() 
                        if arg.key== 'sim_boot': 
                            for i in range(arg.sim.num_samples_eval):   
                                paramsNN[idx,:, i] = deep.predict_supervised_IVIM(
                                    np.expand_dims(IVIM_signal_noisy_eval[i], axis=0) ,
                                    np.expand_dims(labels_eval[i], axis=0), 
                                    arg.sim.bvalues, 
                                    net, 
                                    arg
                                    )
                        elif arg.key=='sim_bvalue':     
                            paramsNN[idx] = deep.predict_supervised_IVIM(
                            IVIM_signal_noisy_eval,labels_eval, 
                            arg.sim.bvalues_eval,  
                            net, 
                            arg
                            )   
                        else:
                            paramsNN[idx] = deep.predict_supervised_IVIM(
                            IVIM_signal_noisy_eval,labels_eval, 
                            arg.sim.bvalues,  
                            net, 
                            arg
                            )
                        X_train = IVIM_signal_noisy_eval
                        nan_idx = isnan(np.mean(X_train, axis=1))
                        #X_train = np.delete(X_train, nan_idx , axis=0)
                        if arg.key=='sim_bvalue':
                            bvalues = arg.sim.bvalues_eval
                        else:
                            bvalues = arg.sim.bvalues
                        b_less_50_idx = np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3 # X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
                        b_greater_50_idx = np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2 #  X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
                        b_greater_150_idx = np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1 # X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
                        thresh_idx = b_less_50_idx & b_greater_50_idx & b_greater_150_idx

                        D_eval, Dp_eval, f_eval = np.delete(D_eval, nan_idx , axis=0), np.delete(Dp_eval, nan_idx , axis=0), np.delete(f_eval, nan_idx , axis=0)
                        D_eval, Dp_eval, f_eval= D_eval[thresh_idx, ], Dp_eval[thresh_idx, ], f_eval[thresh_idx, ]
                        paramsNN_snr= paramsNN[idx]
                        paramsNN_snr= paramsNN_snr[:,thresh_idx]

                            # determine errors and Spearman Rank
                        print(f'\nSNR={snr}')
                        matNN[idx] = print_errors(np.squeeze(D_eval), np.squeeze(f_eval), np.squeeze(Dp_eval), paramsNN_snr)
                        stability = np.zeros(3)    
                    plot_errors(matNN, arg.sim.SNR, work_dir, arg)    
                else:
                    paramsNN = deep.predict_IVIM(IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.sim.bvalues, net, arg)
            elapsed_time = time.time() - start_time
    
            if arg.verbose:
                print('\ntime elapsed for inference: {}\n'.format(elapsed_time))
            
            if arg.train_pars.use_cuda:
               torch.cuda.empty_cache()

        if arg.verbose:
            print('results for NN')

        if not supervised:  
            X_train = IVIM_signal_noisy_eval
            nan_idx = isnan(np.mean(X_train, axis=1))
            #X_train = np.delete(X_train, nan_idx , axis=0)

            D_eval = D[:arg.sim.num_samples_eval]
            Dp_eval = Dp[:arg.sim.num_samples_eval]
            f_eval = f[:arg.sim.num_samples_eval]

            D_eval, Dp_eval, f_eval = np.delete(D_eval, nan_idx , axis=0), np.delete(Dp_eval, nan_idx , axis=0), np.delete(f_eval, nan_idx , axis=0)


        if arg.sim.repeats > 1:
            matNN = np.zeros([arg.sim.repeats, 3, 3])
            for aa in range(arg.sim.repeats):
                # determine errors and Spearman Rank
                matNN[aa] = print_errors(np.squeeze(D_eval), np.squeeze(f_eval), np.squeeze(Dp_eval), paramsNN[aa])
            matNN = np.mean(matNN, axis=0)
            # calculate Stability Factor
            stability = np.sqrt(np.mean(np.square(np.std(paramsNN, axis=0)), axis=1))
            stability = stability[[0, 1, 2]] / [np.mean(D_eval), np.mean(f_eval), np.mean(Dp_eval)]
            # set paramsNN for the plots
            paramsNN_0 = paramsNN[0]
        # else:
        #     matNN = print_errors(np.squeeze(D_eval), np.squeeze(f_eval), np.squeeze(Dp_eval), paramsNN)
        #     stability = np.zeros(3)
        #     paramsNN_0 = paramsNN
        # del paramsNN
        # show figures if requested
        plots(arg, D_eval, Dp_eval, f_eval, paramsNN)
    else:
        # if network is skipped
        stability = np.zeros(3)
        matNN = np.zeros([3, 5])
    if arg.fit.do_fit:
        start_time = time.time()
        # all fitting is done in the fit.fit_dats for the other fitting algorithms (lsq, segmented and Baysesian)
        paramsf = fit.fit_dats(arg.sim.bvalues, IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.fit)
        elapsed_time = time.time() - start_time

        if arg.verbose:
            print('\ntime elapsed for fit: {}\n'.format(elapsed_time))
            print('results for fit')

        # determine errors and Spearman Rank
        matlsq = print_errors(np.squeeze(D_eval), np.squeeze(f_eval), np.squeeze(Dp_eval), paramsf)
        # del paramsf, IVIM_signal_noisy
        # show figures if requested
        plots(arg, D_eval, Dp_eval, f_eval, paramsf)
        return matlsq, matNN, stability
    else:
        # if lsq fit is skipped, don't export lsq results
        return matNN, stability


def plots(arg,D,Dp,f,params):
    if arg.fig:
        dummy = np.array(params)
        # plot correlations
        plt.figure()
        plt.plot(D[:1000], Dp[:1000], 'rx', markersize=5)
        plt.xlim(0, 0.005)
        plt.ylim(0, 0.3)
        plt.xlabel('Dt')
        plt.ylabel('Dp')
        plt.gcf()
        # plt.savefig('plots/inputDtDp.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(f[:1000], Dp[:1000], 'rx', markersize=5)
        plt.xlim(0, 0.6)
        plt.ylim(0, 0.3)
        plt.xlabel('f')
        plt.ylabel('Dp')
        plt.gcf()
        # plt.savefig('plots/inputfDp.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(D[:1000], f[:1000], 'rx', markersize=5)
        plt.xlim(0, 0.005)
        plt.ylim(0, 0.6)
        plt.xlabel('Dt')
        plt.ylabel('f')
        plt.gcf()
        # plt.savefig('plots/inputDtf.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(dummy[0, :1000], dummy[2, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.005)
        plt.ylim(0, 0.3)
        plt.xlabel('Dt')
        plt.ylabel('Dp')
        plt.gcf()
        # plt.savefig('plots/DtDp.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(dummy[1, :1000], dummy[2, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.6)
        plt.ylim(0, 0.3)
        plt.xlabel('f')
        plt.ylabel('Dp')
        plt.gcf()
        # plt.savefig('plots/fDp.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(dummy[0, :1000], dummy[1, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.005)
        plt.ylim(0, 0.6)
        plt.xlabel('Dt')
        plt.ylabel('f')
        plt.gcf()
        # plt.savefig('plots/Dtf.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(Dp[:1000], dummy[2, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.3)
        plt.ylim(0, 0.3)
        plt.ylabel('DpNN')
        plt.xlabel('Dpin')
        plt.gcf()
        # plt.savefig('plots/DpoutDpin.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(D[:1000], dummy[0, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.005)
        plt.ylim(0, 0.005)
        plt.ylabel('DtNN')
        plt.xlabel('Dtin')
        plt.gcf()
        # plt.savefig('plots/DtoutDtin.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(f[:1000], dummy[1, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.6)
        plt.ylim(0, 0.6)
        plt.ylabel('fNN')
        plt.xlabel('fin')
        plt.gcf()
        # plt.savefig('plots/foutfin.png')
        plt.ion()
        plt.show()
        #plt.close('all') # Keep the plot open/close them


def augmented_signal(data, bvalues, arg, fraction=0.3, Dmin=0.3 / 1000, Dmax=4.0 / 1000, fmin=0.0, fmax=0.8, Dsmin=0.01,
                     Dsmax=0.2):
    """
    This simulates IVIM curves with real noise from actual data. This can be used as augmented training data.
    Data is simulated by randomly selecting a value of D, f and D* from within the predefined range.

    input:
    :param data: real data from experiment
    :param bvalues: 1D Array of b-values used
    :param sims: number of simulations to be performed (need a large amount for training)

    optional:
    :param fraction: fraction of data to augment
    :param Dmin: minimal simulated D. Default = 0.0003
    :param Dmax: maximal simulated D. Default = 0.004
    :param fmin: minimal simulated f. Default = 0.0
    :param Dmax: minimal simulated f. Default = 0.8
    :param Dpmin: minimal simulated D*. Default = 0.01
    :param Dpmax: minimal simulated D*. Default = 0.2

    :return IVIM_signal_noisy: 2D array with augmented noisy IVIM signal (x-axis is sims long, y-axis is len(b-values) long)
    """
    arg = deep.checkarg(arg)
    indx = np.argsort(bvalues.copy())
    sdata = np.shape(data)[0]
    # train
    net = deep.learn_IVIM(data[:, indx], bvalues[indx], arg)
    # select random samples to augment
    sels = random.sample(range(sdata), round(sdata * fraction))
    # predict noise
    noise = deep.predict_noise(data[sels][:, indx], bvalues[indx], net, arg)
    # augment data
    IVIM_signal_noisy, D, f, Dp = sim_signal(0, bvalues[indx], sims=len(noise), Dmin=Dmin, Dmax=Dmax, fmin=fmin,
                                             fmax=fmax,
                                             Dsmin=Dsmin, Dsmax=Dsmax)
    # add noise to augmented data
    IVIM_signal_noisy = IVIM_signal_noisy + noise
    indx_back = np.argsort(indx)
    del net
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    return IVIM_signal_noisy[indx_back]


def sim_signal(SNR, bvalues, sims=1000000, Dmin=0.5 / 1000, Dmax=2.0 / 1000, fmin=0.1, fmax=0.5, Dsmin=0.05, Dsmax=0.2,
              rician=False, state=123, key = 'sim'
              ):
    """
    This simulates IVIM curves. Data is simulated by randomly selecting a value of D, f and D* from within the
    predefined range.

    input:
    :param SNR: SNR of the simulated data. If SNR is set to 0, no noise is added
    :param bvalues: 1D Array of b-values used
    :param sims: number of simulations to be performed (need a large amount for training)

    optional:
    :param Dmin: minimal simulated D. Default = 0.0005
    :param Dmax: maximal simulated D. Default = 0.002
    :param fmin: minimal simulated f. Default = 0.1
    :param Dmax: minimal simulated f. Default = 0.5
    :param Dpmin: minimal simulated D*. Default = 0.05
    :param Dpmax: minimal simulated D*. Default = 0.2
    :param rician: boolean giving whether Rician noise is used; default = False

    :return data_sim: 2D array with noisy IVIM signal (x-axis is sims long, y-axis is len(b-values) long)
    :return D: 1D array with the used D for simulations, sims long
    :return f: 1D array with the used f for simulations, sims long
    :return Dp: 1D array with the used D* for simulations, sims long
    """

    # randomly select parameters from predefined range
    rg = np.random.RandomState(state)
    test = rg.uniform(0, 1, (sims, 1))
    D = Dmin + (test * (Dmax - Dmin))
    test = rg.uniform(0, 1, (sims, 1))
    f = fmin + (test * (fmax - fmin))
    test = rg.uniform(0, 1, (sims, 1))
    Dp = Dsmin + (test * (Dsmax - Dsmin))

    # initialise data array
    data_sim = np.zeros([len(D), len(bvalues)])
    bvalues = np.array(bvalues)

    # loop over array to fill with simulated IVIM data
    for aa in range(len(D)):
        data_sim[aa, :] = fit.ivim(bvalues, D[aa][0], f[aa][0], Dp[aa][0], 1)

    # if SNR is set to zero, don't add noise
    if SNR > 0:
        # initialise noise arrays
        noise_imag = np.zeros([sims, len(bvalues)])
        noise_real = np.zeros([sims, len(bvalues)])
        # fill arrays
        for i in range(0, sims):
            noise_real[i,] = rg.normal(0, 1 / SNR,
                                       (1, len(bvalues)))  # wrong! need a SD per input. Might need to loop to maD noise
            noise_imag[i,] = rg.normal(0, 1 / SNR, (1, len(bvalues)))
        if rician:
            # add Rician noise as the square root of squared gaussian distributed real signal + noise and imaginary noise
            data_sim = np.sqrt(np.power(data_sim + noise_real, 2) + np.power(noise_imag, 2))
        else:
            # or add Gaussian noise
            data_sim = data_sim + noise_imag
    else:
        data_sim = data_sim
        
    if key == 'phantom':
        # normalise signal
        S0_noisy = np.mean(data_sim[:, bvalues == 100], axis=1)
        data_sim = data_sim / S0_noisy[:, None]
    else:
        # normalise signal
        S0_noisy = np.mean(data_sim[:, bvalues == 0], axis=1)
        data_sim = data_sim / S0_noisy[:, None]
    return data_sim, D, f, Dp


def print_errors(D, f, Dp, params):

    rmse_D = np.sqrt(np.square(np.subtract(D, params[0])).mean())

    rmse_f = np.sqrt(np.square(np.subtract(f, params[1])).mean())

    rmse_Dp = np.sqrt(np.square(np.subtract(Dp, params[2])).mean())

    # initialise Spearman Rank matrix
    Spearman = np.zeros([3, 2])
    # calculate Spearman Rank correlation coefficient and p-value
    Spearman[0, 0], Spearman[0, 1] = scipy.spearmanr(params[0], params[2])  # DvDp
    Spearman[1, 0], Spearman[1, 1] = scipy.spearmanr(params[0], params[1])  # Dvf
    Spearman[2, 0], Spearman[2, 1] = scipy.spearmanr(params[1], params[2])  # fvDp
    # If spearman is nan, set as 1 (because of constant estimated IVIM parameters)
    Spearman[np.isnan(Spearman)] = 1
    # take absolute Spearman
    Spearman = np.absolute(Spearman)
    
    norm_D_pred = np.mean(params[0])
    norm_f_pred = np.mean(params[1])
    norm_Dp_pred = np.mean(params[2])
    
    std_D_err = (np.subtract(D, params[0])).std()
    std_f_err = (np.subtract(f, params[1])).std()
    std_Dp_err = (np.subtract(Dp, params[2])).std()
    
    del params

    normD_lsq = np.mean(D)
    normf_lsq = np.mean(f)
    normDp_lsq = np.mean(Dp)
    
    print('results from NN: columns show themean, the SD/mean, the systematic error/mean, the RMSE/mean and the Spearman coef [DvDp,Dvf,fvDp] \n'
          'the rows show D, f and D*\n')
    print([normD_lsq, '  ', rmse_D / normD_lsq, ' ', Spearman[0, 0]])
    print([normf_lsq, '  ', rmse_f / normf_lsq, ' ', Spearman[1, 0]])
    print([normDp_lsq, '  ', rmse_Dp / normDp_lsq,' ', Spearman[2, 0]])
    
    mats = [[normD_lsq, norm_D_pred, std_D_err, rmse_D / normD_lsq, Spearman[0, 0]],
            [normf_lsq, norm_f_pred ,std_f_err, rmse_f / normf_lsq, Spearman[1, 0]],
            [normDp_lsq, norm_Dp_pred, std_Dp_err, rmse_Dp / normDp_lsq , Spearman[2, 0]]]

    return mats


def sim_signal_normal_dist(SNR, bvalues, D_init, f_init, Ds_init, sims=100000, rician=False, state=123, key = 'sim'):
    
    # initialise data array
    n =1000000
    data_sim = np.zeros([n, len(bvalues)])
    bvalues = np.array(bvalues)
    
    # D = np.zeros((n, len(bvalues)))
    # f = np.zeros((n, len(bvalues)))
    # Ds = np.zeros((n, len(bvalues)))
    
    D = np.random.normal(D_init, 0.2*D_init, n)
    f = np.random.normal(f_init, 0.2*f_init, n)
    Ds = np.random.normal(Ds_init, 0.2*Ds_init, n)

    
    # loop over array to fill with simulated IVIM data
    for aa in range(n):
        data_sim[aa, :] = fit.ivim(bvalues, D[aa], f[aa], Ds[aa], 1)

    # if SNR is set to zero, don't add noise
    # if SNR > 0:
    #     # initialise noise arrays
    #     noise_imag = np.zeros([sims, len(bvalues)])
    #     noise_real = np.zeros([sims, len(bvalues)])
    #     # fill arrays
    #     for i in range(0, sims - 1):
    #         noise_real[i,] = rg.normal(0, 1 / SNR,
    #                                    (1, len(bvalues)))  # wrong! need a SD per input. Might need to loop to maD noise
    #         noise_imag[i,] = rg.normal(0, 1 / SNR, (1, len(bvalues)))
    #     if rician:
    #         # add Rician noise as the square root of squared gaussian distributed real signal + noise and imaginary noise
    #         data_sim = np.sqrt(np.power(data_sim + noise_real, 2) + np.power(noise_imag, 2))
    #     else:
    #         # or add Gaussian noise
    #         data_sim = data_sim + noise_imag
    # else:
    #     data_sim = data_sim
        
    if key == 'phantom':
        # normalize signal
        S0_noisy = np.mean(data_sim[:, bvalues == 100], axis=1)
        data_sim = data_sim / S0_noisy[:, None]
    else:
        # normalise signal
        S0_noisy = np.mean(data_sim[:, bvalues == 0], axis=1)
        data_sim = data_sim / S0_noisy[:, None]
    return data_sim, D, f, Ds

def sim_signal_predict(arg, SNR):
    # init randomstate
    rg = np.random.RandomState(123)
    ## define parameter values in the three regions
    S0_region0, S0_region1, S0_region2 = 1, 1, 1
    Dp_region0, Dp_region1, Dp_region2 = 0.03, 0.05, 0.07
    Dt_region0, Dt_region1, Dt_region2 = 0.0020, 0.0015, 0.0010
    Fp_region0, Fp_region1, Fp_region2 = 0.15, 0.3, 0.45
    # image size
    sx, sy, sb = 100, 100, len(arg.sim.bvalues)
    # create image
    dwi_image = np.zeros((sx, sy, sb))
    Dp_truth = np.zeros((sx, sy))
    Dt_truth = np.zeros((sx, sy))
    Fp_truth = np.zeros((sx, sy))

    # fill image with simulated values
    for i in range(sx):
        for j in range(sy):
            if (40 < i < 60) and (40 < j < 60):
                # region0
                dwi_image[i, j, :] = fit.ivim(arg.sim.bvalues, Dt_region0, Fp_region0, Dp_region0, S0_region0)
                Dp_truth[i, j], Dt_truth[i, j], Fp_truth[i, j] = Dp_region0, Dt_region0, Fp_region0
            elif (20 < i < 80) and (20 < j < 80):
                # region1
                dwi_image[i, j, :] = fit.ivim(arg.sim.bvalues, Dt_region1, Fp_region1, Dp_region1, S0_region1)
                Dp_truth[i, j], Dt_truth[i, j], Fp_truth[i, j] = Dp_region1, Dt_region1, Fp_region1
            else:
                # region2
                dwi_image[i, j, :] = fit.ivim(arg.sim.bvalues, Dt_region2, Fp_region2, Dp_region2, S0_region2)
                Dp_truth[i, j], Dt_truth[i, j], Fp_truth[i, j] = Dp_region2, Dt_region2, Fp_region2

    # plot simulated diffusion weighted image
    fig, ax = plt.subplots(2, int(np.round(arg.sim.bvalues.shape[0] / 2)), figsize=(20, 20))
    b_id = 0
    for i in range(2):
        for j in range(int(np.round(arg.sim.bvalues.shape[0] / 2))):
            if not b_id == arg.sim.bvalues.shape[0]:
                ax[i, j].imshow(dwi_image[:, :, b_id], cmap='gray', clim=(0, 1))
                ax[i, j].set_title('b = ' + str(arg.sim.bvalues[b_id]))
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            else:
                # ax[i, j].imshow(dwi_image[:, :, b_id], cmap='gray', clim=(0, 1))
                ax[i, j].set_title('End of b-values')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            b_id += 1
    plt.subplots_adjust(hspace=0)
    plt.show()
    if not os.path.isdir('plots'):
        os.makedirs('plots')
    plt.savefig('plots/plot_dwi_without_noise_param_{snr}_{method}.png'.format(snr=SNR, method=arg.save_name))

    # Initialise dwi noise image
    dwi_noise_imag = np.zeros((sx, sy, sb))
    # fill dwi noise image with Gaussian noise
    for i in range(sx):
        for j in range(sy):
            dwi_noise_imag[i, j, :] = rg.normal(0, 1 / SNR, (1, len(arg.sim.bvalues)))
    # Add Gaussian noise to dwi image
    dwi_image_noise = dwi_image + dwi_noise_imag
    # normalise signal
    S0_dwi_noisy = np.mean(dwi_image_noise[:, :, arg.sim.bvalues == 0], axis=1)
    dwi_image_noise_norm = dwi_image_noise / S0_dwi_noisy[:, None]

    # plot simulated diffusion weighted image with noise
    fig, ax = plt.subplots(2, int(np.round(arg.sim.bvalues.shape[0] / 2)), figsize=(20, 20))
    b_id = 0
    for i in range(2):
        for j in range(int(np.round(arg.sim.bvalues.shape[0] / 2))):
            if not b_id == arg.sim.bvalues.shape[0]:
                ax[i, j].imshow(dwi_image_noise_norm[:, :, b_id], cmap='gray', clim=(0, 1))
                ax[i, j].set_title('b = ' + str(arg.sim.bvalues[b_id]))
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            else:
                # ax[i, j].imshow(dwi_image[:, :, b_id], cmap='gray', clim=(0, 1))
                ax[i, j].set_title('End of b-values')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            b_id += 1
    plt.subplots_adjust(hspace=0)
    plt.show()
    plt.savefig('plots/plot_dwi_with_noise_param_{snr}_{method}.png'.format(snr=SNR, method=arg.save_name))

    # reshape image
    dwi_image_long = np.reshape(dwi_image_noise_norm, (sx * sy, sb))
    return dwi_image_long, Dt_truth, Fp_truth, Dp_truth


def plot_example1(paramsNN, paramsf, Dt_truth, Fp_truth, Dp_truth, arg, SNR):
    # initialise figure
    sx, sy, sb = 100, 100, len(arg.sim.bvalues)
    if arg.fit.do_fit:
        fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    else:
        fig, ax = plt.subplots(2, 3, figsize=(20, 20))
    # fill Figure with values
    Dt_t_plot = ax[0, 0].imshow(Dt_truth, cmap='gray', clim=(0, 0.003))
    ax[0, 0].set_title('Dt, ground truth')
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    fig.colorbar(Dt_t_plot, ax=ax[0, 0], fraction=0.046, pad=0.04)

    Dt_plot = ax[1, 0].imshow(np.reshape(paramsNN[0], (sx, sy)), cmap='gray', clim=(0, 0.003))
    ax[1, 0].set_title('Dt, estimate')
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    fig.colorbar(Dt_plot, ax=ax[1, 0], fraction=0.046, pad=0.04)

    if arg.fit.do_fit:
        Dt_fit_plot = ax[2, 0].imshow(np.reshape(paramsf[0], (sx, sy)), cmap='gray', clim=(0, 0.003))
        ax[2, 0].set_title('Dt, fit {fitmethod}'.format(fitmethod=arg.fit.method))
        ax[2, 0].set_xticks([])
        ax[2, 0].set_yticks([])
        fig.colorbar(Dt_fit_plot, ax=ax[2, 0], fraction=0.046, pad=0.04)

    Fp_t_plot = ax[0, 1].imshow(Fp_truth, cmap='gray', clim=(0, 0.5))
    ax[0, 1].set_title('Fp, ground truth')
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    fig.colorbar(Fp_t_plot, ax=ax[0, 1], fraction=0.046, pad=0.04)

    Fp_plot = ax[1, 1].imshow(np.reshape(paramsNN[1], (sx, sy)), cmap='gray', clim=(0, 0.5))
    ax[1, 1].set_title('Fp, estimate')
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    fig.colorbar(Fp_plot, ax=ax[1, 1], fraction=0.046, pad=0.04)
    
    if arg.fit.do_fit:
        Fp_fit_plot = ax[2, 1].imshow(np.reshape(paramsf[1], (sx, sy)), cmap='gray', clim=(0, 0.5))
        ax[2, 1].set_title('f, fit {fitmethod}'.format(fitmethod=arg.fit.method))
        ax[2, 1].set_xticks([])
        ax[2, 1].set_yticks([])
        fig.colorbar(Fp_fit_plot, ax=ax[2, 1], fraction=0.046, pad=0.04)

    Dp_t_plot = ax[0, 2].imshow(Dp_truth, cmap='gray', clim=(0.01, 0.1))
    ax[0, 2].set_title('Dp, ground truth')
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    fig.colorbar(Dp_t_plot, ax=ax[0, 2], fraction=0.046, pad=0.04)

    Dp_plot = ax[1, 2].imshow(np.reshape(paramsNN[2], (sx, sy)), cmap='gray', clim=(0.01, 0.1))
    ax[1, 2].set_title('Dp, estimate')
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    fig.colorbar(Dp_plot, ax=ax[1, 2], fraction=0.046, pad=0.04)
    
    if arg.fit.do_fit:
        Dp_fit_plot = ax[2, 2].imshow(np.reshape(paramsf[2], (sx, sy)), cmap='gray', clim=(0.01, 0.1))
        ax[2, 2].set_title('Dp, fit {fitmethod}'.format(fitmethod=arg.fit.method))
        ax[2, 2].set_xticks([])
        ax[2, 2].set_yticks([])
        fig.colorbar(Dp_fit_plot, ax=ax[2, 2], fraction=0.046, pad=0.04)

        plt.subplots_adjust(hspace=0.2)
        plt.show()
    plt.savefig('plots/plot_imshow_IVIM_param_{snr}.png'.format(snr=SNR, save=arg.save_name))

def isnan(x):
    # this program indicates what are NaNs 
    return x != x

def plot_errors(mat_errors, SNRs, work_dir,arg):
    errors= mat_errors[:,:,3]
    total_error= np.sum(errors, axis=1)
    params_str = ['D', 'f', 'DStar', 'total']
    plt.figure(figsize = (20,5))
    for i in range(3):
        plt.subplot(1, 4, i+1)
        plt.plot(SNRs, mat_errors[:, i , 3])
        plt.title(f'{params_str[i]}')
        plt.xlabel('SNR')
        plt.ylabel('NRMSE')
    plt.subplot(1,4,4)
    plt.plot(SNRs,total_error)
    plt.title(f'{params_str[3]}')
    plt.xlabel('SNR')
    plt.ylabel('NRMSE')
    plt.subplots_adjust(wspace=0.3)
    if arg.key== 'sim_boot':
        plt.savefig(f'{work_dir}/plot_NRMSE_{arg.sim.size_boot_low}_{arg.sim.size_boot_high}_{arg.sim.seed}_boot.png')
    elif arg.key== 'sim_bvalue':
        plt.savefig(f'{work_dir}/plot_NRMSE_{arg.sim.seed}_bvalue.png')    
    else:
        plt.savefig(f'{work_dir}/plot_NRMSE_{arg.sim.seed}_super_ivim_dc.png')        
    plt.show()       


    
