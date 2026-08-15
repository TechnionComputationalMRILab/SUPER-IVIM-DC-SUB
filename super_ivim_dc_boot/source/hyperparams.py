
import torch
import numpy as np

class train_pars:
    def __init__(self,nets):
        self.optim = 'adam' # 'sgd'; 'sgdr'; 'adagrad'
        if nets == 'optim':
            self.lr = 0.0001
        elif nets == 'orig':
            self.lr = 0.001
        else:
            self.lr = 0.0001
        self.patience = 10 # number of epochs without improvement until determining it found its optimum
        self.batch_size= 128
        self.maxit = 500 # max iterations per epoch
        self.split = 0.9 # split of test and validation data
        self.load_nn = False # load the neural network instead of retraining
        self.loss_fun = 'rms' #  L1
        self.skip_net = False # skip the network training and evaluation
        self.scheduler = False # allows to reduce the LR itteratively when there is no improvement throughout 5 consecutive epochs
        # use GPU if available
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.select_best = True
        self.ivim_combine = True # if True uses SUPER-IVIM-DC else IVIMNET
        # print(f'ivim combine value {self.ivim_combine}')

class net_pars:
    def __init__(self,nets):
        # select a network setting
        if (nets == 'optim') or (nets == 'optim_adsig') :
            # the optimized network settings
            self.dropout = 0.1
            self.batch_norm = True # batch normalization
            self.parallel = True # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
            self.con = 'sigmoid' # constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
            #### only if sigmoid constraint is used!
            self.cons_min = [0.0005, 0.05, 0.008, 0.9]#[-0.0001, -0.05, -0.05, 0.7]  #[1.39 / 1000, 0.2475, 13.06 / 1000, 0.7] # # Dt, Fp, Ds, S0 #[0, 0, 0.005, 0.8] #
            self.cons_max = [0.0025, 0.55, 0.1, 1]#[0.005, 0.7, 0.3, 1.3] #[1.97 / 1000, 0.3909, 32.78 / 1000, 1.3] # # Dt, Fp, Ds, S0 #[0.005, 0.7, 0.2, 1.2] #
            ####
            self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 (for normalized signals)
            self.depth = 4 # number of layers
            self.width = 0 # if 0 the width is the number of b-values
        elif nets == 'orig':
            # as summarized in Table 1 from the main article for the original network
            self.dropout = 0
            self.batch_norm = False # batch normalization
            self.parallel = False  # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
            self.con = 'abs' # constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
            #### only if sigmoid constraint is used!
            self.cons_min = [-0.0001, -0.05, -0.05, 0.7]  # Dt, Fp, Ds, S0
            self.cons_max = [0.005, 0.7, 0.3, 1.3]  # Dt, Fp, Ds, S0
            ####
            self.fitS0 = False # indicates whether to fit S0 (True) or fix it to 1 (for normalised signals)
            self.depth = 3 # number of layers
            self.width = 0 # if 0 the width is the number of b-values

        else:
            # chose wisely :)
            self.dropout = 0.3
            self.batch_norm = True # batch normalization
            self.parallel = True # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
            self.con = 'sigmoid' # constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
            #### only if sigmoid constraint is used!
            self.cons_min = [-0.0001, -0.05, -0.05, 0.7]  # Dt, Fp, Ds, S0
            self.cons_max = [0.005, 0.7, 0.3, 1.3]  # Dt, Fp, Ds, S0
            ####
            self.fitS0 = False # indicates whether to fit S0 (True) or fix it to 1 (for normalized signals)
            self.depth = 4 # number of layers
            self.width = 0 # if 0 the width is the number of b-values
        boundsrange = 0.3 * (np.array(self.cons_max)-np.array(self.cons_min)) # ensure that we are on the most linear part of the sigmoid function
        self.cons_min = np.array(self.cons_min) - boundsrange
        self.cons_max = np.array(self.cons_max) + boundsrange
        self.cons_min[-1] = 0.7

class lsqfit:
    def __init__(self):
        self.method = 'bayes' # seg, bayes or lsq
        self.do_fit = False # True # if False skip lsq fitting
        self.load_lsq = False # load last results for lsq fit
        self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 in the least squares fit.
        self.jobs = 4 # number of parallel jobs. If set to 1, no parallel computing is used
        self.bounds = ([0, 0, 0.005, 0.7],[0.005, 0.7, 0.3, 1.3]) #Dt, Fp, Ds, S0


class sim:
    def __init__(self):
        self.bvalues = np.array([0,10,20,40,80,200,400,600,1000])
        self.SNR = [7,10, 15, 20, 30, 50] # the SNRs to simulate
        self.sims = 100000 # number of simulations to run
        self.num_samples_eval = 1000 # number of simulations te evaluate.
        self.repeats = 1 # number of repeats for simulations
        self.rician = True # False # adds rician noise to simulations; if false, gaussian noise is added instead
        self.range = ([0.0005, 0.05, 0.01],
                      [0.003, 0.55, 0.1])
        self.seed= 0
        print(f'simulative model \n {self.SNR} SNR \n {self.sims} samples \n rician noise is set to {self.rician} \n bvalues are: {self.bvalues}')

class sim_boot:
    def __init__(self):
        self.bvalues = np.array([0,10,20,40,80,200,400,600,1000])  #np.array([0,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000])
        self.SNR = [7,10, 15, 20, 30, 50] # [7,10, 15, 20, 30, 50] # the SNRs to simulate
        self.sims = 100000 # number of simulations to run
        self.num_samples_eval = 1000 # number of simulations te evaluate.
        self.repeats = 1 # number of repeats for simulations
        self.rician = True # False # adds rician noise to simulations; if false, gaussian noise is added instead
        self.range = ([0.0005, 0.05, 0.01],
                      [0.003, 0.55, 0.1])   
        self.size_boot_low= 3
        self.size_boot_high= 3
        self.n_subsamples= 16 
        self.seed= 100     
        print(f'simulative model \n {self.SNR} SNR \n {self.sims} samples \n rician noise is set to {self.rician} \n bvalues are: {self.bvalues}')

class sim_bvalue:
    def __init__(self):
        self.bvalues = np.array([0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1000]) 
        self.bvalues_eval= np.array([0,10,20,40,80,200,400,600,1000])
        self.SNR = [7,10, 15, 20, 30, 50] # [15, 20, 30, 50] # the SNRs to simulate
        self.sims = 100000 # number of simulations to run
        self.num_samples_eval = 1000 # number of simulations te evaluate.
        self.repeats = 1 # number of repeats for simulations
        self.rician = True # False # adds rician noise to simulations; if false, gaussian noise is added instead
        self.range = ([0.0005, 0.05, 0.01],
                      [0.003, 0.55, 0.1]) 
        self.seed= 100
        self.size_low = 4
        self.size_high = 4         
        
class sim_clinic:
    def __init__(self):
        self.bvalues = np.array([0,10,20,40,80,200,400,600,1000]) #np.array([0, 50, 100, 200, 400, 600, 800]) # array of b-values
        self.SNR = [50] # [15, 20, 30, 50] # the SNRs to simulate
        self.sims = 1000000 # number of simulations to run
        self.num_samples_eval = 1000 # number of simualtiosn te evaluate. This can be lower than the number run.
        self.repeats = 1 # number of repeats for simulations
        self.rician = True # add rician noise to simulations; if false, gaussian noise is added instead
        self.range = ([0.0005, 0.05, 0.008],
                      [0.0025, 0.55, 0.1])
        #([0.0005, 0.05, 0.01],
                      #[0.003, 0.55, 0.1])
        self.seed= 0
        print(f'clinic model - {self.SNR} SNR, {self.sims} samples, rician noise is set to {self.rician}, bvalues are: {self.bvalues}')

class sim_fetal:
    def __init__(self):
        self.bvalues = np.array([0,10,20,40,80,200,400,600,1000])
        self.SNR = [10] # [15, 10, 30, 50] # the SNRs to simulate
        self.sims = 1000000 # number of simulations to run
        self.num_samples_eval = 1000 # number of simualtiosn te evaluate.
        self.repeats = 1 # number of repeats for simulations
        self.rician = True # add rician noise to simulations; if false, gaussian noise is added instead
        self.range = ([1.39 / 1000, 0.2475, 13.06 / 1000],
                        [1.97 / 1000, 0.3909, 32.78 / 1000])
        self.seed= 0
        print(f'clinic model for fetal data - {self.SNR} SNR, {self.sims} samples, rician noise is set to {self.rician}, bvalues are: {self.bvalues}')
  
class sim_phantom:
    def __init__(self):
        self.bvalues = np.array([100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500]) # array of b-values
        self.SNR = [10] # [15, 20, 30, 50] # the SNRs to simulate
        print(f'phantom model with {self.SNR} SNR')
        self.sims = 1000000 # number of simulations to run
        self.num_samples_eval = 1000 # number of simualtiosn te evaluate. This can be lower than the number run.
        self.repeats = 1 # number of repeats for simulations
        self.rician = True # add rician noise to simulations; if false, gaussian noise is added instead
        self.range = ([0.0005, 0.05, 0.01],
                      [0.003, 0.55, 0.1])
        self.seed= 0
        print(f'clinic model for phantom data - {self.SNR} SNR, {self.sims} samples, rician noise is set to {self.rician}, bvalues are: {self.bvalues}')


class hyperparams:
    def __init__(self, key = 'sim'):
        self.fig = True # plot results and intermediate steps
        self.save_name = 'optim' # orig or optim (or optim_adsig for in vivo)
        self.net_pars = net_pars(self.save_name)
        self.train_pars = train_pars(self.save_name)
        self.fit = lsqfit() #Performs fitting: segmented least square, bayes or lsq
        if key =='clinic':
            self.sim = sim_clinic()
            self.loss_coef_ivim = torch.FloatTensor([0.4])
            self.loss_coef_Dp = torch.FloatTensor([200])
            self.key = 'clinic'
            print('hyperparams class is clinic')
        elif key =='sim':
            self.sim = sim()
            self.loss_coef_ivim = torch.FloatTensor([0.1])
            self.loss_coef_Dp = torch.FloatTensor([200])
            self.key = 'sim'
            print('hyperparams class is sim')
        elif key == 'fetal':
            self.sim = sim_fetal()
            self.loss_coef_ivim = torch.FloatTensor([0.4])
            self.loss_coef_Dp = torch.FloatTensor([200])
            self.key = 'fetal'
            print('hyperparams class is fetal')
        elif key =='phantom':
            self.sim = sim_phantom()
            self.loss_coef_ivim = torch.FloatTensor([0.1])
            self.loss_coef_Dp = torch.FloatTensor([200])
            self.key = 'phantom'
            print('hyperparams class is phantom')
        elif key =='sim_boot':
            self.sim = sim_boot()
            self.loss_coef_ivim = torch.FloatTensor([0.1]) 
            self.loss_coef_Dp = torch.FloatTensor([100])
            self.key = 'sim_boot'
            print('hyperparams class is sim_boot')
        elif key =='sim_bvalue':
            self.sim = sim_bvalue()
            self.loss_coef_ivim = torch.FloatTensor([0.1])
            self.loss_coef_Dp = torch.FloatTensor([10000])
            self.key = 'sim_bvalue'
            print('hyperparams class is sim_bvalue') 
        
        self.loss_coef_Dt = torch.FloatTensor([20000]) 
        self.loss_coef_Fp = torch.FloatTensor([80]) 


