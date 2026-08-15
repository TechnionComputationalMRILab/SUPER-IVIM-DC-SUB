from super_ivim_dc_boot.source.hyperparams import hyperparams as hp
from super_ivim_dc_boot.train import train

import numpy as np
import matplotlib.pyplot as plt

working_dir: str = './working_dir'
ivimnet_filename: str = 'ivimnet'  # do not include .pt
super_ivim_dc_filename: str = 'super_ivim_dc_boot' # do not include .pt

bvalues = np.array([0,10,20,40,80,200,400,600,1000])  

arg= hp('sim_boot')

train(arg=arg,
    bvalues=bvalues, 
    super_ivim_dc=True,
    ivimnet=False,
    work_dir=working_dir,
    super_ivim_dc_filename=super_ivim_dc_filename,
    ivimnet_filename=ivimnet_filename,
    verbose=False,
)