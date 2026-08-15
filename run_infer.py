from super_ivim_dc_boot.source.hyperparams import hyperparams as hp
from super_ivim_dc_boot.infer import infer_from_signal
from super_ivim_dc_boot.IVIMNET import simulations

import numpy as np

bvalues = np.array([0,10,20,40,80,200,400,600,1000]) 
snr = 10
sample_size = 1

IVIM_signal_noisy, Dt, f, Dp = simulations.sim_signal(
    SNR=snr, 
    bvalues=bvalues, 
    sims=sample_size,
    state=100
)

Dt, f, Dp = np.squeeze(Dt), np.squeeze(f), np.squeeze(Dp)


working_dir: str = './working_dir'
super_ivim_dc_filename: str = 'super_ivim_dc_boot'

arg = hp(key ='sim_boot')

Dp_superivimdc_boot, Dt_superivimdc_boot, Fp_superivimdc_boot, S0_superivimdc_boot = infer_from_signal(
    arg=arg,
    signal=IVIM_signal_noisy, 
    bvalues=bvalues,
    model_path=f"{working_dir}/{super_ivim_dc_filename}.pt",
)

Dt_superivimdc_boot, Fp_superivimdc_boot, Dp_superivimdc_boot = np.squeeze(Dt_superivimdc_boot), np.squeeze(Fp_superivimdc_boot), np.squeeze(Dp_superivimdc_boot)

print("Ground Truth D,f,D*: ", Dt, f, Dp)
print("Inferred D,f,D*: ", Dt_superivimdc_boot, Fp_superivimdc_boot, Dp_superivimdc_boot)