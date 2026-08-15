from pathlib import Path
import json
from ..IVIMNET import simulations as sim

def train_model(key, arg, mode, sf, filename, work_dir):

    if (mode == 'SUPER-IVIM-DC'):
        supervised = True
        if (key == 'fetal'):
            arg.loss_coef_ivim = 0.4
        else:
            # coef = [0.1,0.1,0.2,0.35,0.4] #[0.09, 0.1, 0.2, 0.18, 0.25, 0.4]
            coef = [0.09, 0.1, 0.2, 0.18, 0.25, 0.4] 
            arg.loss_coef_ivim = coef[sf-1]
        arg.train_pars.ivim_combine = True
        
    elif (mode == 'IVIMNET'):
        supervised = False
        arg.train_pars.ivim_combine = False
        
    init_settings = dict(range = arg.sim.range, cons_max = arg.net_pars.cons_max, cons_min = arg.net_pars.cons_min, bvalues = arg.sim.bvalues, loss_coef = arg.loss_coef_ivim, depth = arg.net_pars.depth, snr = arg.sim.SNR)

    with open(f'{work_dir}/{filename}_init.json', 'w') as fp:
      json.dump(init_settings, fp, default=str, indent=4, sort_keys=True)
    
    matNN,_  = sim.sim(
       arg=arg, 
       supervised=supervised, 
       sf=sf, 
       mode=mode, 
       work_dir=work_dir,
       filename=filename,
       )
    return matNN
