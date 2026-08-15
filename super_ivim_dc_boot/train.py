from .source.train_model import train_model
from .source.hyperparams import hyperparams as hp
from .IVIMNET import deep
import numpy as np
from pathlib import Path

DEFAULT_BVALUE_LIST = np.array([0,15,30,45,60,75,90,105,120,135,150,175,200,400,600,800])


def train_entry():
    import argparse

    parser = argparse.ArgumentParser(
        description="SUPER-IVIM-DC Simulation", 
        epilog='Developed by the Technion Computational MRI lab: https://tcml-bme.github.io/'
    )

    parser.add_argument("--snr", "-snr", default=10, help="SNR value")

    parser.add_argument("--bval", "-b", default=-1, help="b-value as a comma separated list (without spaces)")

    parser.add_argument("--mode", "-m", default="both", help="Simulation mode (can be SUPER-IVIM-DC, IVIMNET, or both)")

    parser.add_argument("--output", "-o", default="./output", help="Working directory")

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")

    args = parser.parse_args()

    if args.bval == -1:
        bvalues = DEFAULT_BVALUE_LIST
    else:
        bvalues = np.array(args.bval.split(","), dtype=int)

    if args.verbose:
        print(f"SNR: {args.snr}")
        print(f"b-values: {bvalues}")
        print(f"Mode: {args.mode}")
        print(f"Output directory: {args.output}")    

    train(
        SNR=args.snr, 
        bvalues=bvalues, 
        mode=args.mode,
        work_dir=args.output
        )

def train(arg,
        bvalues=DEFAULT_BVALUE_LIST, 
        super_ivim_dc: bool = True,
        ivimnet: bool = True,
        super_ivim_dc_filename: str = 'super_ivim_dc',
        ivimnet_filename: str = 'ivimnet',
        work_dir="./output",
        verbose=False, 
    ):
    # set up arguments
    arg = deep.checkarg(arg)
    arg.sim.bvalues = bvalues
    arg.fig = False
    arg.verbose = verbose

    # create the work directory if it doesn't exist 
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    sf = 1  # sampling factor
    if super_ivim_dc:
        matNN_superivimdc = train_model(
            key='sim_boot', 
            arg=arg, 
            mode="SUPER-IVIM-DC", 
            sf=sf, 
            filename=super_ivim_dc_filename, 
            work_dir=work_dir
            )
        for idx , snr in enumerate(arg.sim.SNR):
            np.savetxt(f'{work_dir}/{super_ivim_dc_filename}_{snr}_NRMSE.csv', np.asarray(matNN_superivimdc[idx]), delimiter=",")

    if ivimnet:
        matNN_ivimnet = train_model(
            key='sim', 
            arg=arg, 
            mode='IVIMNET', 
            sf=sf, 
            work_dir=work_dir,
            filename=ivimnet_filename
            )
        np.savetxt(f'{work_dir}/{ivimnet_filename}_NRMSE.csv', np.asarray(matNN_ivimnet), delimiter=",")
