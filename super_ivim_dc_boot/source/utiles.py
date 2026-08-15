import os
import time
import numpy as np
import SimpleITK as sitk


def nrmse_calc(lsq_params, deep_params):
    rmse = np.sqrt(np.square(np.subtract(deep_params , lsq_params).mean()))
    nrmse = rmse/np.mean(lsq_params)
    return nrmse

def create_working_folder(output_directory):
    print(os.getcwd())

    # create working dir string
    timestamp = time.strftime("%Y%m%d-%H%M%S")#get_name_with_time()
    current_work_dir = os.path.join(output_directory, timestamp)

    # create folders
    if not os.path.exists(current_work_dir):
        os.makedirs(current_work_dir, exist_ok=True)
    #os.chmod(current_work_dir, mode=0o777)
        os.makedirs(os.path.join(current_work_dir, 'init'), exist_ok=True)
    #os.chmod(os.path.join(current_work_dir, 'Models'), mode=0o777)
    print(current_work_dir)

    return current_work_dir

def read_data_sitk(filename):
    sitk_dwi = sitk.ReadImage(filename)
    data = sitk.GetArrayFromImage(sitk_dwi)
    return data


if __name__ == '__main__':

    pass