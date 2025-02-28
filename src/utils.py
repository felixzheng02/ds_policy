import os
import glob
import numpy as np

def load_data(path):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'custom_data/smoothing_window_21_test_saving_contact')
    
    # Get all eef_traj files in the directory
    traj_files = sorted(glob.glob(os.path.join(data_dir, '*_eef_traj.npy')))
    
    # Load each file and store in a list
    trajs = []
    for file in traj_files:
        try:
            traj = np.load(file)
            trajs.append(traj)
        except Exception as e:
            print(f"Error loading {os.path.basename(file)}: {str(e)}")
    
    return trajs