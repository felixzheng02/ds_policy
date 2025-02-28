import os
import glob
import numpy as np

def load_data(path, regex=None):
    if regex is None:
        traj_files = sorted(glob.glob(os.path.join(path, '*')))
    else:
        traj_files = sorted(glob.glob(os.path.join(path, regex)))
    
    # Load each file and store in a list
    trajs = []
    for file in traj_files:
        try:
            traj = np.load(file)
            trajs.append(traj)
        except Exception as e:
            print(f"Error loading {os.path.basename(file)}: {str(e)}")
    
    return trajs