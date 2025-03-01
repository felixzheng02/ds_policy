import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_data(path, regex=None):
    if regex is None:
        traj_files = sorted(glob.glob(os.path.join(path, "*")))
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


def quat_to_euler(quat):
    return R.from_quat(quat).as_euler("xyz", degrees=False)


def euler_to_quat(euler):
    return R.from_euler("xyz", euler, degrees=False).as_quat()
