from ds_policy import DSPolicy
import os
import glob
import numpy as np

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'custom_data/smoothing_window_21_test_saving_contact')
    
# Get all eef_traj files in the directory
eef_traj_files = sorted(glob.glob(os.path.join(data_dir, '*_eef_traj.npy')))

# Load each file and store in a list
eef_traj_data = []
for file_path in eef_traj_files:
    try:
        data = np.load(file_path)
        eef_traj_data.append(data)
    except Exception as e:
        print(f"Error loading {os.path.basename(file_path)}: {str(e)}")

ds_policy = DSPolicy(eef_traj_data, pos_model_path='neural_ode/models/mlp_width64_depth3.eqx')

for i in range(100):
    state = eef_traj_data[0][i]
    action = ds_policy.get_action(state)
    print(action)