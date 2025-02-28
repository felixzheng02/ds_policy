from ds_policy import DSPolicy
import os
import glob
import numpy as np
import utils

demo_trajs = utils.load_data(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'custom_data/smoothing_window_21_test_saving_contact'), '*_eef_traj.npy')

ds_policy = DSPolicy(demo_trajs, pos_model_path='neural_ode/models/mlp_width64_depth3.eqx')

for i in range(100):
    state = demo_trajs[0][i]
    action = ds_policy.get_action(state)
    print(action)