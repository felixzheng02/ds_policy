from ds_policy import DSPolicy
import os
import glob
import numpy as np
import utils
from load_tools import load_data

x, x_dot, r = load_data('custom')

demo_trajs = [np.concatenate([pos, rot], axis=1) for pos, rot in zip(x, r)]


ds_policy = DSPolicy(demo_trajs, x_dot, pos_model_path='neural_ode/models/mlp_width64_depth3.eqx')

for i in range(100):
    state = demo_trajs[0][i]
    action = ds_policy.get_action(state)
    print(action)