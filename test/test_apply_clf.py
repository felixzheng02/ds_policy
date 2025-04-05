import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm

from ds_policy.policy import DSPolicy
from ds_policy.ds_utils import load_data, euler_to_quat, quat_to_euler
from test_ds_policy import update_state


if __name__ == "__main__":
    x, x_dot, q, omega, gripper = load_data("open_single_door", "move_towards", finger=True, transform_to_handle_frame=True, debug_on=False)
    index = 4
    demo_pos = x[index]
    demo_quat = q[index]
    demo_euler = quat_to_euler(demo_quat)
    model_config = {
            'pos_model': {
                'special_mode': 'none',
            },
            'quat_model': {
                'special_mode': 'none',
            }
        }
    ds_policy = DSPolicy(
        x,
        x_dot,
        q,
        omega,
        gripper,
        model_config=model_config,
        dt=1/60, 
        switch=False, 
        lookahead=10
    )

    cur_state = np.concatenate([demo_pos[0], demo_euler[0]])
    traj = [cur_state]
    for i in range(demo_pos.shape[0]):
        cur_quat = euler_to_quat(cur_state[3:])
        vel = ds_policy._apply_clf(np.concatenate([cur_state[:3], cur_quat]), np.zeros(6), alpha_V=10)
        cur_state = update_state(cur_state, vel, 1/60)
        traj.append(cur_state)

    # Convert trajectories to numpy arrays
    traj = np.array(traj)
    
    # Plot the comparison between generated trajectory and demonstration
    plt.figure(figsize=(15, 15))
    
    # Position and angular velocity comparison - separate plots for each dimension
    dimensions = ['X', 'Y', 'Z', 'Angular Velocity X', 'Angular Velocity Y', 'Angular Velocity Z']
    
    # Create subplots for all 7 dimensions (3 position + 4 quaternion)
    for i in range(6):
        plt.subplot(6, 1, i+1)
        plt.plot(traj[:, i], label='Generated Trajectory')
        plt.plot(demo_pos[:, i] if i < 3 else demo_quat[:, i-3], label='Demonstration')
        plt.title(f'{dimensions[i]} Comparison')
        plt.xlabel('Time Step')
        plt.ylabel(dimensions[i])
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()


    
