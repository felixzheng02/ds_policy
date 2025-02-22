import os, sys
import numpy as np
import pyLasaDataset as lasa
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_data(input_opt, show_plot=False, separate=False, shift=True):
    """
    Return:
    -------
        x:     a [M, N] NumPy array: M observations of N dimension
    
        x_dot: a [M, N] NumPy array: M observations velocities of N dimension

        x_att: a [1, N] NumPy array of attractor

        x_init: an L-length list of [1, N] NumPy array: L number of trajectories, each containing an initial point of N dimension
    """

    if input_opt == 1:
        print("\nYou selected PC-GMM benchmark data.\n")
        pcgmm_list = ["3D_sink", "3D_viapoint_1", "3D-cube-pick", "3D_viapoint_2", "2D_Lshape",  "2D_incremental_1", "2D_multi-behavior", "2D_messy-snake"]

        message = """Available Models: \n"""
        for i in range(len(pcgmm_list)):
            message += "{:2}) {: <18} ".format(i+1, pcgmm_list[i])
            if (i+1) % 6 ==0:
                message += "\n"
        message += '\nEnter the corresponding option number [type 0 to exit]: '
        
        # data_opt = int(input(message))
        data_opt = int(7)
        if data_opt == 0:
            sys.exit()
        elif data_opt<0 or data_opt>len(pcgmm_list):
            print("Invalid data option")
            sys.exit()

        data_name  = str(pcgmm_list[data_opt-1]) + ".mat"
        input_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "pc-gmm-data", data_name)

        data_ = loadmat(r"{}".format(input_path))
        data_ = np.array(data_["data"])

        N     = int(data_[0, 0].shape[0]/2)
        if N == 2:
            L = data_.shape[1]
            x     = [data_[0, l][:N, :].T  for l in range(L)]
            x_dot = [data_[0, l][N:, :].T  for l in range(L)]
        elif N == 3:
            L = data_.shape[0]
            L_sub = np.random.choice(range(L), 6, replace=False)

            x     = [data_[l, 0][:N, :].T  for l in range(L)]
            x_dot = [data_[l, 0][N:, :].T  for l in range(L)]


    elif input_opt == 2:
        print("\nYou selected LASA benchmark dataset.\n")

        # suppress print message from lasa package
        original_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')
        sys.stdout = original_stdout

        lasa_list = ["Angle", "BendedLine", "CShape", "DoubleBendedLine", "GShape", "heee", "JShape", "JShape_2", "Khamesh", "Leaf_1",
        "Leaf_2", "Line", "LShape", "NShape", "PShape", "RShape", "Saeghe", "Sharpc", "Sine", "Snake",
        "Spoon", "Sshape", "Trapezoid", "Worm", "WShape", "Zshape", "Multi_Models_1", "Multi_Models_2", "Multi_Models_3", "Multi_Models_4"]

        message = """Available Models: \n"""
        for i in range(len(lasa_list)):
            message += "{:2}) {: <18} ".format(i+1, lasa_list[i])
            if (i+1) % 6 ==0:
                message += "\n"
        message += '\nEnter the corresponding option number [type 0 to exit]: '
        
        data_opt = int(input(message))

        if data_opt == 0:
            sys.exit()
        elif data_opt<0 or data_opt > len(lasa_list):
            print("Invalid data option")
            sys.exit()

        data = getattr(lasa.DataSet, lasa_list[data_opt-1])
        demos = data.demos 
        sub_sample = 1
        L = len(demos)

        x     = [demos[l].pos[:, ::sub_sample].T for l in range(L)]
        x_dot = [demos[l].vel[:, ::sub_sample].T for l in range(L)]


    elif input_opt == 3:
        print("\nYou selected Damm demo dataset.\n")

        damm_list = ["bridge", "Nshape", "orientation"]
        
        message = """Available Models: \n"""
        for i in range(len(damm_list)):
            message += "{:2}) {: <18} ".format(i+1, damm_list[i])
            if (i+1) % 6 ==0:
                message += "\n"
        message += '\nEnter the corresponding option number [type 0 to exit]: '
        
        data_opt = int(input(message))
    
        folder_name = str(damm_list[data_opt-1])
        input_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "damm-demo-data", folder_name, "all.mat")
        x, x_dot    = _process_bag(input_path)



    elif input_opt == 4:
        print("\nYou selected demo.\n")
        input_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "demo", "obstacle", "all.mat")
        x, x_dot    = _process_bag(input_path)



    elif input_opt == 'demo':
        print("\nYou selected demo.\n")
        input_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "demo", "all.mat")
        x, x_dot    = _process_bag(input_path)



    elif input_opt == 'increm':
        print("\nYou selected demo.\n")
        input_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "increm", "all.mat")
        x, x_dot    = _process_bag(input_path)



    elif input_opt == 'apple':
        print("\nYou selected apple.\n")
        input_path = "/home/emp/lfd_ws/src/lfd_ds/demo/1/all.npz"
        x, x_dot    = _process_npz(input_path)

    elif input_opt == 'custom':
        print("\nYou selected custom data.\n")
        input_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "custom_data")
        
        # List all eef trajectory files
        traj_files = [f for f in os.listdir(input_path) if f.endswith('_eef_traj.npy')]
        L = len(traj_files)
        
        x = []
        x_dot = []
        
        for l in range(L):
            # Load EEF and handle trajectory data
            demo_num = traj_files[l].split('_')[1]  # Get demo number
            eef_traj = np.load(os.path.join(input_path, f'demo_{demo_num}_eef_traj.npy'))
            handle_traj = np.load(os.path.join(input_path, f'demo_{demo_num}_handle_traj.npy'))
            # Check trajectory length
            
            # Extract positions and rotation matrices
            eef_pos = eef_traj[:, :3]
            handle_pos = handle_traj[:, :3]
            handle_rot = handle_traj[:, 3:].reshape(-1, 3, 3)  # Reshape to 3x3 rotation matrices
            
            # Compute relative position in world frame
            rel_pos_world = eef_pos - handle_pos
            
            # Transform relative positions to handle frame
            pos_traj = np.zeros_like(rel_pos_world)
            for i in range(len(rel_pos_world)):
                # R.T @ p transforms from world frame to handle frame 
                pos_traj[i] = handle_rot[i].T @ rel_pos_world[i]

            # Define break point (example coordinates - adjust as needed)
            break_point = np.array([0.0, -0.05, -0.05])  # Point in handle frame
            
            # Find closest point to break point
            distances = np.linalg.norm(pos_traj - break_point, axis=1)
            split_idx = np.argmin(distances)
            
            # Split trajectory into pre and post segments
            if len(pos_traj[:split_idx+1]) < 2 or len(pos_traj[split_idx:]) < 2:
                continue
            pos_traj_pre = pos_traj[:split_idx+1]
            pos_traj_post = pos_traj[split_idx:]
            
            # Compute velocities for pre segment
            dt = 1/60
            vel_traj_pre = np.diff(pos_traj_pre, axis=0) / dt
            for i in range(len(vel_traj_pre)):
                vel_traj_pre[i] = handle_rot[i+1].T @ handle_rot[i] @ vel_traj_pre[i]
            vel_traj_pre = np.vstack([vel_traj_pre, vel_traj_pre[-1]])
            
            # Compute velocities for post segment  
            vel_traj_post = np.diff(pos_traj_post, axis=0) / dt
            for i in range(len(vel_traj_post)):
                vel_traj_post[i] = handle_rot[split_idx+i+1].T @ handle_rot[split_idx+i] @ vel_traj_post[i]
            vel_traj_post = np.vstack([vel_traj_post, vel_traj_post[-1]])
            
            # Store pre and post segments separately
            if l == 0:
                x_pre = [pos_traj_pre]
                x_dot_pre = [vel_traj_pre]
                x_post = [pos_traj_post]
                x_dot_post = [vel_traj_post]
            else:
                x_pre.append(pos_traj_pre)
                x_dot_pre.append(vel_traj_pre)
                x_post.append(pos_traj_post)
                x_dot_post.append(vel_traj_post)
        
        # if show_plot:
        # # Plot each trajectory
        #     for pos_traj, vel_traj in zip(x_pre, x_dot_pre):
        #         # Plot position trajectory in blue for pre-grasp
        #         ax.plot(pos_traj[:, 0], pos_traj[:, 1], pos_traj[:, 2], 'b-', label='Pre-grasp')
                
        #         # Plot velocity arrows (every 20 points to avoid clutter)
        #         stride = 30
        #         for i in range(0, len(pos_traj), stride):
        #             ax.quiver(pos_traj[i, 0], pos_traj[i, 1], pos_traj[i, 2],
        #                     vel_traj[i, 0], vel_traj[i, 1], vel_traj[i, 2],
        #                     color='blue', alpha=0.6, length=0.05, normalize=False)
            
        #     # Plot post-grasp trajectories in red
        #     for pos_traj, vel_traj in zip(x_post, x_dot_post):
        #         ax.plot(pos_traj[:, 0], pos_traj[:, 1], pos_traj[:, 2], 'r-', label='Post-grasp')
        #         stride = 30
        #         for i in range(0, len(pos_traj), stride):
        #             ax.quiver(pos_traj[i, 0], pos_traj[i, 1], pos_traj[i, 2],
        #                     vel_traj[i, 0], vel_traj[i, 1], vel_traj[i, 2],
        #                     color='red', alpha=0.6, length=0.05, normalize=False)
    
        #     ax.set_xlabel('X')
        #     ax.set_ylabel('Y') 
        #     ax.set_zlabel('Z')
        #     ax.set_title('Relative EEF-Handle Trajectories with Velocities')
        #     plt.show()
        # Process both trajectory sets
        x_pre_processed, x_dot_pre_processed, x_att_pre, x_init_pre = _pre_process(x_pre, x_dot_pre, separate=separate, shift=shift)
        x_dot_post_reversed = [-vel_traj for vel_traj in x_dot_post]  # Reverse velocities
        x_post_processed, x_dot_post_processed, x_att_post, x_init_post = _pre_process(x_post, x_dot_post_reversed, reverse=True, separate=separate, shift=shift)
        # Plot processed post-grasp trajectories
        if show_plot:
            visualize_trajectories(x_pre_processed, 'Pre-grasp')
        # Select which trajectory set to return based on flag
        use_post = False  # Flag to toggle between pre/post trajectories
        if use_post:
            x, x_dot = x_post_processed, x_dot_post_processed
            x_att, x_init = x_att_post, x_init_post
        else:
            x, x_dot = x_pre_processed, x_dot_pre_processed 
            x_att, x_init = x_att_pre, x_init_pre


        return x, x_dot, x_att, x_init

    else:
        input_path = os.path.join(input_opt, "all.npz")
        x, x_dot    = _process_npz(input_path) 

    return _pre_process(x, x_dot)


def visualize_trajectories(x, label):
    """Visualize multiple trajectories with color gradients representing time progression.
    
    Args:
        x: numpy array of shape (L, M, N) where:
           L is number of trajectories
           M is number of points per trajectory
           N is dimension of each point (should be 3 for 3D visualization)
        label: string label for the plot title
    """
    # Create figure with a specific layout to accommodate colorbar
    fig = plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(1, 20)
    ax = fig.add_subplot(gs[0, :18], projection='3d')
    cax = fig.add_subplot(gs[0, 18:])
    
    # Get number of trajectories and points
    L, M, N = x.shape
    
    # Create a color map from blue to red
    cmap = plt.cm.RdBu_r  # Red-Blue colormap, reversed to go from blue to red
    
    # Create normalized time points for color mapping
    time_points = np.linspace(0, 1, M-1)
    colors = cmap(time_points)
    
    # Plot each trajectory with the same time-based color gradient
    for l in range(L):
        # Plot trajectory segments with color gradient
        for i in range(M-1):
            ax.plot(x[l, i:i+2, 0], x[l, i:i+2, 1], x[l, i:i+2, 2],
                   color=colors[i],
                   linewidth=2)
    
    # Add a colorbar to show time progression
    sm = plt.cm.ScalarMappable(cmap=cmap,
                              norm=plt.Normalize(vmin=0, vmax=M))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Time progression', rotation=270, labelpad=15)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{label} Trajectories')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()



def _pre_process(x, x_dot, reverse=False, separate=False, shift=True):
    """ 
    Roll out nested lists into a single list of M entries or keep trajectories separate

    Parameters:
    -------
        x:     an L-length list of [M, N] NumPy array: L number of trajectories, each containing M observations of N dimension,
    
        x_dot: an L-length list of [M, N] NumPy array: L number of trajectories, each containing M observations velocities of N dimension
        
        reverse: bool, if True use first points as attractor and last points as initial
        
        separate: bool, if True keep trajectories separate and return array of size (L, M, N)
        
        shift: bool, if True shift trajectories to align their attractors, if False return original trajectories

    Note:
    -----
        M can vary and need not be same between trajectories when separate=False
        When separate=True, trajectories will be padded to the same length using the last position and zero velocity
    """

    L = len(x)
    x_init = []
    x_shifted = []

    if reverse:
        x_att = [x[l][0, :] for l in range(L)]  # Use first points as attractor
        x_att_mean = np.mean(np.array(x_att), axis=0, keepdims=True)
        for l in range(L):
            x_init.append(x[l][-1].reshape(1, -1))  # Use last points as initial
            if shift:
                x_diff = x_att_mean - x_att[l]
                x_shifted.append(x_diff.reshape(1, -1) + x[l])
            else:
                x_shifted.append(x[l])
    else:
        x_att = [x[l][-1, :] for l in range(L)]  # Use last points as attractor
        x_att_mean = np.mean(np.array(x_att), axis=0, keepdims=True)
        for l in range(L):
            x_init.append(x[l][0].reshape(1, -1))  # Use first points as initial
            if shift:
                x_diff = x_att_mean - x_att[l]
                x_shifted.append(x_diff.reshape(1, -1) + x[l])
            else:
                x_shifted.append(x[l])

    if separate:    
        # Find maximum length among all trajectories
        max_length = max(x_shifted[l].shape[0] for l in range(L))
        N = x_shifted[0].shape[1]  # Get dimension N from first trajectory
        
        # Initialize arrays with correct shape (L, M, N)
        x_array = np.zeros((L, max_length, N))
        x_dot_array = np.zeros((L, max_length, N))
        
        # Fill arrays with data and pad as needed
        for l in range(L):
            curr_length = x_shifted[l].shape[0]
            # Fill existing trajectory data
            x_array[l, :curr_length, :] = x_shifted[l]
            x_dot_array[l, :curr_length, :] = x_dot[l]
            # Pad remaining positions with last position
            if curr_length < max_length:
                x_array[l, curr_length:, :] = x_shifted[l][-1]
                # Velocities are already zero-padded by initialization
        
        return x_array, x_dot_array, x_att_mean, x_init
    else:
        # Original behavior: roll out into single array
        for l in range(L):
            if l == 0:
                x_rollout = x_shifted[l]
                x_dot_rollout = x_dot[l]
            else:
                x_rollout = np.vstack((x_rollout, x_shifted[l]))
                x_dot_rollout = np.vstack((x_dot_rollout, x_dot[l]))

        return x_rollout, x_dot_rollout, x_att_mean, x_init




def _process_bag(path):
    """ Process .mat files that is converted from .bag files """

    data_ = loadmat(r"{}".format(path))
    data_ = data_['data_ee_pose']
    L = data_.shape[1]

    x     = []
    x_dot = [] 

    sample_step = 4
    vel_thresh  = 1e-3 
    
    for l in range(L):
        data_l = data_[0, l]['pose'][0,0]
        pos_traj  = data_l[:3, ::sample_step]
        quat_traj = data_l[3:7, ::sample_step]
        time_traj = data_l[-1, ::sample_step].reshape(1,-1)

        raw_diff_pos = np.diff(pos_traj)
        vel_mag = np.linalg.norm(raw_diff_pos, axis=0).flatten()
        first_non_zero_index = np.argmax(vel_mag > vel_thresh)
        last_non_zero_index = len(vel_mag) - 1 - np.argmax(vel_mag[::-1] > vel_thresh)

        if first_non_zero_index >= last_non_zero_index:
            raise Exception("Sorry, vel are all zero")

        pos_traj  = pos_traj[:, first_non_zero_index:last_non_zero_index]
        quat_traj = quat_traj[:, first_non_zero_index:last_non_zero_index]
        time_traj = time_traj[:, first_non_zero_index:last_non_zero_index]
        vel_traj = np.diff(pos_traj) / np.diff(time_traj)
        
        x.append(pos_traj[:, 0:-1].T)
        x_dot.append(vel_traj.T)

    return x, x_dot





def _process_npz(path):
    """ Process .npz files that is converted from .bag files """

    data_ = np.load(path, allow_pickle=True)
    data_ = data_['data_ee_pose']
    L = data_.shape[0]

    x     = []
    x_dot = [] 

    sample_step = 1
    vel_thresh  = 1e-3 
    
    cutoff = 20
    for l in range(L):
        data_l = data_[l]['pose']
        pos_traj  = data_l[:3, ::sample_step]
        quat_traj = data_l[3:7, ::sample_step]
        time_traj = data_l[-1, ::sample_step].reshape(1,-1)

        raw_diff_pos = np.diff(pos_traj)
        vel_mag = np.linalg.norm(raw_diff_pos, axis=0).flatten()
        first_non_zero_index = np.argmax(vel_mag > vel_thresh)
        last_non_zero_index = len(vel_mag) - 1 - np.argmax(vel_mag[::-1] > vel_thresh)

        if first_non_zero_index >= last_non_zero_index:
            raise Exception("Sorry, vel are all zero")

        pos_traj  = pos_traj[:, first_non_zero_index:last_non_zero_index]
        quat_traj = quat_traj[:, first_non_zero_index:last_non_zero_index]
        time_traj = time_traj[:, first_non_zero_index:last_non_zero_index]


        pos_traj  = pos_traj[:, cutoff: ]
        quat_traj = quat_traj[:, cutoff: ]
        time_traj = time_traj[:, cutoff: ]

        vel_traj = np.diff(pos_traj) / np.diff(time_traj)
        
        x.append(pos_traj[:, 0:-1].T)
        x_dot.append(vel_traj.T)

    return x, x_dot