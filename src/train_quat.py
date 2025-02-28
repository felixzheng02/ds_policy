import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import glob
import json

# Add the quaternion_ds package to sys.path to import it
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'se3_lpvds/src/quaternion_ds'))

from src.util.plot_tools import plot_gmm, plot_gamma, plot_omega
from src.util.process_tools import pre_process, compute_output, extract_state, rollout_list
from src.quat_class import quat_class
from src.gmm_class import gmm_class
from load_tools import load_data


def load_quat_model(model_path='models/quat_model.json'):
    """
    Load a trained quaternion dynamical system model from JSON files.
    
    Parameters:
    -----------
    model_path : str
        Path to the directory containing the model files.
        
    Returns:
    --------
    quat_obj : quat_class
        The loaded quaternion dynamical system model.
    """
    # Load the main model file
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    
    # Extract parameters from the loaded data
    K = model_data.get('K')
    M = model_data.get('M')
    dt = model_data.get('dt')
    q_att_array = np.array(model_data.get('att_ori'))
    q_att = R.from_quat(q_att_array)
    
    # Initialize the quat_class with placeholder data
    quat_obj = quat_class(None, None, q_att, dt, K_init=K, M=M)
    
    # Now manually set the model parameters from the loaded JSON
    # Reshape GMM parameters
    Prior = np.array(model_data.get('Prior'))
    Mu_flat = np.array(model_data.get('Mu'))
    Sigma_flat = np.array(model_data.get('Sigma'))
    
    # Reshape Mu and Sigma to their original dimensions
    Mu = Mu_flat.reshape(2*K, 4)
    Sigma = Sigma_flat.reshape(2*K, 4, 4)
    
    # Convert Mu to list of Rotation objects
    Mu_rot = [R.from_quat(mu) for mu in Mu]
    
    # Set GMM parameters
    quat_obj.gmm = gmm_class(None, q_att, K, M)
    quat_obj.gmm.Prior = Prior
    quat_obj.gmm.Mu = Mu_rot
    quat_obj.gmm.Sigma = Sigma
    
    # Set the dynamics matrices
    A_ori_flat = np.array(model_data.get('A_ori'))
    quat_obj.A_ori = A_ori_flat.reshape(2*K, 4, 4)
    
    # Set other parameters
    quat_obj.K = K
    quat_obj.dt = dt
    quat_obj.q_att = q_att
    
    print("Successfully loaded quaternion model from:", model_path)
    
    return quat_obj


def train_quat():
    k_init = 10
    output_path = 'models/quat_model.json'
     
    # Set time step for the data
    dt = 0.01  # Assuming 60Hz sampling rate
    
    # Process position and orientation data from the loaded trajectories
    p_raw = []  # Position trajectories
    q_raw = []  # Orientation trajectories
    t_raw = []  # Time vectors
    
    for i, traj in enumerate(eef_traj_data):
        # Verify that the data has the expected format (n, 7)
        if traj.shape[1] != 7:
            print(f"Warning: Trajectory {i} has unexpected shape {traj.shape}. Expected (n, 7). Skipping.")
            continue
        
        # Extract positions (first 3 columns) and quaternions (last 4 columns)
        positions = traj[:, :3]
        quaternions = traj[:, 3:7]
        
        # Create time vector based on dt
        times = np.arange(0, len(positions) * dt, dt)[:len(positions)]
        
        # Convert quaternions to Rotation objects
        rotations = [R.from_quat(quat) for quat in quaternions]
        
        # Append to raw data lists
        p_raw.append(positions)
        q_raw.append(rotations)
        t_raw.append(times)
    
    # Process data
    p_in, q_in, t_in = pre_process(p_raw, q_raw, t_raw, opt="savgol")
    p_out, q_out = compute_output(p_in, q_in, t_in)
    p_init, q_init, p_att, q_att = extract_state(p_in, q_in)
    p_in, q_in, p_out, q_out = rollout_list(p_in, q_in, p_out, q_out)
    
    # Initialize and train the quaternion dynamics model
    quat_obj = quat_class(q_in, q_out, q_att, dt, K_init=k_init, output_path=output_path)
    quat_obj.begin()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the trained model
    quat_obj._logOut()
    
    # Evaluate the model on a test trajectory
    step_size = dt
    q_test, gamma_test, omega_test = quat_obj.sim(q_init[0], step_size)
    # plot_gmm(p_in, quat_obj.gmm)
    # plot_gamma(gamma_test)
    # plot_omega(omega_test)
    # plt.show()
    print("Training and evaluation completed successfully.")
    
    return quat_obj, p_init, q_init, p_att, q_att, dt, q_in, q_out


def test_quat():
    """
    Test a trained quaternion dynamical system on trajectory data.
    
    This function:
    1. Load the quaternion dynamical system from a saved model
    2. Pick the first trajectory from the loaded data
    3. Simulate the quaternion dynamical system starting from the initial state
    4. Plot two trajectories:
        - The original trajectory
        - The simulated trajectory from the quaternion dynamical system
    """
    quat_obj = load_quat_model('models/quat_model.json')
    
    # Get dt and q_att from the loaded model
    dt = quat_obj.dt
    q_att = quat_obj.q_att
    
    # 2. Pick the first trajectory from the loaded data
    first_traj = eef_traj_data[4]
    first_pos = first_traj[:, :3]
    first_quat = first_traj[:, 3:7]
    
    # Convert quaternions to Rotation objects
    first_rot = [R.from_quat(quat) for quat in first_quat]
    
    # Create time vector based on dt
    t = np.arange(0, len(first_pos) * dt, dt)[:len(first_pos)]
    
    # 3. Simulate the quaternion dynamical system starting from the initial state
    # Specify the number of steps to match the length of the original trajectory
    num_steps = len(first_rot)
    q_sim, gamma_sim, omega_sim = quat_obj.sim(R.from_quat(first_quat[0]), dt, steps=num_steps)
    
    # 4. Plot the original and simulated trajectories
    fig = plt.figure(figsize=(12, 10))
    
    # Convert quaternions to Euler angles (in radians)
    # Plot Euler angles components over time
    ax1 = fig.add_subplot(211)
    ax1.set_title('Euler Angles Over Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (degrees)')
    
    # Convert original trajectory quaternions to Euler angles
    euler_seq = 'xyz'  # 'xyz' corresponds to roll, pitch, yaw
    original_euler = np.array([rot.as_euler(euler_seq) for rot in first_rot])
    
    # Unwrap angles to avoid discontinuities
    original_euler_unwrapped = np.unwrap(original_euler, axis=0)
    
    # Convert from radians to degrees
    original_euler_degrees = np.degrees(original_euler_unwrapped)
    
    ax1.plot(t, original_euler_degrees[:, 0], 'r-', label='Original Roll (x)')
    ax1.plot(t, original_euler_degrees[:, 1], 'g-', label='Original Pitch (y)')
    ax1.plot(t, original_euler_degrees[:, 2], 'b-', label='Original Yaw (z)')
    
    # Convert simulated trajectory quaternions to Euler angles
    sim_euler = np.array([rot.as_euler(euler_seq) for rot in q_sim])
    
    # Unwrap angles to avoid discontinuities
    sim_euler_unwrapped = np.unwrap(sim_euler, axis=0)
    
    # Convert from radians to degrees
    sim_euler_degrees = np.degrees(sim_euler_unwrapped)
    
    t_sim = np.arange(0, len(q_sim) * dt, dt)[:len(q_sim)]
    ax1.plot(t_sim, sim_euler_degrees[:, 0], 'r--', label='Simulated Roll (x)')
    ax1.plot(t_sim, sim_euler_degrees[:, 1], 'g--', label='Simulated Pitch (y)')
    ax1.plot(t_sim, sim_euler_degrees[:, 2], 'b--', label='Simulated Yaw (z)')
    
    # Plot attractor quaternion (q_att) as horizontal lines
    # Convert q_att to Euler angles
    q_att_euler = q_att.as_euler(euler_seq)
    q_att_degrees = np.degrees(q_att_euler)
    
    # Plot horizontal lines for each component
    ax1.axhline(y=q_att_degrees[0], color='r', linestyle=':', label='Target Roll (x)')
    ax1.axhline(y=q_att_degrees[1], color='g', linestyle=':', label='Target Pitch (y)')
    ax1.axhline(y=q_att_degrees[2], color='b', linestyle=':', label='Target Yaw (z)')
    
    ax1.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("Quaternion dynamical system testing completed successfully.")
    
    return quat_obj, q_sim, gamma_sim, omega_sim


if __name__ == "__main__":
    # Load data
    # Load all eef_traj files from the specified directory
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
    
    print(f"Total number of eef_traj files loaded: {len(eef_traj_data)}")
    train_quat()
    # test_quat()
