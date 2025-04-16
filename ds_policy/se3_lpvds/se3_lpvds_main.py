import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from src.se3_class import se3_class
from src.util import plot_tools, load_tools, process_tools



'''Load data'''
p_raw, q_raw, t_raw, dt = load_tools.load_UMI() # p_raw[0]: (250, 3), q_raw[0]: list of 250 scipy Rotation objects, t_raw: list of 250 time values, dt: 0.02
# p_raw, q_raw, t_raw, dt = load_tools.load_clfd_dataset(task_id=0, num_traj=9, sub_sample=1)
# p_raw, q_raw, t_raw, dt = load_tools.load_demo_dataset()



'''Process data'''
p_in, q_in, t_in             = process_tools.pre_process(p_raw, q_raw, t_raw, opt= "savgol")
p_out, q_out                 = process_tools.compute_output(p_in, q_in, t_in)
p_init_list, q_init_list, p_att, q_att = process_tools.extract_state(p_in, q_in) # Capture lists
p_in_roll, q_in_roll, p_out_roll, q_out_roll     = process_tools.rollout_list(p_in, q_in, p_out, q_out) # Use rolled versions



'''Run lpvds'''
se3_obj = se3_class(p_in_roll, q_in_roll, p_out_roll, q_out_roll, p_att, q_att, dt, K_init=4)
se3_obj.begin()



'''Evaluate results'''
# Use one of the initial states for simulation
# p_init_sim = p_init_list[0]
p_init_sim = np.zeros((3, 1))
q_init_sim = R.from_quat(q_init_list[0].as_quat())
sim_step_size = 0.01
p_test, q_test, gamma_test, v_test, w_test = se3_obj.sim(p_init_sim, q_init_sim, step_size=sim_step_size)





'''Plot results'''
# plot_tools.plot_vel(p_test, w_test)
# plot_tools.plot_gmm(p_in_roll, se3_obj.gmm)
# plot_tools.plot_result(p_in_roll, p_test, q_test) # Original comparison plot
# plot_tools.plot_gamma(gamma_test)


print("Plotting time-based comparison...")

# Generate time vectors
# Assuming dt is constant for input data after processing/rollout
time_in = np.arange(len(p_in_roll)) * dt
time_test = np.arange(len(p_test)) * sim_step_size

# Convert orientations to Euler angles (XYZ convention, degrees)
q_in_euler = np.array([q.as_euler('xyz', degrees=True) for q in q_in_roll])
q_test_euler = np.array([q.as_euler('xyz', degrees=True) for q in q_test])

# Create figure and subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=False)
fig.suptitle('Input vs. Generated Trajectories Over Time')

# Plot Position (x, y, z)
axs[0, 0].plot(time_in, p_in_roll[:, 0], label='Input (p_in)', alpha=0.7)
axs[0, 0].plot(time_test, p_test[:, 0], label='Generated (p_test)', linestyle='--')
axs[0, 0].set_title('Position - X')
axs[0, 0].set_ylabel('X (m)')
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].plot(time_in, p_in_roll[:, 1], label='Input (p_in)', alpha=0.7)
axs[0, 1].plot(time_test, p_test[:, 1], label='Generated (p_test)', linestyle='--')
axs[0, 1].set_title('Position - Y')
axs[0, 1].set_ylabel('Y (m)')
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[0, 2].plot(time_in, p_in_roll[:, 2], label='Input (p_in)', alpha=0.7)
axs[0, 2].plot(time_test, p_test[:, 2], label='Generated (p_test)', linestyle='--')
axs[0, 2].set_title('Position - Z')
axs[0, 2].set_ylabel('Z (m)')
axs[0, 2].legend()
axs[0, 2].grid(True)

# Plot Orientation (Euler angles: roll, pitch, yaw)
axs[1, 0].plot(time_in, q_in_euler[:, 0], label='Input (q_in)', alpha=0.7)
axs[1, 0].plot(time_test, q_test_euler[:, 0], label='Generated (q_test)', linestyle='--')
axs[1, 0].set_title('Orientation - Roll (X-axis)')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Angle (degrees)')
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].plot(time_in, q_in_euler[:, 1], label='Input (q_in)', alpha=0.7)
axs[1, 1].plot(time_test, q_test_euler[:, 1], label='Generated (q_test)', linestyle='--')
axs[1, 1].set_title('Orientation - Pitch (Y-axis)')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Angle (degrees)')
axs[1, 1].legend()
axs[1, 1].grid(True)

axs[1, 2].plot(time_in, q_in_euler[:, 2], label='Input (q_in)', alpha=0.7)
axs[1, 2].plot(time_test, q_test_euler[:, 2], label='Generated (q_test)', linestyle='--')
axs[1, 2].set_title('Orientation - Yaw (Z-axis)')
axs[1, 2].set_xlabel('Time (s)')
axs[1, 2].set_ylabel('Angle (degrees)')
axs[1, 2].legend()
axs[1, 2].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

plt.show()