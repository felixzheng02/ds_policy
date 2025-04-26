import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import time
import os

# Ensure the ds_policy package is found (adjust path if necessary)
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ds_policy.policy import DSPolicy, UnifiedModelConfig
from ds_policy.ds_utils import load_data # Added import

# --- 1. Load Real Data ---
print("Loading real data...")
# Example: Load data similar to test_ds_policy.py
option = "OpenSingleDoor_MoveTowards_option"
task_name = "OpenSingleDoor"
# Assuming load_data is accessible and returns data in the expected format
# Note: ds_policy expects lists of numpy arrays for x, x_dot, quat, omega, gripper
x, x_dot, q, omega, gripper = load_data(task_name, option, finger=True, transform_to_object_of_interest_frame=True, debug_on=False)
# Determine dt from the loaded data if possible, otherwise use a default
# Example: inferring dt from timestamps if available, or assuming a fixed rate
# If load_data doesn't provide timestamps or dt, you might need to estimate it
# or use a value consistent with the data source (e.g., 1/60 for 60Hz)
dt = 1/60 # Assuming 60Hz, adjust if needed based on loaded data
print(f"Data loading complete. Loaded {len(x)} trajectories for task '{task_name}', option '{option}'.")
print(f"Using dt = {dt:.4f}")

# --- 2. Define Config ---
unified_config = UnifiedModelConfig(
    mode="se3_lpvds",
    K_candidates=[1], # Number of Gaussian components
    attractor_resampling_mode="Gaussian", # How to pick the next attractor
    # Parameters for simple PD controller near target (optional)
    enable_simple_ds_near_target=False,
    simple_ds_pos_threshold=0.03,
    simple_ds_ori_threshold=0.1,
    K_pos=1.0,
    K_ori=1.0,
)

# --- 3. Instantiate Policy ---
print("Initializing DSPolicy (SE3-LPVDS)... This may take a moment for initial training.")
policy = DSPolicy(
    x=x,
    x_dot=x_dot,
    quat=q, # Pass the loaded quaternions (q) to the 'quat' argument
    omega=omega,
    gripper=gripper,
    unified_config=unified_config,
    dt=dt, # Use the determined dt
)
print("Policy initialized.")

# --- 4. Visualization Function ---
def visualize_state(policy, current_att_pos, current_att_r, iteration, original_demos_x):
    """Visualizes trajectories, attractor, vector field, and attractor space."""
    # Increase figure width to accommodate three plots
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f"Iteration {iteration}", fontsize=16)

    # --- Plot 1: Trajectories and Attractor ---
    ax1 = fig.add_subplot(131, projection='3d') # Changed to 131
    ax1.set_title(f"Trajectories & Current Attractor")

    # Original Demo (use the input data)
    for i, traj in enumerate(original_demos_x):
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', alpha=0.3, linewidth=1.5, label='Original Demo' if i == 0 else "")

    # Shifted Demos (recalculate based on current attractor)
    try:
        # Use policy's internal method to get shifted trajectories
        shifted_pos, shifted_rot_list_of_lists = policy._shift_trajs(current_att_pos, current_att_r)
        # The shifted trajectories are used for internal training, plot them for visualization
        for i, traj in enumerate(shifted_pos):
             ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'g-', alpha=0.8, linewidth=2, label='Shifted Demo (for training)' if i == 0 else "")
    except Exception as e:
        print(f"Warning: Could not get/plot shifted trajectories: {e}")
        shifted_pos = original_demos_x # Fallback for bounds calculation

    # Attractor Position
    ax1.plot(current_att_pos[0], current_att_pos[1], current_att_pos[2], 'c*', markersize=15, label='Current Attractor')

    # Attractor Orientation (as axes)
    axes_len = 0.25 # Increased length further from 0.15
    try:
        att_axes = current_att_r.apply(np.eye(3) * axes_len)
        colors = ['r', 'g', 'b'] # X, Y, Z
        for i in range(3):
            ax1.quiver(current_att_pos[0], current_att_pos[1], current_att_pos[2],
                       att_axes[i, 0], att_axes[i, 1], att_axes[i, 2],
                       color=colors[i], length=axes_len, normalize=False, linewidth=2.5) # Increased linewidth again
    except AttributeError:
        print("Warning: Attractor rotation object invalid, skipping orientation plot.")

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()
    # Calculate dynamic plot limits based on shifted trajectories
    try:
        all_vis_pos = np.concatenate(shifted_pos + original_demos_x, axis=0)
    except NameError: # Handle case where shifted_pos failed
        all_vis_pos = np.concatenate(original_demos_x, axis=0)
    x_min, x_max = all_vis_pos[:, 0].min(), all_vis_pos[:, 0].max()
    y_min, y_max = all_vis_pos[:, 1].min(), all_vis_pos[:, 1].max()
    z_min, z_max = all_vis_pos[:, 2].min(), all_vis_pos[:, 2].max()
    center = np.array([(x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2])
    max_range = np.max([x_max-x_min, y_max-y_min, z_max-z_min]) * 1.2 # Add padding
    if max_range == 0: max_range = 1.0 # Avoid zero range
    ax1.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
    ax1.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
    ax1.set_zlim(center[2] - max_range/2, center[2] + max_range/2)


    # --- Plot 2: Position Vector Field ---
    ax2 = fig.add_subplot(132, projection='3d') # Changed to 132
    ax2.set_title(f"Position Vector Field")

    # Define sampling grid based on shifted trajectory bounds
    padding = 0.1
    x_range = max(0.1, x_max - x_min)
    y_range = max(0.1, y_max - y_min)
    z_range = max(0.1, z_max - z_min)
    plot_x_min = x_min - padding * x_range; plot_x_max = x_max + padding * x_range
    plot_y_min = y_min - padding * y_range; plot_y_max = y_max + padding * y_range
    plot_z_min = z_min - padding * z_range; plot_z_max = z_max + padding * z_range

    grid_points = 6 # Fewer points for faster visualization
    x_grid = np.linspace(plot_x_min, plot_x_max, grid_points)
    y_grid = np.linspace(plot_y_min, plot_y_max, grid_points)
    z_grid = np.linspace(plot_z_min, plot_z_max, grid_points)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    W = np.zeros_like(Z)

    # Use a fixed representative quaternion for sampling the field (attractor orientation)
    # Ensure it's in xyzw format for get_action
    try:
        rep_quat = current_att_r.as_quat() # xyzw
        if len(rep_quat) != 4: raise ValueError("Invalid quaternion from attractor")
    except Exception as e:
        print(f"Warning: Cannot use attractor orientation ({e}), using identity quat [0,0,0,1].")
        rep_quat = np.array([0., 0., 0., 1.]) # Default to identity xyzw

    print("Calculating vector field...")
    field_calculation_start = time.time()
    for i in range(grid_points):
        for j in range(grid_points):
            for k in range(grid_points):
                pos = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                try:
                    state = np.concatenate([pos, rep_quat])
                    # Call get_action, disable CLF for raw dynamics if desired
                    # SE3-LPVDS get_action doesn't use CLF currently in the main path
                    action = policy.get_action(state, clf=False)
                    vel = action[:3]
                    U[i, j, k] = vel[0]
                    V[i, j, k] = vel[1]
                    W[i, j, k] = vel[2]
                except Exception as e: # Catch potential errors during policy call
                    # print(f"Error getting action at {pos}: {e}") # Can be noisy
                    U[i, j, k], V[i, j, k], W[i, j, k] = 0, 0, 0 # Indicate error point
    print(f"Vector field calculation took {time.time() - field_calculation_start:.2f}s")

    # Plot vector field arrows
    norm = np.sqrt(U**2 + V**2 + W**2) + 1e-9 # Add epsilon for stability
    quiver_len_factor = 0.04 * np.mean([plot_x_max-plot_x_min, plot_y_max-plot_y_min, plot_z_max-plot_z_min]) # Adjust length based on plot scale

    ax2.quiver(
        X, Y, Z,
        U / norm, V / norm, W / norm, # Normalized vectors for direction
        length=quiver_len_factor, normalize=False, color="purple", alpha=0.7, linewidth=1.0
    )

    # Plot shifted demos for context
    try:
        for i, traj in enumerate(shifted_pos):
             ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'g-', alpha=0.5, linewidth=1.5, label='Shifted Demo' if i == 0 else "")
    except Exception as e:
        print(f"Warning: Could not plot shifted trajectories in vector field plot: {e}")

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_xlim(plot_x_min, plot_x_max)
    ax2.set_ylim(plot_y_min, plot_y_max)
    ax2.set_zlim(plot_z_min, plot_z_max)
    if any(ax2.get_legend_handles_labels()[1]): # Add legend only if labels exist
        ax2.legend()

    # --- Plot 3: Attractor Space ---
    ax3 = fig.add_subplot(133, projection='3d') # New subplot 133
    ax3.set_title("Attractor Space & Sampled Attractor")

    # Get end points from the policy's attractor generator
    end_points = policy.se3_lpvds_attractor_generator.end_pts
    end_positions = np.array([p for p, r in end_points])
    end_rotations = [r for p, r in end_points]

    # Plot all possible end points (positions)
    ax3.scatter(end_positions[:, 0], end_positions[:, 1], end_positions[:, 2], c='grey', s=50, alpha=0.6, label='Possible Attractors (Demo End Pts)')

    ax3.plot(current_att_pos[0], current_att_pos[1], current_att_pos[2], 'c*', markersize=20, label='Current Sampled Attractor')

    # Optional: Plot orientations as quivers for all end points
    axes_len_end = 0.03 # Smaller axes for end points
    colors_end = ['grey'] * 3 # Fainter colors for end point axes
    for i in range(len(end_positions)):
        try:
            end_axes = end_rotations[i].apply(np.eye(3) * axes_len_end)
            for j in range(3):
                ax3.quiver(end_positions[i, 0], end_positions[i, 1], end_positions[i, 2],
                           end_axes[j, 0], end_axes[j, 1], end_axes[j, 2],
                           color=colors_end[j], length=axes_len_end, normalize=False, linewidth=1.0, alpha=0.5)
        except AttributeError:
            pass # Skip if rotation object is invalid

    # Plot current attractor orientation
    axes_len_curr = 0.25 # Increased length further from 0.15
    colors_curr = ['r', 'g', 'b']
    try:
        att_axes = current_att_r.apply(np.eye(3) * axes_len_curr)
        for i in range(3):
            ax3.quiver(current_att_pos[0], current_att_pos[1], current_att_pos[2],
                       att_axes[i, 0], att_axes[i, 1], att_axes[i, 2],
                       color=colors_curr[i], length=axes_len_curr, normalize=False, linewidth=3.0) # Increased linewidth again
    except AttributeError:
        print("Warning: Attractor rotation object invalid, skipping current orientation plot in ax3.")

    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.legend()

    # Set limits for ax3 based on end points and current attractor
    all_attractor_pos = np.vstack([end_positions, current_att_pos])
    att_x_min, att_x_max = all_attractor_pos[:, 0].min(), all_attractor_pos[:, 0].max()
    att_y_min, att_y_max = all_attractor_pos[:, 1].min(), all_attractor_pos[:, 1].max()
    att_z_min, att_z_max = all_attractor_pos[:, 2].min(), all_attractor_pos[:, 2].max()
    att_center = np.array([(att_x_min+att_x_max)/2, (att_y_min+att_y_max)/2, (att_z_min+att_z_max)/2])
    att_max_range = np.max([att_x_max-att_x_min, att_y_max-att_y_min, att_z_max-att_z_min]) * 1.4 # Extra padding
    if att_max_range == 0: att_max_range = 1.0 # Avoid zero range

    ax3.set_xlim(att_center[0] - att_max_range/2, att_center[0] + att_max_range/2)
    ax3.set_ylim(att_center[1] - att_max_range/2, att_center[1] + att_max_range/2)
    ax3.set_zlim(att_center[2] - att_max_range/2, att_center[2] + att_max_range/2)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    # Change to blocking show for interaction and waiting
    plt.show(block=True)


# --- 5. Resampling Loop ---
num_resamples = 5 # Number of times to resample and visualize
resample_interval = 2.0 # Seconds between resamples

for i in range(num_resamples):
    print(f"--- Visualization {i} ---")
    current_pos, current_r = policy.se3_lpvds_attractor_generator.sample()
    print(f"Current attractor: Pos={np.round(current_pos, 3)}, Quat={np.round(current_r.as_quat(), 3)}")
    visualize_state(policy, current_pos, current_r, i, x) # Pass original demos for reference

    # # Check if it's the last iteration before trying to resample
    # if i < num_resamples - 1:
    #     print(f"--- Resampling Attractor {i+1}/{num_resamples} ---")
    #     resample_start_time = time.time()
    #     # Call the internal resampling method which includes retraining
    #     policy.se3_lpvds_attractor_generator.sample()
    #     resample_duration = time.time() - resample_start_time
    #     new_pos = policy.model.p_att
    #     new_quat = policy.model.q_att.as_quat()
    #     print(f"Resampling and retraining took {resample_duration:.2f}s")
    #     print(f"New attractor: Pos={np.round(new_pos, 3)}, Quat={np.round(new_quat, 3)}")

    # Remove the sleep, plt.show(block=True) handles waiting
    # time.sleep(max(0, resample_interval - resample_duration))



# The final plt.show() is now implicitly handled by the last call to visualize_state
# plt.show() # Keep final plot open until manually closed
