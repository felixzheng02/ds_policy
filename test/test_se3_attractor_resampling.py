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
option = "OpenDrawer-Op0"
task_name = "OpenDrawer"
# Assuming load_data is accessible and returns data in the expected format
# Note: ds_policy expects lists of numpy arrays for x, x_dot, quat, omega, gripper
# x, x_dot, q, omega, gripper = load_data(task_name, option, finger=True, transform_to_object_of_interest_frame=True, debug_on=False)
x = np.load(f"./trajectory_data/x_{option}.npy", allow_pickle=True)
x_dot = np.load(f"./trajectory_data/x_dot_{option}.npy", allow_pickle=True)
quat = np.load(f"./trajectory_data/quat_{option}.npy", allow_pickle=True)
omega = np.load(f"./trajectory_data/omega_{option}.npy", allow_pickle=True)
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
    K_candidates=[3], # Number of Gaussian components
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
    quat=quat, # Pass the loaded quaternions (q) to the 'quat' argument
    omega=omega,
    gripper=[],
    unified_config=unified_config,
    dt=dt, # Use the determined dt
)
print("Policy initialized.")

# --- 4. Visualization Function ---
def visualize_state(policy, current_att_pos, current_att_r, iteration, original_demos_x):
    """Visualize initial trajectories & attractor (iteration 0) and subsequent sampled goal poses."""

    if iteration == 0:
        # =======================
        # Figure 1: Demonstrations & Mean Attractor
        # =======================
        fig_demo = plt.figure(figsize=(8, 6))
        fig_demo.suptitle("Demonstrations & Mean Attractor", fontsize=14)

        ax_demo = fig_demo.add_subplot(111, projection="3d")

        # Plot original demonstrations
        for i, traj in enumerate(original_demos_x):
            ax_demo.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                "b-",
                alpha=0.3,
                linewidth=1.5,
                label="Original Demo" if i == 0 else "",
            )

        # Plot shifted demos for training (relative to mean attractor)
        try:
            shifted_pos, _ = policy._shift_trajs(current_att_pos, current_att_r)
            for i, traj in enumerate(shifted_pos):
                ax_demo.plot(
                    traj[:, 0],
                    traj[:, 1],
                    traj[:, 2],
                    "g-",
                    alpha=0.8,
                    linewidth=2,
                    label="Shifted Demo" if i == 0 else "",
                )
        except Exception as e:
            print(f"Warning: Could not get/plot shifted trajectories: {e}")

        # Plot attractor position
        ax_demo.plot(
            current_att_pos[0],
            current_att_pos[1],
            current_att_pos[2],
            "c*",
            markersize=15,
            label="Mean Attractor Pos",
        )

        # Plot mean attractor orientation axes
        axes_len_demo = 0.5
        try:
            mean_att_r = policy.se3_lpvds_attractor_generator.gaussian.q_att
        except AttributeError:
            mean_att_r = current_att_r

        try:
            att_axes = mean_att_r.apply(np.eye(3) * axes_len_demo)
            colors = ["r", "g", "b"]
            for i in range(3):
                ax_demo.quiver(
                    current_att_pos[0],
                    current_att_pos[1],
                    current_att_pos[2],
                    att_axes[i, 0],
                    att_axes[i, 1],
                    att_axes[i, 2],
                    color=colors[i],
                    length=axes_len_demo,
                    normalize=False,
                    linewidth=2.5,
                )
        except AttributeError:
            print("Warning: Mean attractor rotation object invalid, skipping orientation plot.")

        # Axis labels & limits
        ax_demo.set_xlabel("X")
        ax_demo.set_ylabel("Y")
        ax_demo.set_zlabel("Z")
        ax_demo.legend()

        # Auto-scale
        all_vis_pos = np.concatenate(original_demos_x, axis=0)
        x_min, x_max = all_vis_pos[:, 0].min(), all_vis_pos[:, 0].max()
        y_min, y_max = all_vis_pos[:, 1].min(), all_vis_pos[:, 1].max()
        z_min, z_max = all_vis_pos[:, 2].min(), all_vis_pos[:, 2].max()
        center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
        max_range = np.max([x_max - x_min, y_max - y_min, z_max - z_min]) * 1.2
        if max_range == 0:
            max_range = 1.0
        ax_demo.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
        ax_demo.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
        ax_demo.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)

        plt.tight_layout()
        plt.show(block=True)

    # =======================
    # Figure 2-n: Attractor Space & Sampled Goal Pose
    # Always shown (including iteration 0)
    # =======================
    fig_att = plt.figure(figsize=(7, 6))
    fig_att.suptitle("All End Poses (Positions & Orientations)", fontsize=14)

    ax_att = fig_att.add_subplot(111, projection="3d")

    # Plot end points / attractor space
    end_points = policy.se3_lpvds_attractor_generator.end_pts
    end_positions = np.array([p for p, _ in end_points])
    end_rotations = [r for _, r in end_points]

    ax_att.scatter(
        end_positions[:, 0],
        end_positions[:, 1],
        end_positions[:, 2],
        c="grey",
        s=50,
        alpha=0.6,
        label="Possible Attractors",
    )

    # Plot orientation axes for each end pose
    axes_len_each = 0.07
    colors_each = ["r", "g", "b"]
    for p_end, r_end in end_points:
        try:
            end_axes = r_end.apply(np.eye(3) * axes_len_each)
            for j in range(3):
                ax_att.quiver(
                    p_end[0],
                    p_end[1],
                    p_end[2],
                    end_axes[j, 0],
                    end_axes[j, 1],
                    end_axes[j, 2],
                    color=colors_each[j],
                    length=axes_len_each,
                    normalize=False,
                    linewidth=1.0,
                    alpha=0.6,
                )
        except AttributeError:
            pass

    # --- Highlight the currently sampled pose ---
    ax_att.plot(
        current_att_pos[0],
        current_att_pos[1],
        current_att_pos[2],
        "c*",
        markersize=18,
        label="Sampled Pose",
    )

    axes_len_sample = 0.25
    try:
        att_axes_sample = current_att_r.apply(np.eye(3) * axes_len_sample)
        colors_sample = ["r", "g", "b"]
        for i in range(3):
            ax_att.quiver(
                current_att_pos[0],
                current_att_pos[1],
                current_att_pos[2],
                att_axes_sample[i, 0],
                att_axes_sample[i, 1],
                att_axes_sample[i, 2],
                color=colors_sample[i],
                length=axes_len_sample,
                normalize=False,
                linewidth=2.5,
            )
    except AttributeError:
        print("Warning: Sampled attractor rotation invalid - cannot draw axes.")

    ax_att.set_xlabel("X")
    ax_att.set_ylabel("Y")
    ax_att.set_zlabel("Z")
    ax_att.legend()

    # Auto-scale based on attractor space
    all_att_pos = np.vstack([end_positions, current_att_pos])
    att_x_min, att_x_max = all_att_pos[:, 0].min(), all_att_pos[:, 0].max()
    att_y_min, att_y_max = all_att_pos[:, 1].min(), all_att_pos[:, 1].max()
    att_z_min, att_z_max = all_att_pos[:, 2].min(), all_att_pos[:, 2].max()
    att_center = np.array(
        [
            (att_x_min + att_x_max) / 2,
            (att_y_min + att_y_max) / 2,
            (att_z_min + att_z_max) / 2,
        ]
    )
    att_max_range = (
        np.max([att_x_max - att_x_min, att_y_max - att_y_min, att_z_max - att_z_min]) * 1.4
    )
    if att_max_range == 0:
        att_max_range = 1.0
    ax_att.set_xlim(att_center[0] - att_max_range / 2, att_center[0] + att_max_range / 2)
    ax_att.set_ylim(att_center[1] - att_max_range / 2, att_center[1] + att_max_range / 2)
    ax_att.set_zlim(att_center[2] - att_max_range / 2, att_center[2] + att_max_range / 2)

    plt.tight_layout()
    plt.show(block=True)


# --- 5. Show Figures ---
# Use mean of end positions as representative attractor position
end_points = policy.se3_lpvds_attractor_generator.end_pts
current_att_pos = np.mean(np.array([p for p, _ in end_points]), axis=0)

# Mean orientation from Gaussian, if available
try:
    current_att_r = policy.se3_lpvds_attractor_generator.gaussian.q_att
except AttributeError:
    current_att_r = end_points[0][1]

# Visualize initial figures (iteration 0)
visualize_state(policy, current_att_pos, current_att_r, 0, x)

# --- 6. Resampling Loop ---
# Show additional sampled goal poses after the initial figures
num_resamples = 0  # Feel free to adjust
for i in range(num_resamples):
    print(f"--- Resampling {i+1}/{num_resamples} ---")
    res_pos, res_r = policy.se3_lpvds_attractor_generator.sample()
    visualize_state(policy, res_pos, res_r, i + 1, x)
