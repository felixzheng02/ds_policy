import time
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import equinox as eqx
import numpy as np
from jaxopt import OSQP
import matplotlib.animation as animation
import os
import sys

# Add the root directory to Python path
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)

from load_tools import load_data
from node_clf import Func_rot, NeuralODE_rot
from node_policy import NODEPolicy


def sphere_obstacle_constraint(
    state, center, radius, safety_margin=0.1
):  # TODO: not used for now
    """
    Compute the distance and gradient to a sphere obstacle.

    Args:
        state: Current state (position) of the system
        center: Center coordinates of the sphere obstacle
        radius: Radius of the sphere obstacle
        safety_margin: Additional safety distance from obstacle surface

    Returns:
        distance: Distance to obstacle surface (negative inside obstacle)
        gradient: Gradient of the distance function
    """
    diff = state - center
    dist = jnp.linalg.norm(diff)
    # Distance to obstacle surface (negative inside obstacle)
    distance = dist - (radius + safety_margin)
    # Gradient of the distance function
    gradient = diff / (dist + 1e-6)  # Add small epsilon to avoid division by zero
    return distance, gradient


def check_stagnation(velocity_history, window_size=10, threshold=1e-2):
    """
    Check if the trajectory has become stagnant by looking at recent velocities.

    Args:
        velocity_history: List of velocity vectors
        window_size: Number of recent steps to check
        threshold: Minimum average velocity magnitude to consider as moving

    Returns:
        bool: True if trajectory is stagnant
    """
    if len(velocity_history) < window_size:
        return False

    recent_velocities = velocity_history[-window_size:]
    cumulated_velocity = np.sum(recent_velocities)
    return cumulated_velocity < threshold


def follow_trajectory(
    model_path,
    x,
    number_of_trajectories=10,
    dt=0.01,
    n_steps=1000,
    save_path="trajectories.npz",
    key=None,
    switch=False,
    obstacle=None,
    backtrack_steps=0,
):
    """
    Follow a predefined target trajectory using the trained model with CLF-based control.
    Tests multiple initial points and saves their trajectories.

    Args:
        model_path: Path to the saved model file
        x: Training data to determine bounds
        target_trajs: Target trajectories to follow (L, M, 3) where L is number of trajectories, M is points per trajectory
        number_of_trajectories: Number of different initial points to test
        dt: Time step for integration
        n_steps: Number of simulation steps
        save_path: Path to save the generated trajectories
        key: Random key for initialization
        switch: If True, allows switching between trajectories based on closest point. If False, sticks to initially closest trajectory.
        obstacle: Dict containing sphere obstacle parameters {center: array, radius: float} or None
        backtrack_steps: Number of steps to backtrack on collision
    """
    policy = NODEPolicy(model_path, x, dt, switch, backtrack_steps, key)

    # Store all trajectories
    all_trajectories = []

    # Get min and max bounds from data for each dimension
    x_min = jnp.min(x[..., 0])
    x_max = jnp.max(x[..., 0])
    y_min = jnp.min(x[..., 1])
    y_max = jnp.max(x[..., 1])
    z_min = jnp.min(x[..., 2])
    z_max = jnp.max(x[..., 2])

    # Generate multiple trajectories with different initial points
    for traj_idx in range(number_of_trajectories):
        # Generate new key for this trajectory
        if key is None:
            key = jrandom.PRNGKey(int(time.time()))
        key, subkey = jrandom.split(key)

        # Initialize state randomly within the bounds of the data
        current_state = jnp.array(
            [
                jrandom.uniform(subkey, (), minval=x_min, maxval=x_max),
                jrandom.uniform(subkey, (), minval=y_min, maxval=-0.15),
                jrandom.uniform(subkey, (), minval=z_min, maxval=z_max),
            ]
        )

        # Storage for trajectory data
        actual_traj = [current_state]
        recovered_traj_idx = []
        target_indices = []
        velocity_history = []
        removed_sample_traj_indices = {}  # {time_step: removed_traj_idx}
        target_trajs = x.copy()  # Make a copy to modify

        # Main simulation loop
        i = 0
        while i < n_steps:
            """
            if obstacle is not None:
                # collision: bool = check_obstacle_collision(current_state, obstacle)
                collision = i == 50 # TODO: test
                if collision:
                    removed_sample_traj_indices[i] = current_target_idx
                    target_trajs = jnp.delete(target_trajs, current_target_idx, axis=0)
                    flattened_trajs = target_trajs.reshape(-1, 3)
                    num_trajs, points_per_traj, _ = target_trajs.shape

                    backtrack_steps = min(backtrack_steps, len(velocity_history))
                    current_state, actual_traj, velocity_history, recovered_traj_idx, target_indices, i = recover_from_collision(
                        current_state, actual_traj, recovered_traj_idx, velocity_history, backtrack_steps, dt, i, target_indices, num_trajs
                    )
                    if not switch:
                        # Find closest point across all trajectories
                        distances = jnp.linalg.norm(flattened_trajs - current_state, axis=1)
                        closest_flat_idx = jnp.argmin(distances)
                        fixed_target_idx = closest_flat_idx // points_per_traj
            """

            x_dot = policy.get_x_dot(current_state)

            target_indices.append(policy.ref_traj_idx)
            velocity_history.append(x_dot)  # Store velocity
            current_state = current_state + dt * x_dot

            # Store current state
            actual_traj.append(current_state)

            # Check for stagnation
            if check_stagnation(velocity_history, window_size=100, threshold=5e-4):
                print(f"Trajectory {traj_idx} stopped due to stagnation at step {i}")
                break

            i += 1

        actual_traj = jnp.array(actual_traj)
        recovered_traj_idx = jnp.array(recovered_traj_idx)
        target_indices = jnp.array(target_indices)
        velocity_history = jnp.array(velocity_history)
        all_trajectories.append(
            (
                actual_traj,
                recovered_traj_idx,
                target_indices,
                removed_sample_traj_indices,
            )
        )

    # L: number of trajectories
    # M: points per trajectory
    # N: state dimension
    # trajectories: Array of shape (L, M, N)
    trajectories = np.array([traj for traj, _, _, _ in all_trajectories], dtype=object)
    # recovered_traj_idx: Array of shape (L, recovered_steps)
    recovered_traj_idx = np.array(
        [recovered_traj_idx for _, recovered_traj_idx, _, _ in all_trajectories],
        dtype=object,
    )
    # target_indices: Array of shape (L, M-recovered_steps)
    target_indices = np.array(
        [indices for _, _, indices, _ in all_trajectories], dtype=object
    )
    # removed_indices: Array of shape (L, {timestep: trajectory_idx})
    removed_indices = np.array(
        [removed_indices for _, _, _, removed_indices in all_trajectories], dtype=object
    )

    # Save trajectories and target_trajs to file
    np.savez(
        save_path,
        trajectories=trajectories,
        recovered_traj_idx=recovered_traj_idx,
        target_indices=target_indices,
        removed_indices=removed_indices,
        target_trajs=x,
        allow_pickle=True,
    )

    return all_trajectories


def check_obstacle_collision(state, obstacle):
    """
    Check if a trajectory enters the obstacle region.

    Args:
        state: Array of points (N, 3)
        obstacle: Dict with 'center' and 'radius'

    Returns:
        collisions: Array of booleans indicating collision at each point
    """
    center = obstacle["center"]
    radius = obstacle["radius"]

    # Calculate distances from each point to obstacle center
    distance = np.linalg.norm(state - center, axis=1)

    # Check which points are inside the obstacle (distance < radius)
    return distance < radius


def recover_from_collision(
    current_state,
    actual_traj,
    recovered_traj_idx,
    velocity_history,
    backtrack_steps,
    dt,
    i,
    target_indices,
    num_trajs,
):
    """
    Recover from a collision by reverting to a previous state and clearing recent velocity history.
    TODO: this is not adaptive, might need to train a reverse node policy.

    Args:
        state: Current state of the system
        actual_traj: Actual trajectory
        recovered_traj_idx: Indices of frames where recovery occurred
        velocity_history: Velocity history
        backtrack_steps: Number of steps to backtrack
        dt: Time step
        i: Current time step
        target_indices: List tracking which target trajectory is being followed
        num_trajs: Total number of trajectories

    Returns:
        state: Recovered state
        actual_traj: Recovered trajectory
        velocity_history: Updated velocity history
        recovered_traj_idx: Updated recovery indices
        target_indices: Updated target indices
        i: Updated time step
    """
    # Use num_trajs + 1 as a special index to indicate recovery trajectory
    # This index doesn't correspond to any real trajectory
    special_recovery_idx = num_trajs

    for j in range(backtrack_steps):
        current_state = current_state - dt * velocity_history.pop()
        actual_traj.append(current_state)
        recovered_traj_idx.append(i)
        target_indices.append(special_recovery_idx)  # Add special index
        i += 1

    return (
        current_state,
        actual_traj,
        velocity_history,
        recovered_traj_idx,
        target_indices,
        i,
    )


def visualize_follow_trajectory(
    trajectories_path="trajectories.npz",
    output_path="trajectory_animation.mp4",
    obstacle=None,
):
    """
    Visualize pre-computed trajectories following target trajectories.

    Args:
        trajectories_path: Path to the .npz file containing trajectories and target data
        output_path: Path to save the animation. If None, displays the animation instead.
        obstacle: Dict containing sphere obstacle parameters {center: array, radius: float} or None
    """
    # Load data
    data = np.load(trajectories_path, allow_pickle=True)

    # all_trajectories is a list of tuples, where each tuple contains:
    # 1. current_traj: array of shape (n_steps, 3) containing trajectory points
    # 2. recovered_idx: array of indices where collision recovery occurred
    # 3. target_indices: array of indices showing which target trajectory was followed at each step
    # 4. removed_indices: dict mapping {timestep -> trajectory_idx} showing when trajectories were removed due to collisions
    all_trajectories = list(
        zip(
            data["trajectories"],
            data["recovered_traj_idx"],
            data["target_indices"],
            data["removed_indices"],
        )
    )
    # target_trajs: array of shape (n_trajectories, n_points, 3) containing the original trajectories to follow
    target_trajs = data["target_trajs"]

    # Setup plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Initialize with a single hidden point
    dummy_point = np.array([[0, 0, 0]])

    # Plot target trajectories
    target_lines = []
    for i, traj in enumerate(target_trajs):
        (line,) = ax.plot3D(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            "b-",
            alpha=0.05,
            linewidth=1,
            label="Sample Trajectory" if i == 0 else None,
        )
        target_lines.append(line)

    # Initialize plot elements with dummy points
    (generated_line,) = ax.plot3D(
        [], [], [], "#3A7D44", linewidth=3, label="Generated Trajectory"
    )
    (recovered_line,) = ax.plot3D(
        [], [], [], "r-", linewidth=2, label="Recovery Trajectory"
    )
    start_point = ax.scatter3D([], [], [], c="red", s=100, label="Start Point")
    (followed_line,) = ax.plot3D(
        [], [], [], "g-", linewidth=1, alpha=1.0, label="Target Trajectory"
    )

    # Set axis limits based on data
    all_points = np.vstack([traj for traj in target_trajs])
    margin = 0.1  # Add 10% margin
    x_range = np.max(all_points[:, 0]) - np.min(all_points[:, 0])
    y_range = np.max(all_points[:, 1]) - np.min(all_points[:, 1])
    z_range = np.max(all_points[:, 2]) - np.min(all_points[:, 2])
    ax.set_xlim(
        [
            np.min(all_points[:, 0]) - margin * x_range,
            np.max(all_points[:, 0]) + margin * x_range,
        ]
    )
    ax.set_ylim(
        [
            np.min(all_points[:, 1]) - margin * y_range,
            np.max(all_points[:, 1]) + margin * y_range,
        ]
    )
    ax.set_zlim(
        [
            np.min(all_points[:, 2]) - margin * z_range,
            np.max(all_points[:, 2]) + margin * z_range,
        ]
    )

    def update(frame):
        # Calculate which trajectory we're on and the frame within that trajectory
        total_frames = 0
        current_traj_idx = 0
        frame_in_current_traj = frame

        for idx, (traj, recovered_idx, indices, removed_indices) in enumerate(
            all_trajectories
        ):
            if frame_in_current_traj >= len(traj):
                frame_in_current_traj -= len(traj)
                current_traj_idx += 1
            else:
                break

        # If we've shown all trajectories, hide everything
        if current_traj_idx >= len(all_trajectories):
            generated_line.set_data([], [])
            generated_line.set_3d_properties([])
            recovered_line.set_data([], [])
            recovered_line.set_3d_properties([])
            start_point._offsets3d = ([], [], [])
            followed_line.set_data([], [])
            followed_line.set_3d_properties([])
            for line in target_lines:
                line.set_alpha(0.05)
                line.set_color("blue")
            return (
                generated_line,
                recovered_line,
                start_point,
                followed_line,
                *target_lines,
            )

        current_traj, recovered_idx, current_indices, removed_indices = (
            all_trajectories[current_traj_idx]
        )

        # Get list of removed trajectories up to current frame
        current_frame_removed = [
            idx
            for time_step, idx in removed_indices.items()
            if time_step <= frame_in_current_traj
        ]

        # Find the first recovery point in the current frame range
        if len(recovered_idx) > 0:
            recovered_idx = np.array(recovered_idx, dtype=np.int32)
            # Instead of just finding first recovery index, find all recovery segments
            recovery_mask = np.zeros(frame_in_current_traj + 1, dtype=bool)
            recovery_indices = recovered_idx[recovered_idx <= frame_in_current_traj]
            recovery_mask[recovery_indices] = True

            # Also mark frames as recovery based on the special target index
            if frame_in_current_traj < len(current_indices):
                for idx in range(frame_in_current_traj + 1):
                    if idx < len(current_indices) and current_indices[idx] >= len(
                        target_trajs
                    ):
                        recovery_mask[idx] = True
        else:
            recovery_mask = np.zeros(frame_in_current_traj + 1, dtype=bool)
            # Check for recovery based on special target index even if recovered_idx is empty
            if frame_in_current_traj < len(current_indices):
                for idx in range(frame_in_current_traj + 1):
                    if idx < len(current_indices) and current_indices[idx] >= len(
                        target_trajs
                    ):
                        recovery_mask[idx] = True

        # Split points into normal and recovery trajectories
        recovery_segments = []
        normal_segments = []

        # Process segments based on recovery mask
        i = 0
        while i <= frame_in_current_traj:
            if recovery_mask[i]:
                # Start of a recovery segment
                start = i
                while i <= frame_in_current_traj and recovery_mask[i]:
                    i += 1
                # Add this recovery segment
                recovery_segments.append(current_traj[start:i])
            else:
                # Start of a normal segment
                start = i
                while i <= frame_in_current_traj and not recovery_mask[i]:
                    i += 1
                # Add this normal segment
                normal_segments.append((start, i, current_traj[start:i]))

        # Determine whether to show normal points or recovery points
        show_normal = False
        normal_points = None

        # If we have recovery segments, only show normal segments that come after the last recovery
        if recovery_segments and recovery_indices.size > 0:
            last_recovery_idx = max(recovery_indices)
            post_recovery_segments = [
                segment
                for start, end, segment in normal_segments
                if start > last_recovery_idx
            ]

            if post_recovery_segments:
                # We have normal segments after recovery - show only these
                normal_points = np.vstack(post_recovery_segments)
                show_normal = True
        elif normal_segments:
            # No recovery segments (or none visible yet) - show all normal segments
            normal_points = np.vstack([segment for _, _, segment in normal_segments])
            show_normal = True

        # Update visualization based on what we're showing
        if show_normal:
            # Show normal trajectory
            generated_line.set_data(normal_points[:, 0], normal_points[:, 1])
            generated_line.set_3d_properties(normal_points[:, 2])
            generated_line.set_alpha(1)
            generated_line.set_color("#3A7D44")

            # Hide recovery trajectory
            recovered_line.set_data([], [])
            recovered_line.set_3d_properties([])
        else:
            # Hide normal trajectory
            generated_line.set_data([], [])
            generated_line.set_3d_properties([])

            # Show recovery trajectory (if any)
            if recovery_segments:
                recovery_points = np.vstack(recovery_segments)
                recovered_line.set_data(recovery_points[:, 0], recovery_points[:, 1])
                recovered_line.set_3d_properties(recovery_points[:, 2])
                recovered_line.set_alpha(1)
            else:
                recovered_line.set_data([], [])
                recovered_line.set_3d_properties([])

        # Update start point
        start_point._offsets3d = (
            [current_traj[0, 0]],
            [current_traj[0, 1]],
            [current_traj[0, 2]],
        )

        # Update target trajectory colors and followed points
        # Only show target trajectory if we're in normal mode (not recovery)
        if frame_in_current_traj < len(current_indices) and show_normal:
            current_target = current_indices[frame_in_current_traj]

            # Check if this is a recovery index (which equals num_trajectories)
            if current_target < len(target_trajs):
                # Update the followed points to match the current target trajectory
                target_traj = target_trajs[current_target]
                followed_line.set_data(target_traj[:, 0], target_traj[:, 1])
                followed_line.set_3d_properties(target_traj[:, 2])
                followed_line.set_color("#3A7D44")
                followed_line.set_alpha(1.0)

                # Update target trajectory line colors
                for i, line in enumerate(target_lines):
                    if i == current_target:
                        line.set_alpha(
                            0.05
                        )  # Hide the original line when being followed
                    elif i in current_frame_removed:
                        line.set_color("red")  # Show removed trajectories in red
                        line.set_alpha(0.05)  # Keep same opacity
                    else:
                        line.set_alpha(0.05)
                        line.set_color("blue")
            else:
                # This is a recovery trajectory, don't highlight any target trajectory
                followed_line.set_data([], [])
                followed_line.set_3d_properties([])

                # Reset all target lines to default appearance during recovery
                for i, line in enumerate(target_lines):
                    if i in current_frame_removed:
                        line.set_color("red")  # Show removed trajectories in red
                    else:
                        line.set_color("blue")
                    line.set_alpha(0.05)  # Keep all lines at low opacity
        else:
            # We're in recovery mode or past the end of indices, don't show any followed trajectory
            followed_line.set_data([], [])
            followed_line.set_3d_properties([])

            # Reset all target lines to default appearance during recovery
            for i, line in enumerate(target_lines):
                if i in current_frame_removed:
                    line.set_color("red")  # Show removed trajectories in red
                else:
                    line.set_color("blue")
                line.set_alpha(0.05)  # Keep all lines at low opacity

        return generated_line, recovered_line, start_point, followed_line, *target_lines

    # Calculate total number of frames needed
    total_frames = sum(len(traj) for traj, _, _, _ in all_trajectories)

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=50, blit=True
    )

    # Plot obstacle if present
    if obstacle is not None:
        # Create sphere
        center = obstacle["center"]
        radius = obstacle["radius"]

        # Create a sphere mesh
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot the sphere
        ax.plot_surface(x, y, z, color="gray", alpha=0.3)

    if output_path is None:
        plt.show()
        return

    # Save animation
    writer = animation.FFMpegWriter(fps=20, bitrate=3000)
    anim.save(output_path, writer=writer)
    plt.close()


if __name__ == "__main__":

    # Load data
    x, x_dot, x_att, x_init = load_data(
        "custom", show_plot=True, separate=True, shift=False
    )
    x = jnp.array(x, dtype=jnp.float32)

    model_path = "models/mlp_width256_depth5.eqx"

    # Example usage with obstacle:
    obstacle = {
        "center": jnp.array([0.0, -0.06, -0.06]),  # Center of the sphere
        "radius": 0.01,  # Radius of the sphere
    }

    if False:
        # Generate and save trajectories with obstacle
        follow_trajectory(
            model_path=model_path,
            x=x,
            number_of_trajectories=1,
            dt=0.01,
            n_steps=500,
            save_path="data/trajectories.npz",
            switch=True,
            obstacle=None,
            backtrack_steps=30,
        )

    if False:
        # Visualize the trajectories with obstacle
        visualize_follow_trajectory(
            "data/trajectories.npz", output_path=None, obstacle=None
        )
