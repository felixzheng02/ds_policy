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
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from load_tools import load_data
from node_clf import Func_rot, NeuralODE_rot

def follow_trajectory(model_path, x, target_trajs, number_of_trajectories=10, dt=0.01, n_steps=1000, save_path="trajectories.npz", key=None, switch=False):
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
    """
    # Create a template model with same structure
    _, _, data_size = x.shape
    if key is None:
        key = jrandom.PRNGKey(int(time.time()))
    # Extract width and depth from model path string
    # Assuming format like "mlp_width64_depth1.eqx"
    model_name = os.path.basename(model_path)
    width_str = model_name.split("width")[1].split("_")[0]
    depth_str = model_name.split("depth")[1].split(".")[0]
    width_size = int(width_str)
    depth = int(depth_str)
    template_model = NeuralODE_rot(data_size, width_size, depth, key=key)
    
    # Load the saved model
    model = eqx.tree_deserialise_leaves(model_path, template_model)

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
        key, subkey = jrandom.split(key)
        
        # Initialize state randomly within the bounds of the data
        current_state = jnp.array([
            jrandom.uniform(subkey, (), minval=x_min, maxval=x_max),
            jrandom.uniform(subkey, (), minval=y_min, maxval=y_max),
            jrandom.uniform(subkey, (), minval=z_min, maxval=z_max)
        ])
        
        # Storage for actual trajectory and target indices
        actual_traj = [current_state]
        target_indices = []  # Store which trajectory was followed at each step
        
        # CLF parameters
        alpha_V = 50.0  # CLF convergence rate
        
        # Initialize OSQP solver
        qp = OSQP()
        
        # Reshape target_trajs to (L*M, N) for easier distance computation
        num_trajs, points_per_traj, _ = target_trajs.shape
        flattened_trajs = target_trajs.reshape(-1, 3)
        
        # Find initial closest trajectory if not switching
        if not switch:
            # Find closest point across all trajectories
            distances = jnp.linalg.norm(flattened_trajs - current_state, axis=1)
            closest_flat_idx = jnp.argmin(distances)
            fixed_target_idx = closest_flat_idx // points_per_traj
        
        # Main simulation loop
        for i in range(n_steps):
            if switch:
                # Find closest point across all trajectories
                distances = jnp.linalg.norm(flattened_trajs - current_state, axis=1)
                closest_flat_idx = jnp.argmin(distances)
                current_target_idx = closest_flat_idx // points_per_traj
                closest_idx = closest_flat_idx % points_per_traj
                
            else:
                # Stay with initially closest trajectory
                current_target_idx = fixed_target_idx
                # Find closest point only within the fixed trajectory
                traj_points = target_trajs[current_target_idx]
                distances = jnp.linalg.norm(traj_points - current_state, axis=1)
                closest_idx = jnp.argmin(distances)
            
            target_indices.append(current_target_idx)
            
            # Get reference state from the identified trajectory
            ref_state = target_trajs[current_target_idx, closest_idx]
            
            # Look ahead on the current trajectory for smoother tracking
            lookahead = 5
            next_idx = min(closest_idx + lookahead, points_per_traj - 1)
            ref_next = target_trajs[current_target_idx, next_idx]
            
            # Compute tracking error
            error = current_state - ref_state
            
            # CLF (Control Lyapunov Function)
            V = jnp.sum(jnp.square(error)) / 2
            
            # Get nominal dynamics from neural ODE
            f_x = model.func_rot(0, current_state, None)
            
            # Get reference dynamics
            f_xref = model.func_rot(0, ref_state, None)
            
            # Compute CLF derivative terms
            s = jnp.dot(error, f_x - f_xref)
            
            # QP parameters for CLF-based control
            Q_opt = jnp.eye(3)  # Cost on control input
            G_opt = 2 * error.reshape(1, -1)  # CLF derivative terms
            h_opt = jnp.array([-alpha_V * V - 2 * s])  # CLF constraint
            
            # Solve QP using OSQP
            sol = qp.run(params_obj=(Q_opt, jnp.zeros(3)), params_ineq=(G_opt, h_opt)).params
            u_star = sol.primal.reshape(-1)
            
            # Apply control input and integrate dynamics
            x_dot = f_x + u_star
            current_state = current_state + dt * x_dot
            
            # Store current state
            actual_traj.append(current_state)
        
        actual_traj = jnp.array(actual_traj)
        target_indices = jnp.array(target_indices)
        all_trajectories.append((actual_traj, target_indices))
    
    # Save trajectories and target_trajs to file
    np.savez(save_path, 
             trajectories=[traj for traj, _ in all_trajectories],
             target_indices=[indices for _, indices in all_trajectories],
             target_trajs=target_trajs)
    
    return all_trajectories

def visualize_follow_trajectory(trajectories_path="trajectories.npz", output_path='trajectory_animation.mp4'):
    """
    Visualize pre-computed trajectories following target trajectories.
    
    Args:
        trajectories_path: Path to the .npz file containing trajectories and target data
        output_path: Path to save the animation. If None, displays the animation instead.
    """
    # Load the saved trajectories
    data = np.load(trajectories_path)
    all_trajectories = list(zip(data['trajectories'], data['target_indices']))
    target_trajs = data['target_trajs']
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all target trajectories initially with low opacity
    target_lines = []
    for l in range(len(target_trajs)):
        line = ax.plot3D(target_trajs[l, :, 0], target_trajs[l, :, 1], target_trajs[l, :, 2],
                 'b-', alpha=0.3, label='Sample Trajectory' if l == 0 else None)[0]
        target_lines.append(line)
    
    # Initialize the line that will show the generated trajectory
    generated_line, = ax.plot3D([], [], [], 'r-', linewidth=2, label='Generated Trajectory')
    
    # Initialize scatter plot for start point
    start_point = ax.scatter([], [], [], color='red', s=100, label='Start Point')

    # Add a dummy line for the "Trajectory Being Followed" label
    followed_line = ax.plot([], [], [], 'g-', alpha=1.0, label='Target Trajectory')[0]
    followed_line.set_visible(True)  # Make the line visible
    
    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory Following')
    ax.legend()
    
    # Animation update function
    def update(frame):
        # Calculate which trajectory we're on and the frame within that trajectory
        total_frames = 0
        current_traj_idx = 0
        frame_in_current_traj = frame
        
        for idx, (traj, indices) in enumerate(all_trajectories):
            if frame_in_current_traj >= len(traj):
                frame_in_current_traj -= len(traj)
                current_traj_idx += 1
            else:
                break
        
        # If we've shown all trajectories, clear everything
        if current_traj_idx >= len(all_trajectories):
            generated_line.set_data([], [])
            generated_line.set_3d_properties([])
            start_point._offsets3d = ([], [], [])
            followed_line.set_data([], [])
            followed_line.set_3d_properties([])
            for line in target_lines:
                line.set_alpha(0.3)
                line.set_color('b')
            return generated_line, start_point, followed_line, *target_lines
        
        current_traj, current_indices = all_trajectories[current_traj_idx]
        
        # Update generated trajectory
        generated_line.set_data(current_traj[:frame_in_current_traj+1, 0], 
                              current_traj[:frame_in_current_traj+1, 1])
        generated_line.set_3d_properties(current_traj[:frame_in_current_traj+1, 2])
        
        # Update start point
        start_point._offsets3d = ([current_traj[0, 0]], 
                                [current_traj[0, 1]], 
                                [current_traj[0, 2]])
        
        # Update target trajectory colors and followed line
        if frame_in_current_traj < len(current_indices):
            current_target = current_indices[frame_in_current_traj]
            # Update the followed line to match the current target trajectory
            target_traj = target_trajs[current_target]
            followed_line.set_data(target_traj[:, 0], target_traj[:, 1])
            followed_line.set_3d_properties(target_traj[:, 2])
            followed_line.set_color('g')
            followed_line.set_alpha(1.0)
            
            for i, line in enumerate(target_lines):
                if i == current_target:
                    line.set_alpha(0.0)  # Hide the original line when it's being followed
                else:
                    line.set_alpha(0.3)
                    line.set_color('b')
        
        return generated_line, start_point, followed_line, *target_lines
    
    # Calculate total number of frames needed
    total_frames = sum(len(traj) for traj, _ in all_trajectories)
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                 interval=50, blit=True)
    
    if output_path is None:
        plt.show()
        return
    
    # Save animation
    writer = animation.FFMpegWriter(fps=60, bitrate=3000)
    anim.save(output_path, writer=writer)
    plt.close()

if __name__ == "__main__":
    
    # Load data
    x, x_dot, x_att, x_init = load_data('custom',show_plot=False, separate=True, shift=False)
    x = jnp.array(x, dtype=jnp.float32)
    
    model_path = "models/mlp_width64_depth1.eqx"
    
    # Generate and save trajectories
    follow_trajectory(
        model_path=model_path,
        x=x,
        target_trajs=x,
        number_of_trajectories=5,
        dt=0.01,
        n_steps=150,
        save_path="data/trajectories.npz",
        switch=True
    )
    
    # Visualize the trajectories
    visualize_follow_trajectory("data/trajectories.npz", output_path=None) 