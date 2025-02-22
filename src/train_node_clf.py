import time

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax
from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate
import jax.tree_util as jtu
import functools as ft
import numpy as np
from matplotlib.lines import Line2D
import sys
import os
from jaxopt import OSQP
import matplotlib.animation as animation

# # Add the root directory to Python path
# root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# sys.path.append(root_dir)

from load_tools import load_data
from node_clf import Func_rot, NeuralODE_rot

def import_data(show_plot=True, separate=True, shift=False):
    x, x_dot, x_att, x_init = load_data('custom', show_plot=show_plot, separate=separate, shift=shift)  # x: positions (N, 3), x_dot: velocities (N, 3)
    return x, x_dot, x_att, x_init

def plot_vector_field(model_path, x, width_size, depth, show=True):
    """
    Visualize the vector field of the trained model along with training data.
    
    Args:
        model_path: Path to the saved model file
        x: Training trajectory data of shape (L, M, 3) where L is number of trajectories,
           M is points per trajectory, and 3 is the dimension
        save_path: Path to save the plot
    """
    # Create a template model with same structure
    _, _, data_size = x.shape
    key = jrandom.PRNGKey(0)  # Dummy key for template
    template_model = NeuralODE_rot(data_size, width_size, depth, key=key)
    
    # Load the saved model using the template
    model = eqx.tree_deserialise_leaves(model_path, template_model)

    # Compute bounds across all trajectories
    x_min, x_max = x[..., 0].min(), x[..., 0].max()
    y_min, y_max = -0.05, x[..., 1].max()
    z_min, z_max = x[..., 2].min(), x[..., 2].max()
    
    # Add some padding to the bounds
    padding = 0.1
    x_min, x_max = x_min - padding, x_max + padding
    y_min, y_max = y_min - padding, y_max + padding
    z_min, z_max = z_min - padding, z_max + padding
    
    # Create grid points
    grid_points = 10
    x_grid = jnp.linspace(x_min, x_max, grid_points)
    y_grid = jnp.linspace(y_min, y_max, grid_points)
    z_grid = jnp.linspace(z_min, z_max, grid_points)
    
    X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid)
    # Evaluate vector field at each point
    U = jnp.zeros_like(X)
    V = jnp.zeros_like(Y)
    W = jnp.zeros_like(Z)
    
    for i in range(grid_points):
        for j in range(grid_points):
            for k in range(grid_points):
                point = jnp.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                velocity = model.func_rot(0, point, None)
                U = U.at[i,j,k].set(velocity[0])
                V = V.at[i,j,k].set(velocity[1])
                W = W.at[i,j,k].set(velocity[2])
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot vector field
    # Downsample for clarity
    stride = 1
    ax.quiver(X[::stride,::stride,::stride], 
              Y[::stride,::stride,::stride], 
              Z[::stride,::stride,::stride],
              U[::stride,::stride,::stride], 
              V[::stride,::stride,::stride], 
              W[::stride,::stride,::stride],
              length=0.01, normalize=True, color='blue', alpha=0.3)
    
    # Plot all training trajectories
    for i in range(x.shape[0]):
        ax.plot3D(x[i, :, 0], x[i, :, 1], x[i, :, 2], 'r-', linewidth=2, 
                 label='Training Data' if i == 0 else None)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Vector Field and Training Trajectories')
    ax.legend()
    
    if show:
        plt.show()
    else:
        plt.savefig("vector_field_3d.png")
    plt.close()

def test_random_trajectory(model_path, x, width_size, depth, n_steps=1000, show=True, key=None):
    """
    Test the trained model with a random initial point and visualize the resulting trajectory.
    
    Args:
        model_path: Path to the saved model file
        x: Training data to determine bounds for random initialization
        n_steps: Number of time steps to simulate
        show: Whether to display the plot (True) or save it (False)
    """
    # Create a template model with same structure
    x = x.reshape(-1, 3)
    _, data_size = x.shape
    if key is None:
        key = jrandom.PRNGKey(int(time.time()))  # Random key based on current time
    template_model = NeuralODE_rot(data_size, width_size, depth, key=key)
    
    # Load the saved model
    model = eqx.tree_deserialise_leaves(model_path, template_model)

    # Generate random initial point within the bounds of training data
    x_bounds = (x[:, 0].min(), x[:, 0].max())
    y_bounds = (x[:, 1].min(), x[:, 1].max())
    z_bounds = (x[:, 2].min(), x[:, 2].max())
    
    init_point = jnp.array([
        jrandom.uniform(key, (), minval=x_bounds[0], maxval=x_bounds[1]),
        jrandom.uniform(key, (), minval=y_bounds[0], maxval=y_bounds[1]),
        jrandom.uniform(key, (), minval=z_bounds[0], maxval=z_bounds[1])
    ])

    # Create time points
    dt = 0.01
    ts = jnp.arange(0, n_steps * dt, dt)

    # Generate trajectory
    trajectory = model(ts, init_point)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits with padding
    padding = 0.1  # 10% padding
    x_range = x[:, 0].max() - x[:, 0].min()
    y_range = x[:, 1].max() - x[:, 1].min() 
    z_range = x[:, 2].max() - x[:, 2].min()
    
    ax.set_xlim([x[:, 0].min() - padding * x_range, x[:, 0].max() + padding * x_range])
    ax.set_ylim([x[:, 1].min() - padding * y_range, x[:, 1].max() + padding * y_range]) 
    ax.set_zlim([x[:, 2].min() - padding * z_range, x[:, 2].max() + padding * z_range])
    
    # Plot training data for reference
    ax.plot3D(x[:, 0], x[:, 1], x[:, 2], 'b-', alpha=0.3, label='Training Data')
    
    # Plot generated trajectory
    ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r-', linewidth=2, label='Generated Trajectory')
    
    # Plot initial point
    ax.scatter(init_point[0], init_point[1], init_point[2], color='green', s=100, label='Initial Point')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Random Initial Point Trajectory')
    ax.legend()
    
    if show:
        plt.show()
    else:
        plt.savefig("random_trajectory.png")
    plt.close()

    return init_point, trajectory

def train(
    model_path,
    x,
    x_dot,  # Added x_dot as input
    data_size=3,
    batch_size=1,
    lr_strategy=(1e-3, 1e-3, 1e-3),
    steps_strategy=(5000, 5000, 5000),
    length_strategy=(0.4, 0.7, 1),
    width_size=64,
    depth=3,
    seed=1000,
    plot=True,
    print_every=100,
    save_every=1000,
):
    key = jrandom.PRNGKey(seed)
    data_key, model_key, loader_key = jrandom.split(key, 3)

    # Ensure input data has correct shape
    if x.ndim != 3 or x_dot.ndim != 3:
        raise ValueError("Input data must be 3D arrays with shape (n_trajectories, n_points, n_features)")
    
    n_trajectories, n_points, data_size = x.shape
    if x_dot.shape != (n_trajectories, n_points, data_size):
        raise ValueError("x and x_dot must have the same shape")

    # Initialize model with 3D input/output
    model = NeuralODE_rot(data_size, width_size, depth, key=model_key)

    @eqx.filter_value_and_grad
    @eqx.filter_jit
    def grad_loss(model, x_batch, x_dot_batch):
        """Compute loss between predicted and actual velocities"""
        # Predict velocities directly
        v_pred = model.predict_velocity(x_batch)
        
        # MSE loss on velocities
        loss = jnp.mean((x_dot_batch - v_pred) ** 2)
        return loss

    @eqx.filter_jit
    def make_step(x_batch, x_dot_batch, model, opt_state):
        loss, grads = grad_loss(model, x_batch, x_dot_batch)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    # Training loop with curriculum learning
    for lr, steps in zip(lr_strategy, steps_strategy):
        # Learning rate schedule
        decay_scheduler = optax.cosine_decay_schedule(lr, decay_steps=steps, alpha=0.9)
        optim = optax.adabelief(learning_rate=decay_scheduler)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        
        # Get partial trajectory based on current length
        # curr_length = int(x.shape[0] * length)
        # _ts = ts[:curr_length]
        # _x = x[:curr_length]

        for step in range(steps):
            start_time = time.time()
            # Train on all trajectories at once since we've updated the model to handle 3D input
            loss, model, opt_state = make_step(x, x_dot, model, opt_state)
            end_time = time.time()
            
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end_time - start_time}")
            if (step % save_every) == 0 or step == steps-1:
                eqx.tree_serialise_leaves(model_path, model)

    if plot:
        # Plot velocity predictions vs actual for each trajectory
        plt.figure(figsize=(15, 5))
        
        # Get velocity predictions for all trajectories
        v_pred = model.predict_velocity(x)
        
        # Plot average velocity components across all trajectories
        for i, comp in enumerate(['x', 'y', 'z']):
            plt.subplot(1, 3, i+1)
            # Plot mean velocities across trajectories
            plt.plot(x_dot[:, :, i].mean(axis=0), label=f'Real d{comp}/dt')
            plt.plot(v_pred[:, :, i].mean(axis=0), '--', label=f'Pred d{comp}/dt')
            plt.legend()
            plt.title(f'{comp} velocity component (mean across trajectories)')
        
        plt.tight_layout()
        plt.show()
        plt.close()

        # Generate and plot a test trajectory
        # dt = 0.01
        # ts = jnp.arange(0, x.shape[0] * dt, dt)
        # pred_traj = model.generate_trajectory(ts, x[0])

        # # 3D trajectory plot
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection='3d')
        
        # ax.plot3D(x[:, 0], x[:, 1], x[:, 2], 'b-', label='Real')
        # ax.plot3D(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 'r--', label='Predicted')
        
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.legend()
        # plt.title('3D Trajectory: Real vs Predicted')
        # plt.show()
        # plt.close()

    return model

def main(model_path, train_model=True, plot_field=True, test_trajectory=True, width_size=256, depth=5):
    
    # Load data including velocities
    x, x_dot, x_att, x_init = import_data(show_plot=False, separate=True, shift=False)
    
    # Ensure data is in correct format (float32)
    x = jnp.array(x, dtype=jnp.float32)
    x_dot = jnp.array(x_dot, dtype=jnp.float32)
    
    # Reshape data if needed (assuming x and x_dot are already in (L, M, 3) format)
    if x.ndim != 3:
        raise ValueError("Input data must be in (L, M, 3) format where L is number of trajectories, M is points per trajectory")
    
    if train_model:
        model = train( 
            model_path,
            x,
            x_dot,
            data_size=x.shape[2],  # Use the last dimension as data_size
            lr_strategy=(1e-3, 1e-4, 1e-5),
            steps_strategy=(5000, 5000, 5000),
            length_strategy=(1, 1, 1),
            width_size=width_size,
            depth=depth
        )

    if plot_field:
        plot_vector_field(model_path, x, width_size, depth)
    
    # Test with random trajectories
    if test_trajectory:
        for i in range(10):
            init_point, trajectory = test_random_trajectory(
                model_path, 
                x,
                width_size,
                depth,
                n_steps=1000
        )

if __name__ == "__main__":
    width_size = 64
    depth = 1
    model_path = f"models/mlp_width{width_size}_depth{depth}.eqx"
    main(model_path, train_model=False, plot_field=False, test_trajectory=True, width_size=width_size, depth=depth)