import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from load_tools import load_data
from neural_ode import NeuralODE


def plot_vector_field(
    model_path: str,
    demo_trajs: list[np.ndarray],
    width_size: int,
    depth: int,
    save_path: str = None,
):
    """
    Visualize the vector field of the trained model along with training data.

    Args:
        model_path: Path to the saved model file
        demo_trajs: Training trajectory data
        width_size: Width of hidden layers
        depth: Number of hidden layers
        save_path: Path to save the plot
    """
    data_size = demo_trajs[0].shape[-1]
    if data_size == 7:
        print("plot_vector_field only works for position data")
        return
    model = NeuralODE(data_size, width_size, depth)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    demo_trajs_flat = np.concatenate(demo_trajs, axis=0)

    x_min, x_max = demo_trajs_flat[:, 0].min(), demo_trajs_flat[:, 0].max()
    y_min, y_max = demo_trajs_flat[:, 1].min(), demo_trajs_flat[:, 1].max()
    z_min, z_max = demo_trajs_flat[:, 2].min(), demo_trajs_flat[:, 2].max()

    padding = 0.1  # 10% padding
    x_range = demo_trajs_flat[:, 0].max() - demo_trajs_flat[:, 0].min()
    y_range = demo_trajs_flat[:, 1].max() - demo_trajs_flat[:, 1].min()
    z_range = demo_trajs_flat[:, 2].max() - demo_trajs_flat[:, 2].min()
    x_min, x_max = (
        demo_trajs_flat[:, 0].min() - padding * x_range,
        demo_trajs_flat[:, 0].max() + padding * x_range,
    )
    y_min, y_max = (
        demo_trajs_flat[:, 1].min() - padding * y_range,
        demo_trajs_flat[:, 1].max() + padding * y_range,
    )
    z_min, z_max = (
        demo_trajs_flat[:, 2].min() - padding * z_range,
        demo_trajs_flat[:, 2].max() + padding * z_range,
    )

    # Create grid points
    grid_points = 10
    x_grid = torch.linspace(x_min, x_max, grid_points)
    y_grid = torch.linspace(y_min, y_max, grid_points)
    z_grid = torch.linspace(z_min, z_max, grid_points)

    X, Y, Z = torch.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

    X_np = X.numpy()
    Y_np = Y.numpy()
    Z_np = Z.numpy()

    # Evaluate vector field at each point
    U = np.zeros_like(X_np)
    V = np.zeros_like(Y_np)
    W = np.zeros_like(Z_np)

    for i in range(grid_points):
        for j in range(grid_points):
            for k in range(grid_points):
                point = torch.tensor(
                    [X_np[i, j, k], Y_np[i, j, k], Z_np[i, j, k]], dtype=torch.float32
                )
                with torch.no_grad():
                    velocity = model.forward(point)
                velocity_np = velocity.numpy()
                U[i, j, k] = velocity_np[0]
                V[i, j, k] = velocity_np[1]
                W[i, j, k] = velocity_np[2]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    stride = 1
    ax.quiver(
        X_np[::stride, ::stride, ::stride],
        Y_np[::stride, ::stride, ::stride],
        Z_np[::stride, ::stride, ::stride],
        U[::stride, ::stride, ::stride],
        V[::stride, ::stride, ::stride],
        W[::stride, ::stride, ::stride],
        length=0.01,
        normalize=True,
        color="red",
        alpha=0.3,
    )

    ax.plot3D(
        demo_trajs_flat[:, 0],
        demo_trajs_flat[:, 1],
        demo_trajs_flat[:, 2],
        "b-",
        linewidth=1,
        label="Training Data",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Vector Field and Training Trajectories")
    ax.legend()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def rollout_trajectory(
    model_path: str,
    demo_trajs: list[np.ndarray],
    init_point: np.ndarray,
    width_size: int,
    depth: int,
    dt: float = 0.01,
    n_steps: int = 1000,
    save_path: str = None,
):
    """
    Test the trained model with a random initial point and visualize the resulting trajectory.

    Args:
        model_path: Path to the saved model file
        demo_trajs: Training trajectory data
        init_point: Initial point for the trajectory
        width_size: Width of hidden layers
        depth: Number of hidden layers
        dt: Time step
        n_steps: Number of time steps to simulate
        save_path: Path to save the plot
    """
    data_size = demo_trajs[0].shape[-1]
    if data_size == 7:
        print("rollout_trajectory only works for position data")
        return
    model = NeuralODE(data_size, width_size, depth)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    traj = [init_point]
    for i in range(n_steps):
        with torch.no_grad():
            pos = traj[-1]
            vel = model.forward(torch.tensor(pos, dtype=torch.float32))
            traj.append(np.array(pos + vel.numpy() * dt))
    traj = np.array(traj)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    demo_trajs_flat = np.concatenate(demo_trajs, axis=0)

    padding = 0.1  # 10% padding
    x_range = demo_trajs_flat[:, 0].max() - demo_trajs_flat[:, 0].min()
    y_range = demo_trajs_flat[:, 1].max() - demo_trajs_flat[:, 1].min()
    z_range = demo_trajs_flat[:, 2].max() - demo_trajs_flat[:, 2].min()

    ax.set_xlim(
        [
            demo_trajs_flat[:, 0].min() - padding * x_range,
            demo_trajs_flat[:, 0].max() + padding * x_range,
        ]
    )
    ax.set_ylim(
        [
            demo_trajs_flat[:, 1].min() - padding * y_range,
            demo_trajs_flat[:, 1].max() + padding * y_range,
        ]
    )
    ax.set_zlim(
        [
            demo_trajs_flat[:, 2].min() - padding * z_range,
            demo_trajs_flat[:, 2].max() + padding * z_range,
        ]
    )

    # Plot training data for reference
    ax.plot3D(
        demo_trajs_flat[:, 0],
        demo_trajs_flat[:, 1],
        demo_trajs_flat[:, 2],
        "b-",
        alpha=0.3,
        label="Training Data",
    )

    # Plot generated trajectory
    ax.plot3D(
        traj[:, 0],
        traj[:, 1],
        traj[:, 2],
        "r-",
        linewidth=2,
        label="Generated Trajectory",
    )

    ax.scatter(
        init_point[0],
        init_point[1],
        init_point[2],
        color="green",
        s=100,
        label="Initial Point",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Rollout Trajectory")
    ax.legend()

    if save_path is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

    return init_point, traj


def train(
    x: list[np.ndarray],
    x_dot: list[np.ndarray],
    save_path: str = None,
    batch_size: int = 1,  # TODO: not used
    lr_strategy: tuple = (1e-3, 1e-3, 1e-3),
    steps_strategy: tuple = (5000, 5000, 5000),
    length_strategy: tuple = (0.4, 0.7, 1),  # TODO: not used
    width_size: int = 64,
    depth: int = 3,
    plot: bool = False,
    print_every: int = 100,
):
    """
    Train the NeuralODE model.

    Args:
        x: Training data. Each array can be of size (N, 3) or (N, 7).
        x_dot: Training velocity data. Each array can be of size (N, 3) or (N, 6).
        save_path: Path to save the trained model
        batch_size: Number of trajectories to train on at a time
        lr_strategy: Learning rate in each phase
        steps_strategy: Number of steps in each phase
        length_strategy: Fraction of the trajectories available for training in each phase
        width_size: Width of hidden layers
        depth: Number of hidden layers
        plot: Whether to plot the training progress
        print_every: Print the training progress every print_every steps
    """
    data_size = x[0].shape[-1]
    if save_path is None:
        if data_size == 3:
            save_path = f"DS-Policy/models/mlp_width{width_size}_depth{depth}_pos.pt"
        elif data_size == 7:
            save_path = f"DS-Policy/models/mlp_width{width_size}_depth{depth}_pos_quat.pt"
        else:
            raise ValueError(f"Invalid data size: {data_size}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    x_flat = np.concatenate(x, axis=0)
    x_dot_flat = np.concatenate(x_dot, axis=0)

    x_flat_tensor = torch.tensor(x_flat, dtype=torch.float32)
    x_dot_flat_tensor = torch.tensor(x_dot_flat, dtype=torch.float32)

    # Initialize model
    model = NeuralODE(data_size, width_size, depth)

    # Training loop with curriculum learning
    for lr, steps in zip(lr_strategy, steps_strategy):
        # Setup optimizer with learning rate scheduler
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=0.1 * lr
        )
        loss_fn = nn.MSELoss()

        for step in range(steps):
            start_time = time.time()

            optimizer.zero_grad()
            v_pred = model.forward(x_flat_tensor)
            loss = loss_fn(v_pred, x_dot_flat_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()

            end_time = time.time()

            if (step % print_every) == 0 or step == steps - 1:
                print(
                    f"Step: {step}, Loss: {loss.item()}, Computation time: {end_time - start_time}"
                )

    torch.save(model.state_dict(), save_path)

    if plot:
        # Plot velocity predictions vs actual for each trajectory
        model.eval()
        with torch.no_grad():
            v_pred = model.forward(x_flat_tensor)

        v_pred_np = v_pred.numpy()
        
        # Determine if we're dealing with position only or position+quaternion
        if data_size == 3:  # Position only - plot translational velocity
            plt.figure(figsize=(15, 5))
            # Plot translational velocity components
            for i, comp in enumerate(["x", "y", "z"]):
                plt.subplot(1, 3, i + 1)
                plt.plot(x_dot_flat[:, i], label=f"Real d{comp}/dt")
                plt.plot(v_pred_np[:, i], "--", label=f"Pred d{comp}/dt")
                plt.legend()
                plt.title(f"{comp} translational velocity")
            
            plt.tight_layout()
            plt.show()
            plt.close()
            
        elif data_size == 7:  # Position + quaternion - plot both translational and angular velocity
            # First figure: Translational velocity (first 3 components)
            plt.figure(figsize=(15, 5))
            for i, comp in enumerate(["x", "y", "z"]):
                plt.subplot(1, 3, i + 1)
                plt.plot(x_dot_flat[:, i], label=f"Real d{comp}/dt")
                plt.plot(v_pred_np[:, i], "--", label=f"Pred d{comp}/dt")
                plt.legend()
                plt.title(f"{comp} translational velocity")
            
            plt.tight_layout()
            plt.show()
            plt.close()
            
            # Second figure: Angular velocity (last 3 components)
            plt.figure(figsize=(15, 5))
            for i, comp in enumerate(["x", "y", "z"]):
                plt.subplot(1, 3, i + 1)
                plt.plot(x_dot_flat[:, i+3], label=f"Real ω_{comp}")
                plt.plot(v_pred_np[:, i+3], "--", label=f"Pred ω_{comp}")
                plt.legend()
                plt.title(f"{comp} angular velocity")
            
            plt.tight_layout()
            plt.show()
            plt.close()

    return model


if __name__ == "__main__":
    width_size = 64
    depth = 3
    model_path = f"DS-Policy/models/mlp_width{width_size}_depth{depth}_seg1.pt"
    x, x_dot, quat, omega = load_data("custom")

    if True:
        model = train(
            [np.concatenate([x[i], quat[i]], axis=-1) for i in range(len(x))],
            [np.concatenate([x_dot[i], omega[i]], axis=-1) for i in range(len(x_dot))],
            batch_size=1,
            lr_strategy=(1e-3, 1e-4, 1e-5),
            steps_strategy=(100, 100, 100),
            length_strategy=(1, 1, 1),
            width_size=width_size,
            depth=depth,
            plot=True,
            print_every=100,
        )

    if False:
        plot_vector_field(model_path, x, data_size, width_size, depth)

    if False:
        rollout_trajectory(
            model_path, x, np.array([0, -0.2, -0.2]), data_size, width_size, depth
        )
