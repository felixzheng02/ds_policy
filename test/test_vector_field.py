import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from ds_policy.policy import DSPolicy, PositionModelConfig, QuaternionModelConfig, UnifiedModelConfig
from ds_policy.ds_utils import load_data


def plot_position_vector_field(
    model: Callable[[np.ndarray], np.ndarray],
    demo_trajs: list[np.ndarray],
    save_path: str = None,
):
    """
    Visualize the vector field of the trained model along with training data.

    Args:
        model: Trained model
        demo_trajs: Training trajectory data
        save_path: Path to save the plot
    """


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
                pos = np.array(
                    [X_np[i, j, k], Y_np[i, j, k], Z_np[i, j, k]]
                )
                quat = np.array(
                    [0, 0, 0, 1]
                )
                vel = model(np.concatenate([pos, quat]))
                U[i, j, k] = vel[0]
                V[i, j, k] = vel[1]
                W[i, j, k] = vel[2]

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
        length=0.03,
        normalize=True,
        color="red",
        alpha=0.3,
    )

    for traj in demo_trajs:
        ax.plot3D(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            "b-",
            linewidth=1,
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


if __name__ == "__main__":
    x, x_dot, quat, omega, gripper = load_data("OpenSingleDoor", "OpenSingleDoor_MoveTowards_option", finger=False, transform_to_object_of_interest_frame=True, debug_on=False)
    unified_config = UnifiedModelConfig(
        mode='se3_lpvds',
        k_init=3
    )
    ds_policy = DSPolicy(x, x_dot, quat, omega, gripper, unified_config=unified_config, dt=1/60, switch=False, lookahead=5)
    ds_policy._add_modulation_point([0, 0, 0, 0, 0, 0, 1])
    pos_to_vel = lambda x: ds_policy.get_action(x)[:3]
    plot_position_vector_field(pos_to_vel, x)