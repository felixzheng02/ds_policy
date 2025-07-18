import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional
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


def plot_position_vector_field_planes(
    model: Callable[[np.ndarray], np.ndarray],
    demo_trajs: list[np.ndarray],
    fixed_point: np.ndarray,
    grid_points: int = 20,
    save_path: Optional[str] = None,
):
    """Plot 2-D vector field slices in the XY, XZ, and YZ planes.

    Args:
        model: Callable that maps a 7-D state (pos[3] + quat[4]) to a 6-D twist. Only the
            positional velocity components are used for the quiver plot.
        demo_trajs: Demonstration trajectories used for computing plotting bounds and
            overlaying on the quiver.
        fixed_point: 3-D point representing the centre of modulation. For each slice we
            keep the coordinate orthogonal to the plane fixed at the corresponding
            component of *fixed_point*.
        grid_points: Number of points per axis for the meshgrid.
        save_path: Optional path. If provided, the figure will be saved here; otherwise
            ``plt.show()`` is called.
    """

    demo_trajs_flat = np.concatenate(demo_trajs, axis=0)

    # Determine axis ranges with 10% padding so arrows are not on the border.
    padding = 0.1
    mins = demo_trajs_flat.min(axis=0)
    maxs = demo_trajs_flat.max(axis=0)
    ranges = maxs - mins
    mins -= padding * ranges
    maxs += padding * ranges
    x_min, y_min, z_min = mins
    x_max, y_max, z_max = maxs

    # Helper to evaluate the learnt policy at a specific 3-D point.
    quat_identity = np.array([0, 0, 0, 1], dtype=float)

    def _vel(pos: np.ndarray) -> np.ndarray:
        return model(np.concatenate([pos, quat_identity]))[:3]

    # Prepare figure.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plane_names = [
        ("XY", ("X", "Y"), fixed_point[2]),
        ("XZ", ("X", "Z"), fixed_point[1]),
        ("YZ", ("Y", "Z"), fixed_point[0]),
    ]

    # XY plane --------------------------------------------------------------
    x_lin = np.linspace(x_min, x_max, grid_points)
    y_lin = np.linspace(y_min, y_max, grid_points)
    X, Y = np.meshgrid(x_lin, y_lin)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(grid_points):
        for j in range(grid_points):
            vel = _vel(np.array([X[i, j], Y[i, j], fixed_point[2]]))
            U[i, j], V[i, j] = vel[0], vel[1]
    ax = axes[0]
    ax.quiver(X, Y, U, V, color="red", alpha=0.5)
    for traj in demo_trajs:
        ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"XY plane (Z = {fixed_point[2]:.2f})")

    # XZ plane --------------------------------------------------------------
    x_lin = np.linspace(x_min, x_max, grid_points)
    z_lin = np.linspace(z_min, z_max, grid_points)
    X, Z = np.meshgrid(x_lin, z_lin)
    U = np.zeros_like(X)
    W = np.zeros_like(Z)
    for i in range(grid_points):
        for j in range(grid_points):
            vel = _vel(np.array([X[i, j], fixed_point[1], Z[i, j]]))
            U[i, j], W[i, j] = vel[0], vel[2]
    ax = axes[1]
    ax.quiver(X, Z, U, W, color="red", alpha=0.5)
    for traj in demo_trajs:
        ax.plot(traj[:, 0], traj[:, 2], "b-", linewidth=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title(f"XZ plane (Y = {fixed_point[1]:.2f})")

    # YZ plane --------------------------------------------------------------
    y_lin = np.linspace(y_min, y_max, grid_points)
    z_lin = np.linspace(z_min, z_max, grid_points)
    Y, Z = np.meshgrid(y_lin, z_lin)
    V = np.zeros_like(Y)
    W = np.zeros_like(Z)
    for i in range(grid_points):
        for j in range(grid_points):
            vel = _vel(np.array([fixed_point[0], Y[i, j], Z[i, j]]))
            V[i, j], W[i, j] = vel[1], vel[2]
    ax = axes[2]
    ax.quiver(Y, Z, V, W, color="red", alpha=0.5)
    for traj in demo_trajs:
        ax.plot(traj[:, 1], traj[:, 2], "b-", linewidth=1)
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.set_title(f"YZ plane (X = {fixed_point[0]:.2f})")

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    x, x_dot, quat, omega, gripper = load_data("OpenSingleDoor", "OpenSingleDoor_MoveTowards_option", finger=False, transform_to_object_of_interest_frame=True, debug_on=False)
    unified_config = UnifiedModelConfig(
        mode='se3_lpvds'
    )
    ds_policy = DSPolicy(x, x_dot, quat, omega, gripper, unified_config=unified_config, dt=1/60, switch=False, lookahead=5)
    modulation_centor = np.array([0, -0.1, -0.1])
    # ds_policy._add_spherical_modulation(modulation_centor, 0.05)
    ds_policy._add_ellipsoid_modulation(modulation_centor, [0.1, 0.05, 0.05], np.eye(3))
    pos_to_vel = lambda x: ds_policy.get_action(x)[:3]
    # Plot 3-D vector field
    # plot_position_vector_field(pos_to_vel, x)
    # Plot 2-D slices through the modulation centre
    plot_position_vector_field_planes(pos_to_vel, x, modulation_centor)
    