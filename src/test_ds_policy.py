from ds_policy import DSPolicy
import os
import glob
import numpy as np
import utils
from load_tools import load_data
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Simulator:
    def __init__(self, ds_policy):
        self.ds_policy = ds_policy
        self.traj = []

    def simulate(self, init_state, n_steps=100, save_path=None):
        state = init_state
        self.traj.append(state)
        for i in range(n_steps):
            quat = utils.euler_to_quat(state[3:])
            action = self.ds_policy.get_action(np.concatenate([state[:3], quat]))
            state += action * self.ds_policy.dt
            self.traj.append(state)
        if save_path is not None:
            np.savez(
                save_path,
                demo_trajs=self.ds_policy.demo_trajs,
                traj=self.traj,
                allow_pickle=True,
            )
        return self.traj, self.ds_policy.demo_trajs


def animate(data_path, save_path=None):
    """
    Args:
        data_path: path to the data file
            'traj': generated trajectory (n, 6)
            'demo_trajs': demo trajectories [(n, 6), (n, 6), ...]
        save_path: path to save the animation
    """
    data = np.load(data_path, allow_pickle=True)
    demo_trajs = data["demo_trajs"]
    traj = data["traj"]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    for i, demo_traj in enumerate(demo_trajs):
        (line,) = ax.plot3D(
            demo_traj[:, 0],
            demo_traj[:, 1],
            demo_traj[:, 2],
            "b-",
            alpha=0.05,
            linewidth=1,
            label="Demo Trajectory" if i == 0 else None,
        )

    # Initialize plot elements with dummy points
    (generated_line,) = ax.plot3D(
        [], [], [], "#3A7D44", linewidth=3, label="Generated Trajectory"
    )
    start_point = ax.scatter3D([], [], [], c="red", s=100, label="Start Point")

    # Store the orientation arrow in a list so we can update it
    arrow_container = [None]

    # Arrow text label
    orientation_label = ax.text(0, 0, 0, "Orientation", color="orange")
    orientation_label.set_visible(False)  # Hide initially

    def update(frame):
        # Update trajectory line
        generated_line.set_data(traj[:frame, 0], traj[:frame, 1])
        generated_line.set_3d_properties(traj[:frame, 2])
        generated_line.set_alpha(1)
        generated_line.set_color("#3A7D44")

        # Update start point
        start_point._offsets3d = (
            [traj[0, 0]],
            [traj[0, 1]],
            [traj[0, 2]],
        )

        # Get current position
        pos = traj[frame, :3]

        # Convert Euler angles to direction vector
        euler_angles = traj[frame, 3:]
        quat = utils.euler_to_quat(euler_angles)

        # Create direction vector from quaternion
        w, x, y, z = quat

        # Calculate the raw direction vector from quaternion (before scaling)
        dx = 1 - 2 * (y**2 + z**2)
        dy = 2 * (x * y - w * z)
        dz = 2 * (x * z + w * y)

        # Normalize to unit vector
        magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
        dx /= magnitude
        dy /= magnitude
        dz /= magnitude

        # Apply arrow length scaling for visualization
        arrow_length = 0.05  # Reduced from 0.2 to make arrow shorter
        dx_scaled = arrow_length * dx
        dy_scaled = arrow_length * dy
        dz_scaled = arrow_length * dz

        # Remove previous arrow if it exists
        if arrow_container[0] is not None:
            arrow_container[0].remove()

        # Create new arrow
        arrow_container[0] = ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            dx_scaled,
            dy_scaled,
            dz_scaled,
            color="orange",
            linewidth=3,
            label="Orientation" if frame == 0 else None,
        )

        # Update label position and make it visible
        orientation_label.set_position(
            (
                pos[0] + dx_scaled * 0.7,
                pos[1] + dy_scaled * 0.7,
                pos[2] + dz_scaled * 0.7,
            )
        )
        orientation_label.set_visible(True)

        return generated_line, start_point, arrow_container[0], orientation_label

    total_frames = len(traj)

    # Use a slower interval to make orientation changes more visible
    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=100, blit=False
    )

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Trajectory Animation with Orientation")

    if save_path is None:
        plt.show()
        return

    writer = animation.FFMpegWriter(fps=15, bitrate=3000)
    anim.save(save_path, writer=writer)
    plt.close()


if __name__ == "__main__":
    if (
        True
    ):  # this will save trajectory data. use False to directlly animate without simulating every time
        x, x_dot, r = load_data("custom")
        demo_trajs = [np.concatenate([pos, rot], axis=1) for pos, rot in zip(x, r)]
        ds_policy = DSPolicy(demo_trajs, x_dot)
        # NOTE: add model path to avoid training model every time
        # e.g. ds_policy = DSPolicy(demo_trajs, x_dot, pos_model_path='neural_ode/models/mlp_width64_depth3.eqx')
        # TODO: sometimes "cannot load model when allow_pickle=False"
        # TODO: cannot load quat model from json file now
        simulator = Simulator(ds_policy)
        simulator.simulate(np.array([0, 0, 0, 0, 0, 0]))
    animate("neural_ode/data/test_ds_policy.npz")
