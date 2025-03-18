import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

from ds_policy.ds_policy import DSPolicy
from ds_policy import ds_utils
from ds_policy.load_tools import load_data


class Simulator:
    def __init__(self, ds_policy):
        self.ds_policy = ds_policy
        self.traj = []
        self.ref_traj_indices = []

    def simulate(
        self,
        init_state,
        save_path: str,
        n_steps: int = 100,
        clf: bool = True,
        alpha_V: float = 10,
        lookahead: int = 5,
    ):
        state = init_state
        for i in range(n_steps):
            self.traj.append(state.copy())
            quat = ds_utils.euler_to_quat(state[3:])
            action = self.ds_policy.get_action(
                np.concatenate([state[:3], quat]),
                clf=clf,
                alpha_V=alpha_V,
                lookahead=lookahead,
            )
            if clf:
                self.ref_traj_indices.append(self.ds_policy.ref_traj_idx)
            state += action * 0.02

        demo_trajs = [
            np.concatenate([self.ds_policy.x[i], self.ds_policy.quat[i]], axis=-1)
            for i in range(len(self.ds_policy.x))
        ]
        # Create directory for save_path if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(
            save_path,
            demo_trajs=demo_trajs,
            traj=self.traj,
            ref_traj_indices=self.ref_traj_indices,
            allow_pickle=True,
        )
        return self.traj, demo_trajs, self.ref_traj_indices


class Animator:

    def __init__(
        self,
        traj: np.ndarray,
        demo_trajs: list[np.ndarray],
        ref_traj_indices: list[int],
    ):
        self.traj = traj
        self.demo_trajs = demo_trajs
        self.ref_traj_indices = ref_traj_indices
        self.generated_line = None
        self.demo_lines = []
        self.ref_line = None
        self.start_point = None
        self.arrow_container = [None]
        self.orientation_label = None
        self.ax = None
        self.demo_line_alpha = None

    def animate(self, save_path: str = None, interval: int = 100):
        fig = plt.figure(figsize=(12, 12))
        self.ax = fig.add_subplot(111, projection="3d")

        if self.demo_line_alpha is None:
            self.demo_line_alpha = 0.05

        for i, demo_traj in enumerate(self.demo_trajs):
            (line,) = self.ax.plot3D(
                demo_traj[:, 0],
                demo_traj[:, 1],
                demo_traj[:, 2],
                "b-",
                alpha=self.demo_line_alpha,
                linewidth=1,
                label="Demo Trajectory" if i == 0 else None,
            )
            self.demo_lines.append(line)
        # Initialize plot elements with dummy points
        (self.generated_line,) = self.ax.plot3D(
            [], [], [], "#3A7D44", linewidth=3, label="Generated Trajectory"
        )
        (self.ref_line,) = self.ax.plot3D(
            [], [], [], "r--", linewidth=2, label="Reference Trajectory"
        )
        self.start_point = self.ax.scatter3D(
            [], [], [], c="red", s=100, label="Start Point"
        )
        self.start_point._offsets3d = (
            [self.traj[0, 0]],
            [self.traj[0, 1]],
            [self.traj[0, 2]],
        )
        # Arrow text label
        self.orientation_label = self.ax.text(0, 0, 0, "Orientation", color="orange")
        self.orientation_label.set_visible(False)  # Hide initially

        total_frames = len(self.traj) - 1

        # Use a slower interval to make orientation changes more visible
        anim = animation.FuncAnimation(
            fig, self.update, frames=total_frames, interval=interval, blit=False
        )

        self.ax.legend()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Trajectory Animation with Orientation")

        if save_path is None:
            plt.show()
            return

        writer = animation.FFMpegWriter(fps=5, bitrate=3000)
        anim.save(save_path, writer=writer)
        plt.close()

    def update(self, frame):
        # Update trajectory line
        self.generated_line.set_data(
            self.traj[: frame + 1, 0], self.traj[: frame + 1, 1]
        )
        self.generated_line.set_3d_properties(self.traj[: frame + 1, 2])
        self.generated_line.set_alpha(1)
        self.generated_line.set_color("#3A7D44")

        if len(self.ref_traj_indices) > 0:
            ref_traj = self.demo_trajs[self.ref_traj_indices[frame]]
            self.ref_line.set_data(ref_traj[:, 0], ref_traj[:, 1])
            self.ref_line.set_3d_properties(ref_traj[:, 2])
            self.ref_line.set_alpha(1)
            self.ref_line.set_color("r")

        # Get current position
        pos = self.traj[frame, :3]

        # Convert Euler angles to direction vector
        euler_angles = self.traj[frame, 3:]
        quat = ds_utils.euler_to_quat(euler_angles)

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
        if self.arrow_container[0] is not None:
            self.arrow_container[0].remove()

        # Create new arrow
        self.arrow_container[0] = self.ax.quiver(
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
        self.orientation_label.set_position(
            (
                pos[0] + dx_scaled * 0.7,
                pos[1] + dy_scaled * 0.7,
                pos[2] + dz_scaled * 0.7,
            )
        )
        self.orientation_label.set_visible(True)

        return (
            self.generated_line,
            self.ref_line,
            self.start_point,
            self.arrow_container[0],
            self.orientation_label,
        )


if __name__ == "__main__":
    if (
        True
    ): # this will save trajectory data. use False to directlly animate without simulating every time
        option = "move_towards"
        x, x_dot, q, omega = load_data("custom", option)
        demo_trajs = [np.concatenate([pos, rot], axis=1) for pos, rot in zip(x, q)]
        demo_traj_probs = np.ones(len(x))
        demo_traj_probs[1] = 0

        # Define model configuration using the new structure
        model_config = {
            'pos_model': {
                # Either use average velocities
                # 'use_avg': True,
                # Or specify load_path to load an existing model
                'load_path': f"ds_policy/models/mlp_width128_depth3_{option}.pt",
                # Or provide training parameters if model doesn't exist yet
                # 'width': 128,
                # 'depth': 3,
                # 'save_path': f"ds_policy/models/mlp_width128_depth3_{option}.pt",
                # 'batch_size': 100,
                # 'device': "cpu",  # Can be "cpu", "cuda", or "mps"
                # 'lr_strategy': (1e-3, 1e-4, 1e-5),
                # 'epoch_strategy': (10, 10, 10),
                # 'length_strategy': (0.4, 0.7, 1),
                # 'plot': True,
                # 'print_every': 10
            },
            'quat_model': {
                # Either use load_path
                # 'load_path': f"ds_policy/models/quat_model_{option}.json",
                # Or specify training parameters
                'save_path': f"ds_policy/models/quat_model_{option}.json",
                'k_init': 10
            }
            # Alternatively, you could use a unified model:
            # 'unified_model': {
            #     'load_path': f"ds_policy/models/mlp_width256_depth5_unified_{option}.pt",
            #     'device': "cpu" 
            # }
        }

        # Initialize DS policy with the new model_config parameter
        ds_policy = DSPolicy(
            x=x, 
            x_dot=x_dot, 
            quat=q, 
            omega=omega, 
            model_config=model_config,
            dt=1/60, 
            switch=True, 
            demo_traj_probs=demo_traj_probs
        )

        simulator = Simulator(ds_policy)
        
        # Randomly initialize starting point
        # Set random seed for reproducibility
        rng = np.random.default_rng(seed=3)
        init_pos_x = rng.uniform(low=-0.3, high=0.3)  # Random x,y,z position
        init_pos_y = rng.uniform(low=-0.4, high=-0.1)
        init_pos_z = rng.uniform(low=-0.3, high=0.3)
        # Set the same random seed for quaternion initialization to ensure reproducibility
        quat_rng = np.random.RandomState(seed=4)
        init_euler = R.random(random_state=quat_rng).as_euler("xyz", degrees=False)
        init_state = np.concatenate(
            [np.array([init_pos_x, init_pos_y, init_pos_z]), init_euler]
        )
        
        simulator.simulate(
            np.concatenate([x[1][0], ds_utils.quat_to_euler(q[1][0])]),
            # init_state,
            "ds_policy/data/test_ds_policy.npz",
            n_steps=100,
            clf=True,
            alpha_V=10,
            lookahead=10,
        )
    
    # Load and visualize the results
    data = np.load("ds_policy/data/test_ds_policy.npz", allow_pickle=True)
    demo_trajs = data["demo_trajs"]
    traj = data["traj"]
    ref_traj_indices = data["ref_traj_indices"]
    animator = Animator(traj, demo_trajs, ref_traj_indices)
    animator.animate(None)
