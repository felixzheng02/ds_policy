import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from ds_policy.policy import DSPolicy, PositionModelConfig, QuaternionModelConfig, UnifiedModelConfig
from ds_policy.ds_utils import load_data


# State is [x, y, z, qx, qy, qz, qw] (7D)
def update_state(cur_state, vel, dt):
    pos_vel = vel[:3]
    ang_vel = vel[3:6]

    new_pos = cur_state[:3] + pos_vel * dt

    # Update orientation using quaternion integration
    current_quat = cur_state[3:]
    # Ensure quaternion is normalized
    current_quat /= np.linalg.norm(current_quat)
    
    # Convert angular velocity to rotation vector
    rotation_vector = ang_vel * dt
    delta_rotation = R.from_rotvec(rotation_vector)
    
    # Combine rotations: R_new = delta_R * R_cur
    # In quaternion terms: q_new = delta_q * q_cur
    current_rotation = R.from_quat(current_quat)
    new_rotation = delta_rotation * current_rotation
    new_quat = new_rotation.as_quat()
    
    # Ensure the new quaternion is normalized
    new_quat /= np.linalg.norm(new_quat)

    return np.concatenate([new_pos, new_quat])

class Simulator:
    def __init__(self, ds_policy):
        self.ds_policy = ds_policy
        self.traj = []
        self.ref_traj_indices = []
        self.ref_point_indices = []

    def simulate(
        self,
        init_state, # Expecting 7D state [pos, quat]
        save_path: str,
        n_steps: int = 100,
        clf: bool = True,
        alpha_V: float = 10,
        lookahead: int = 5,
    ):
        state = init_state # State is now 7D
        for i in range(n_steps):
            self.traj.append(state.copy()) # Store 7D state
            # Policy expects 7D state [pos, quat]
            action = self.ds_policy.get_action(
                state, # Pass the 7D state directly
                clf=clf,
                alpha_V=alpha_V,
                lookahead=lookahead,
            )
            if clf:
                self.ref_traj_indices.append(self.ds_policy.ref_traj_idx)
                self.ref_point_indices.append(self.ds_policy.ref_point_idx_lookahead)
            state = update_state(state, action, 1/60) # Update 7D state

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
            ref_point_indices=self.ref_point_indices,
            allow_pickle=True,
        )
        return self.traj, demo_trajs, self.ref_traj_indices, self.ref_point_indices


class Animator:

    def __init__(
        self,
        traj: np.ndarray, # Trajectory is now N x 7
        demo_trajs: list[np.ndarray],
        ref_traj_indices: list[int],
        ref_point_indices: list[int],
    ):
        self.traj = traj
        self.demo_trajs = demo_trajs
        self.ref_traj_indices = ref_traj_indices
        self.ref_point_indices = ref_point_indices
        self.generated_line = None
        self.demo_lines = []
        self.ref_line = None
        self.start_point = None
        self.arrow_container = [None]
        self.ref_arrow_container = [None]
        self.orientation_label = None
        self.ref_orientation_label = None
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
        # Arrow text labels
        self.orientation_label = self.ax.text(0, 0, 0, "Orientation", color="orange")
        self.ref_orientation_label = self.ax.text(0, 0, 0, "Ref Orientation", color="purple")
        self.orientation_label.set_visible(False)  # Hide initially
        self.ref_orientation_label.set_visible(False)  # Hide initially

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

        # Get current orientation quaternion directly from state
        quat = self.traj[frame, 3:]
        rotation = R.from_quat(quat)

        # Create direction vector from rotation matrix (using x-axis)
        direction_vector = rotation.as_matrix()[:, 2]
        dx, dy, dz = direction_vector

        # Apply arrow length scaling for visualization
        arrow_length = 0.05
        dx_scaled = arrow_length * dx
        dy_scaled = arrow_length * dy
        dz_scaled = arrow_length * dz

        # Remove previous arrows if they exist
        if self.arrow_container[0] is not None:
            self.arrow_container[0].remove()
        if self.ref_arrow_container[0] is not None:
            self.ref_arrow_container[0].remove()

        # Create new arrow for current pose
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

        # Update current orientation label position and make it visible
        self.orientation_label.set_position(
            (
                pos[0] + dx_scaled * 0.7,
                pos[1] + dy_scaled * 0.7,
                pos[2] + dz_scaled * 0.7,
            )
        )
        self.orientation_label.set_visible(True)

        # Add reference point visualization if available
        if len(self.ref_point_indices) > 0 and self.ref_point_indices[frame] is not None:
            ref_traj_idx = self.ref_traj_indices[frame]
            ref_point_idx = self.ref_point_indices[frame]
            ref_point = self.demo_trajs[ref_traj_idx][ref_point_idx]
            ref_pos = ref_point[:3]
            ref_quat = ref_point[3:7]

            # Convert reference quaternion to rotation object
            ref_rotation = R.from_quat(ref_quat)

            # Calculate reference orientation vector (using x-axis)
            ref_direction_vector = ref_rotation.as_matrix()[:, 2]
            ref_dx, ref_dy, ref_dz = ref_direction_vector

            # Scale reference arrow
            ref_dx_scaled = arrow_length * ref_dx
            ref_dy_scaled = arrow_length * ref_dy
            ref_dz_scaled = arrow_length * ref_dz

            # Create reference arrow
            self.ref_arrow_container[0] = self.ax.quiver(
                ref_pos[0],
                ref_pos[1],
                ref_pos[2],
                ref_dx_scaled,
                ref_dy_scaled,
                ref_dz_scaled,
                color="purple",
                linewidth=3,
                label="Reference Orientation" if frame == 0 else None,
            )

            # Update reference orientation label
            self.ref_orientation_label.set_position(
                (
                    ref_pos[0] + ref_dx_scaled * 0.7,
                    ref_pos[1] + ref_dy_scaled * 0.7,
                    ref_pos[2] + ref_dz_scaled * 0.7,
                )
            )
            self.ref_orientation_label.set_visible(True)

        return (
            self.generated_line,
            self.ref_line,
            self.start_point,
            self.arrow_container[0],
            self.orientation_label,
            self.ref_arrow_container[0],
            self.ref_orientation_label,
        )


def main():
    if (
        True
    ): # this will save trajectory data. use False to directlly animate without simulating every time
        option = "OpenSingleDoor_MoveTowards_option"
        x, x_dot, q, omega, gripper = load_data("OpenSingleDoor", option, finger=True, transform_to_object_of_interest_frame=True, debug_on=False)
        demo_trajs = [np.concatenate([pos, rot], axis=1) for pos, rot in zip(x, q)]
        demo_traj_probs = np.ones(len(x))

        # Create configuration objects for position and quaternion models
        pos_config = PositionModelConfig(mode="none")
        quat_config = QuaternionModelConfig(mode="simple")
        
        # Alternatively, you could use a unified model:
        unified_config = UnifiedModelConfig(
            mode="se3_lpvds",
            k_init=1,
            enable_simple_ds_near_target=True,
            simple_ds_pos_threshold=0.1,
            simple_ds_ori_threshold=0.5,
            K_pos=10.0,
            K_ori=10.0,
            # width=256,
            # depth=5,
            # save_path=f"models/mlp_width256_depth5_pos_ang_vel_{option}.pt",
            # batch_size=100,
            # device="mps",
            # lr_strategy=(1e-3, 1e-4, 1e-5),
            # epoch_strategy=(50, 50, 50),
            # length_strategy=(0.4, 0.7, 1),
            # plot=True,
            # print_every=10
        )

        ds_policy = DSPolicy(
            x=x, 
            x_dot=x_dot, 
            quat=q, 
            omega=omega, 
            gripper=gripper,
            # pos_config=pos_config,
            # quat_config=quat_config,
            unified_config=unified_config,
            dt=1/60, 
            switch=False, 
            demo_traj_probs=demo_traj_probs
        )

        simulator = Simulator(ds_policy)
        
        # Randomly initialize starting point
        # Set random seed for reproducibility
        # rng = np.random.default_rng(seed=3)
        rng = np.random.RandomState()
        init_pos_x = rng.uniform(low=-0.3, high=0.3)  # Random x,y,z position
        init_pos_y = rng.uniform(low=-0.4, high=-0.1)
        init_pos_z = rng.uniform(low=-0.3, high=0.3)
        # Set the same random seed for quaternion initialization to ensure reproducibility
        # quat_rng = np.random.RandomState(seed=3)
        quat_rng = np.random.RandomState()
        # init_euler = R.random(random_state=quat_rng).as_euler("xyz", degrees=False) # Old Euler init
        init_quat = R.random(random_state=quat_rng).as_quat() # New Quaternion init
        init_pos = np.array([init_pos_x, init_pos_y, init_pos_z])
        init_state = np.concatenate(
            [init_pos, init_quat] # Create 7D state
        )
        init_state = np.array([-0.2, -0.25, -0.3, 0.1, 0, 0, 0])
        
        simulator.simulate(
            np.concatenate([x[0][0], q[0][0]]), # Old Euler example
            # init_state, # Pass 7D state
            "ds_policy/data/test_ds_policy.npz",
            n_steps=300,
            clf=True,
            alpha_V=10,
            lookahead=20,
        )
    
    # Load and visualize the results
    data = np.load("ds_policy/data/test_ds_policy.npz", allow_pickle=True)
    demo_trajs = data["demo_trajs"]
    traj = data["traj"] # Trajectory is N x 7
    ref_traj_indices = data["ref_traj_indices"]
    ref_point_indices = data["ref_point_indices"]
    animator = Animator(traj, demo_trajs, ref_traj_indices, ref_point_indices)
    animator.animate(None)

if __name__ == "__main__":
    option = "CloseSingleDoor-Op0"
    x = np.load(f"./trajectory_data/x_{option}.npy", allow_pickle=True)
    x_dot = np.load(f"./trajectory_data/x_dot_{option}.npy", allow_pickle=True)
    quat = np.load(f"./trajectory_data/quat_{option}.npy", allow_pickle=True)
    omega = np.load(f"./trajectory_data/omega_{option}.npy", allow_pickle=True)

    unified_config = UnifiedModelConfig(
        mode="se3_lpvds",
        K_candidates=[3],
        enable_simple_ds_near_target=True,
    )

    ds_policy = DSPolicy(
        x=x,
        x_dot=x_dot,
        quat=quat,
        omega=omega,
        gripper=[],
        unified_config=unified_config,
        dt=1/60,
        switch=False,
        )
    
    simulator = Simulator(ds_policy)
        
    # Randomly initialize starting point
    # Set random seed for reproducibility
    # rng = np.random.default_rng(seed=3)
    rng = np.random.RandomState()
    init_pos_x = rng.uniform(low=-0.3, high=0.3)  # Random x,y,z position
    init_pos_y = rng.uniform(low=-0.4, high=-0.1)
    init_pos_z = rng.uniform(low=-0.3, high=0.3)
    # Set the same random seed for quaternion initialization to ensure reproducibility
    # quat_rng = np.random.RandomState(seed=3)
    quat_rng = np.random.RandomState()
    # init_euler = R.random(random_state=quat_rng).as_euler("xyz", degrees=False) # Old Euler init
    init_quat = R.random(random_state=quat_rng).as_quat() # New Quaternion init
    init_pos = np.array([init_pos_x, init_pos_y, init_pos_z])
    init_state = np.concatenate(
        [init_pos, init_quat] # Create 7D state
    )
    init_state = np.array([-0.2, -0.25, -0.3, 0.1, 0, 0, 0])
    
    simulator.simulate(
        np.concatenate([x[0][0], quat[0][0]]),
        # init_state, # Pass 7D state
        "ds_policy/data/test_ds_policy.npz",
        n_steps=300,
        clf=True,
        alpha_V=10,
        lookahead=20,
    )
    
    # Load and visualize the results
    data = np.load("ds_policy/data/test_ds_policy.npz", allow_pickle=True)
    demo_trajs = data["demo_trajs"]
    traj = data["traj"] # Trajectory is N x 7
    ref_traj_indices = data["ref_traj_indices"]
    ref_point_indices = data["ref_point_indices"]
    animator = Animator(traj, demo_trajs, ref_traj_indices, ref_point_indices)
    animator.animate(None)