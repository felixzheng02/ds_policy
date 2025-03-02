import os
import sys
from neural_ode import NeuralODE
import numpy as np
import torch
from jaxopt import OSQP
from scipy import sparse  # Add this import for sparse matrices
# Add the quaternion_ds package to sys.path to import it
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "se3_lpvds/src/quaternion_ds",
    )
)

import json

from train_neural_ode import train
from scipy.spatial.transform import Rotation as R
from src.gmm_class import gmm_class
from src.quat_class import quat_class
from src.util.process_tools import (
    pre_process,
    compute_output,
    extract_state,
    rollout_list,
)


class DSPolicy:
    """
    __init__(model_path: str,
        demo_trajs: List[np.ndarray(n_steps, n_dim=x_dim+r_dim)],
        demo_vels: List[np.ndarray(n_steps, x_dim)],
        dt: float,
        switch: bool,
        backtrack_steps: int)
    NOTE: either load_pos_model or train_pos_model should be called before get_action
    train_pos_model(save_path: str, 
        batch_size: int=1, 
        lr_strategy: tuple=(1e-3, 1e-3, 1e-3), 
        steps_strategy: tuple=(5000, 5000, 5000), 
        length_strategy: tuple=(0.4, 0.7, 1), 
        plot: bool=True, 
        print_every: int=100)
    load_pos_model(model_path: str)
    NOTE: either train_quat_model or load_quat_model should be called before get_action
    train_quat_model(save_path: str, k_init: int=10)
    load_quat_model(model_path: str)
    get_action(state: np.ndarray(n_dims)) -> (dx, dy, dz, droll, dpitch, dyaw)
    """

    def __init__(
        self,
        demo_trajs: list[np.ndarray],
        demo_vels: list[np.ndarray],
        dt: float=0.01,
        switch: bool=False,
        backtrack_steps: int=0
    ):
        """
        Args:
            demo_trajs: List of demo trajectories [np.ndarray(n_steps, n_dim=x_dim+r_dim)]. Assume n_steps can vary, n_dim is fixed.
            demo_vels: List of velocities [np.ndarray(n_steps, n_dim=x_dim)]. Assume n_steps can vary, n_dim is fixed.
            pos_model_path: String of NODE model file path
            quat_model_path: String of quaternion model file path
            dt: Time step
            switch: If True, allows switching between trajectories at runtime. If False, sticks to initially chosen trajectory.
            backtrack_steps: Number of steps to backtrack on collision
            key: jax.random.PRNGKey
        """
        if demo_trajs[0].shape[1] == 7:  # quaternion
            self.x_dim = 3
            self.r_dim = 4
        else:
            raise ValueError(f"Invalid demo trajectory shape: {demo_trajs[0].shape}")

        self.demo_trajs = demo_trajs
        self.demo_trajs_flat = np.concatenate(demo_trajs, axis=0)
        self.demo_segment_idx = np.cumsum(
            [0] + [traj.shape[0] for traj in demo_trajs]
        )[:-1]
        self.demo_vels = demo_vels
        self.dt = dt
        self.switch = switch
        self.backtrack_steps = backtrack_steps
            
        self.ref_traj_idx = None
        self.ref_point_idx = None

        self.pos_model = None
        self.quat_model = None
        self.p_in, self.q_in, self.p_out, self.q_out, self.p_init, self.q_init, self.p_att, self.q_att = (
            self.quad_data_preprocess(demo_trajs)
        )
        

    def load_pos_model(self, model_path: str):
        model_name = os.path.basename(model_path)
        width_str = model_name.split("width")[1].split("_")[0]
        depth_str = model_name.split("depth")[1].split(".")[0]
        width_size = int(width_str)
        depth = int(depth_str)
        self.pos_model = NeuralODE(self.x_dim, width_size, depth)
        self.pos_model.load_state_dict(torch.load(model_path, weights_only=True))
        self.pos_model.eval()
    
    def train_pos_model(self, save_path: str, batch_size: int=1, lr_strategy: tuple=(1e-3, 1e-3, 1e-3), steps_strategy: tuple=(5000, 5000, 5000), length_strategy: tuple=(0.4, 0.7, 1), plot: bool=True, print_every: int=100):
        """
        Args:
            save_path: path to save the trained model
            batch_size: number of trajectories to train on at a time
            lr_strategy: learning rate in each phase
            steps_strategy: number of steps in each phase
            length_strategy: fraction of the trajectories available for training in each phase
            plot: whether to plot the training progress
            print_every: print the training progress every print_every steps
        """
        width_str = save_path.split("width")[1].split("_")[0]
        depth_str = save_path.split("depth")[1].split(".")[0]
        width_size = int(width_str)
        depth = int(depth_str)
        self.pos_model = train(
            [traj[:, :self.x_dim] for traj in self.demo_trajs],
            self.demo_vels,
            save_path,
            data_size=self.x_dim,
            batch_size=batch_size,
            lr_strategy=lr_strategy,
            steps_strategy=steps_strategy,
            length_strategy=length_strategy,
            width_size=width_size,
            depth=depth,
            plot=plot,
            print_every=print_every
        )
        self.pos_model.eval()

    def pos_data_preprocess(self):
        """
        Process position data from demo trajectories to create position-velocity pairs for training.

        Returns:
            x: np.ndarray of shape (N, x_dim) containing positions
            x_dot: np.ndarray of shape (N, x_dim) containing velocities
            TODO: not used
        """
        # Initialize lists to store positions and velocities
        x_list = []
        x_dot_list = []

        # Process each trajectory
        for traj in self.demo_trajs:
            # Extract positions (first x_dim components)
            positions = traj[:, : self.x_dim]

            # Compute velocities using finite differences
            velocities = np.diff(positions, axis=0) / self.dt

            # Add to lists (excluding last position since it has no velocity)
            x_list.append(positions[:-1])
            x_dot_list.append(velocities)

        # Concatenate all positions and velocities
        x = np.concatenate(x_list, axis=0)
        x_dot = np.concatenate(x_dot_list, axis=0)

        return x, x_dot

    def train_quat_model(self, save_path: str, k_init: int=10):
        quat_obj = quat_class(
            self.q_in, self.q_out, self.q_att, self.dt, K_init=k_init, output_path=save_path
        )
        quat_obj.begin()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        quat_obj._logOut()

        self.quat_model = quat_obj

    def load_quat_model(self, model_path: str):
        with open(model_path, "r") as f:
            model_data = json.load(f)

        # Extract parameters from the loaded data
        K = model_data.get("K")
        M = model_data.get("M")
        dt = model_data.get("dt")
        assert dt == self.dt
        q_att_array = np.array(model_data.get("att_ori"))
        q_att = R.from_quat(q_att_array)

        self.quat_model = quat_class(self.q_in, self.q_out, self.q_att, self.dt, K_init=K)

        Prior = np.array(model_data.get("Prior"))
        Mu_flat = np.array(model_data.get("Mu"))
        Sigma_flat = np.array(model_data.get("Sigma"))

        Mu = Mu_flat.reshape(2 * K, 4)
        Sigma = Sigma_flat.reshape(2 * K, 4, 4)

        Mu_rot = [R.from_quat(mu) for mu in Mu]

        self.quat_model.gmm = gmm_class(self.q_in, self.q_att, K)
        self.quat_model.gmm.Prior = Prior
        self.quat_model.gmm.Mu = Mu_rot
        self.quat_model.gmm.Sigma = Sigma

        A_ori_flat = np.array(model_data.get("A_ori"))
        self.quat_model.A_ori = A_ori_flat.reshape(2 * K, 4, 4)

        self.quat_model.K = K
        self.quat_model.dt = dt
        self.quat_model.q_att = q_att

    def quad_data_preprocess(self, trajs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p_raw = []  # Position trajectories
        q_raw = []  # Orientation trajectories
        t_raw = []  # Time vectors
        for i, traj in enumerate(trajs):
            # Verify that the data has the expected format (n, 7)
            if traj.shape[1] != 7:
                print(
                    f"Warning: Trajectory {i} has unexpected shape {traj.shape}. Expected (n, 7). Skipping."
                )
                continue

            # Extract positions (first 3 columns) and quaternions (last 4 columns)
            positions = traj[:, :3]
            quaternions = traj[:, 3:7]

            # Create time vector based on dt
            times = np.arange(0, len(positions) * self.dt, self.dt)[: len(positions)]

            # Convert quaternions to Rotation objects
            rotations = [R.from_quat(quat) for quat in quaternions]

            # Append to raw data lists
            p_raw.append(positions)
            q_raw.append(rotations)
            t_raw.append(times)

        # Process data
        p_in, q_in, t_in = pre_process(p_raw, q_raw, t_raw, opt="savgol")
        p_out, q_out = compute_output(p_in, q_in, t_in)
        p_init, q_init, p_att, q_att = extract_state(p_in, q_in)
        p_in, q_in, p_out, q_out = rollout_list(p_in, q_in, p_out, q_out)

        return p_in, q_in, p_out, q_out, p_init, q_init, p_att, q_att

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Args:
            state: position + quaternion
        Returns:
            action: np.ndarray(dx, dy, dz, droll, dpitch, dyaw)
        """

        x_dot: np.ndarray = self.get_x_dot(state[: self.x_dim], alpha_V=50.0, lookahead=5)
        r_dot: np.ndarray = self.get_r_dot(state[self.x_dim : self.x_dim + self.r_dim])
        return np.concatenate([x_dot, r_dot])

    def get_x_dot(self, x_state: np.ndarray, alpha_V: float=20.0, lookahead: int=5) -> np.ndarray:
        """
        Args:
            state: np.ndarray(n_dims)
            alpha_V: float, CLF parameter, controls how close to follow the reference trajectory
            lookahead: number of points to look ahead on the chosen trajectory for smoother tracking
        Returns:
            x_dot: np.ndarray(dx, dy, dz)
        """
        if self.switch or self.ref_traj_idx is None or self.ref_point_idx is None:
            self.ref_traj_idx, self.ref_point_idx = self.choose_traj(x_state)
        self.ref_point_idx = min(self.ref_point_idx+lookahead, self.demo_trajs[self.ref_traj_idx].shape[0] - 1)
        qp = OSQP()
        x_dot, u_star = self.compute_clf(
            x_state,
            self.demo_trajs[self.ref_traj_idx][self.ref_point_idx][: self.x_dim],
            qp,
            self.pos_model,
            alpha_V,
        )
        return x_dot

    def get_r_dot(self, r_state: np.ndarray) -> np.ndarray:
        """
        Args:
            state: np.ndarray(n_dims)
        Returns:
            r_dot: np.ndarray(droll, dpitch, dyaw)
        """
        if self.r_dim == 0:
            return np.array([])

        # Extract quaternion from state
        q_curr = R.from_quat(r_state)

        # Compute output using quaternion DS
        _, _, omega = self.quat_model._step(q_curr, self.dt)

        return omega

    def compute_clf(self, current_state: np.ndarray, ref_state: np.ndarray, qp: OSQP, model: NeuralODE, alpha_V: float) -> tuple[np.ndarray, np.ndarray]:
        # Compute tracking error
        error = current_state - ref_state

        # CLF (Control Lyapunov Function)
        V = np.sum(np.square(error)) / 2

        with torch.no_grad():
            f_x: np.ndarray = model.forward(torch.tensor(current_state, dtype=torch.float32)).numpy()
            f_xref: np.ndarray = model.forward(torch.tensor(ref_state, dtype=torch.float32)).numpy()

        # Compute CLF derivative terms
        s = np.dot(error, f_x - f_xref)

        # QP parameters for CLF-based control
        Q_opt = np.eye(3) # Cost on control input
        G_opt = 2 * error.reshape(1, -1)  # CLF derivative terms
        h_opt = np.array([-alpha_V * V - 2 * s])  # CLF constraint

        sol = qp.run(
            params_obj=(Q_opt, np.zeros(3)), params_ineq=(G_opt, h_opt)
        ).params
        u_star = sol.primal.reshape(-1)

        # Apply control input and integrate dynamics
        x_dot = f_x + u_star

        return x_dot, u_star

    def choose_traj(self, state: np.ndarray) -> tuple[int, int]:
        """
        Choose trajectory index based on state.
        TODO: for now uses Euclidean distance.
        Args:
            state: np.ndarray(n_dims)
        Returns:
            traj_idx: int
            point_idx within the traj: int
        """
        # Compute Euclidean distances between state and all points in flattened trajectories
        distances = np.sqrt(
            np.sum((self.demo_trajs_flat[:, : self.x_dim] - state) ** 2, axis=1)
        )
        # Find the index of the closest point in flattened trajectories
        closest_idx = np.argmin(distances)
        # Determine which trajectory the closest point belongs to
        traj_idx = (
            np.searchsorted(self.demo_segment_idx, closest_idx, side="right") - 1
        )
        # Compute point index within the chosen trajectory
        if traj_idx == 0:
            point_idx = closest_idx
        else:
            point_idx = closest_idx - self.demo_segment_idx[traj_idx]

        return int(traj_idx), int(point_idx)
