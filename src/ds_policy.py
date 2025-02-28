import os
import sys
import jax.random as jrandom
import time
from node_clf import NeuralODE_rot
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxopt import OSQP

# Add the quaternion_ds package to sys.path to import it
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'se3_lpvds/src/quaternion_ds'))

import json

from train_node_clf import train
from scipy.spatial.transform import Rotation as R
from src.gmm_class import gmm_class
from src.quat_class import quat_class
from src.util.process_tools import pre_process, compute_output, extract_state, rollout_list


class DSPolicy:
    """
    __init__(model_path: str,
        demo_trajs: List[np.ndarray(n_steps, n_dim=x_dim+r_dim)],
        dt: float,
        switch: bool,
        backtrack_steps: int,
        key: jax.random.PRNGKey)
    get_x_dot(state: np.ndarray(n_dims), alpha_V: float, lookahead: int) -> np.ndarray(x_dim)
    get_r_dot(state: np.ndarray(n_dims)) -> np.ndarray(r_dim)
    """

    def __init__(
        self, demo_trajs, pos_model_path=None, quat_model_path=None, dt=0.01, switch=False, backtrack_steps=0, key=None
    ):
        """
        Args:
            demo_trajs: List of demo trajectories [np.ndarray(n_steps, n_dim=x_dim+r_dim)]. Assume n_steps can vary, n_dim is fixed.
            pos_model_path: String of NODE model file path
            quat_model_path: String of quaternion model file path
            dt: Time step
            switch: If True, allows switching between trajectories at runtime. If False, sticks to initially chosen trajectory.
            backtrack_steps: Number of steps to backtrack on collision
            key: jax.random.PRNGKey
        """
        if demo_trajs[0].shape[1] == 7: # quaternion
            self.x_dim = 3
            self.r_dim = 4
        elif demo_trajs[0].shape[1] == 3: # position
            self.x_dim = 3
            self.r_dim = 0
        else:
            raise ValueError(f"Invalid demo trajectory shape: {demo_trajs[0].shape}")

        self.demo_trajs = demo_trajs
        self.demo_trajs_flat = jnp.concatenate([traj for traj in demo_trajs], axis=0)
        self.demo_trajs_segment_idx = np.cumsum(
            [0] + [traj.shape[0] for traj in demo_trajs]
        )[:-1]
        self.dt = dt
        self.switch = switch
        self.backtrack_steps = backtrack_steps

        self.ref_traj_idx = None
        self.ref_point_idx = None

        # NODE model
        if pos_model_path is None:
            self.pos_model = self.train_pos_model(demo_trajs, self.dt, key)
        else:
            model_name = os.path.basename(pos_model_path)
            width_str = model_name.split("width")[1].split("_")[0]
            depth_str = model_name.split("depth")[1].split(".")[0]
            width_size = int(width_str)
            depth = int(depth_str)
            if key is None:
                key = jrandom.PRNGKey(int(time.time()))
            template_model = NeuralODE_rot(self.x_dim, width_size, depth, key=key)
            self.pos_model = eqx.tree_deserialise_leaves(pos_model_path, template_model)
        
        # quaternion DS model
        p_in, q_in, p_out, q_out, p_init, q_init, p_att, q_att = self.quad_data_preprocess(demo_trajs)
        if quat_model_path is None:
            self.quat_model = self.train_quat_model(q_in, q_out, q_att)
        else:
            self.quat_model = self.load_quat_model(quat_model_path, q_in, q_out)
    
    def train_pos_model(self, key):
        model_path = 'neural_ode/models/mlp_width64_depth3.eqx'
        x, x_dot = pos_data_preprocess()
        return train(model_path, x, x_dot, data_size=3, batch_size=1, lr_strategy=(1e-3, 1e-3, 1e-3), steps_strategy=(5000, 5000, 5000), length_strategy=(0.4, 0.7, 1), width_size=64, depth=3, seed=1000, plot=True, print_every=100, save_every=1000)

    def pos_data_preprocess(self):
        
        

    def train_quat_model(self, q_in, q_out, q_att, k_init=10):
        output_path = 'neural_ode/models/quat_model.json'
        quat_obj = quat_class(q_in, q_out, q_att, self.dt, K_init=k_init, output_path=output_path)
        quat_obj.begin()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        quat_obj._logOut()

        return quat_obj

    def load_quat_model(self, quat_model_path, q_in, q_out):
        with open(quat_model_path, 'r') as f:
            model_data = json.load(f)
    
        # Extract parameters from the loaded data
        K = model_data.get('K')
        M = model_data.get('M')
        dt = model_data.get('dt')
        assert dt == self.dt
        q_att_array = np.array(model_data.get('att_ori'))
        q_att = R.from_quat(q_att_array)

        quat_obj = quat_class(q_in, q_out, q_att, dt, K_init=K)

        Prior = np.array(model_data.get('Prior'))
        Mu_flat = np.array(model_data.get('Mu'))
        Sigma_flat = np.array(model_data.get('Sigma'))
        
        Mu = Mu_flat.reshape(2*K, 4)
        Sigma = Sigma_flat.reshape(2*K, 4, 4)
        
        Mu_rot = [R.from_quat(mu) for mu in Mu]
        
        quat_obj.gmm = gmm_class(q_in, q_att, K)
        quat_obj.gmm.Prior = Prior
        quat_obj.gmm.Mu = Mu_rot
        quat_obj.gmm.Sigma = Sigma
        
        A_ori_flat = np.array(model_data.get('A_ori'))
        quat_obj.A_ori = A_ori_flat.reshape(2*K, 4, 4)
        
        quat_obj.K = K
        quat_obj.dt = dt
        quat_obj.q_att = q_att

        return quat_obj

    def quad_data_preprocess(self, trajs):
        p_raw = []  # Position trajectories
        q_raw = []  # Orientation trajectories
        t_raw = []  # Time vectors
        for i, traj in enumerate(trajs):
            # Verify that the data has the expected format (n, 7)
            if traj.shape[1] != 7:
                print(f"Warning: Trajectory {i} has unexpected shape {traj.shape}. Expected (n, 7). Skipping.")
                continue
            
            # Extract positions (first 3 columns) and quaternions (last 4 columns)
            positions = traj[:, :3]
            quaternions = traj[:, 3:7]
            
            # Create time vector based on dt
            times = np.arange(0, len(positions) * self.dt, self.dt)[:len(positions)]
            
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

    def get_action(self, state):
        """
        Args:
            state: np.ndarray(n_dims)
        Returns:
            action: np.ndarray(dx, dy, dz, droll, dpitch, dyaw)
        """
        # x_dot = self.get_x_dot(state)
        x_dot = np.zeros(self.x_dim)
        r_dot = self.get_r_dot(state)
        return jnp.concatenate([x_dot, r_dot])

    def get_x_dot(self, state, alpha_V=20.0, lookahead=5):
        """
        Args:
            state: np.ndarray(n_dims)
            alpha_V: float, CLF parameter, controls how close to follow the reference trajectory
            lookahead: number of points to look ahead on the chosen trajectory for smoother tracking, TODO: not used for now
        Returns:
            x_dot: np.ndarray(dx, dy, dz)
        """
        if self.switch or self.ref_traj_idx is None or self.ref_point_idx is None:
            self.ref_traj_idx, self.ref_point_idx = self.choose_traj(state)
        qp = OSQP()
        x_dot, u_star = self.compute_clf(
            state[:self.x_dim],
            self.demo_trajs[self.ref_traj_idx][self.ref_point_idx][:self.x_dim],
            qp,
            self.pos_model,
            alpha_V,
        )
        return x_dot

    def get_r_dot(self, state):
        """
        Args:
            state: np.ndarray(n_dims)
        Returns:
            r_dot: np.ndarray(droll, dpitch, dyaw)
        """
        if self.r_dim == 0:
            return np.array([])
        
        # Extract quaternion from state
        q_curr = R.from_quat(state[self.x_dim:self.x_dim+self.r_dim])
        
        # Compute output using quaternion DS
        _, _, omega = self.quat_model._step(q_curr, self.dt)
        
        return omega

    def compute_clf(self, current_state, ref_state, qp, model, alpha_V):
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
        sol = qp.run(
            params_obj=(Q_opt, jnp.zeros(3)), params_ineq=(G_opt, h_opt)
        ).params
        u_star = sol.primal.reshape(-1)

        # Apply control input and integrate dynamics
        x_dot = f_x + u_star

        return x_dot, u_star

    def choose_traj(self, state) -> tuple[int, int]:
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
        distances = jnp.sqrt(jnp.sum((self.demo_trajs_flat - state) ** 2, axis=1))
        # Find the index of the closest point in flattened trajectories
        closest_idx = jnp.argmin(distances)
        # Determine which trajectory the closest point belongs to
        traj_idx = (
            jnp.searchsorted(self.demo_trajs_segment_idx, closest_idx, side="right") - 1
        )
        # Compute point index within the chosen trajectory
        if traj_idx == 0:
            point_idx = closest_idx
        else:
            point_idx = closest_idx - self.demo_trajs_segment_idx[traj_idx]

        return int(traj_idx), int(point_idx)


if __name__ == "__main__":
    model_path = "neural_ode/models/mlp_width64_depth3.eqx"
    demo_trajs = [np.zeros((10, 7)), np.ones((5, 7))]
    policy = NODEPolicy(model_path, demo_trajs)
    print(policy.demo_trajs_flat.shape)
    print(policy.demo_trajs_segment_idx)
