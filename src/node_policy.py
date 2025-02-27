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
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'se3_lpvds/src/quaternion_ds'))

import json
from scipy.spatial.transform import Rotation as R
from se3_lpvds.src.quaternion_ds.src.quat_class import quat_class, compute_ang_vel
from se3_lpvds.src.quaternion_ds.src.util import quat_tools


class NODEPolicy:
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
        self, model_path, demo_trajs, dt=0.01, switch=False, backtrack_steps=0, key=None
    ):
        """
        Args:
            model_path: String of model file path
            demo_trajs: List of demo trajectories [np.ndarray(n_steps, n_dim=x_dim+r_dim)]. Assume n_steps can vary, n_dim is fixed.
            dt: Time step
            switch: If True, allows switching between trajectories at runtime. If False, sticks to initially chosen trajectory.
            backtrack_steps: Number of steps to backtrack on collision
            key: jax.random.PRNGKey
        """
        if demo_trajs[0].shape[1] == 7:  # quaternion
            self.x_dim = 3
            self.r_dim = 4
        elif demo_trajs[0].shape[1] == 3:  # position
            self.x_dim = 3
            self.r_dim = 0
        else:
            raise ValueError(f"Invalid demo trajectory shape: {demo_trajs[0].shape}")

        model_name = os.path.basename(model_path)
        width_str = model_name.split("width")[1].split("_")[0]
        depth_str = model_name.split("depth")[1].split(".")[0]
        width_size = int(width_str)
        depth = int(depth_str)
        if key is None:
            key = jrandom.PRNGKey(int(time.time()))
        template_model = NeuralODE_rot(self.x_dim, width_size, depth, key=key)
        self.model = eqx.tree_deserialise_leaves(model_path, template_model)

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
        
        # Load quaternion DS model if r_dim is 4 (quaternion)
        if self.r_dim == 4:
            self.quat_ds = quat_class(self.demo_trajs[0][:, self.x_dim:self.x_dim+self.r_dim], self.demo_trajs[0][:, self.x_dim:self.x_dim+self.r_dim], R.from_quat(self.demo_trajs[0][0, self.x_dim:self.x_dim+self.r_dim]), self.dt, 4)

    def get_action(self, state):
        """
        Args:
            state: np.ndarray(n_dims)
        Returns:
            action: np.ndarray(dx, dy, dz, droll, dpitch, dyaw)
        """
        x_dot = self.get_x_dot(state)
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
        if self.switch:
            self.ref_traj_idx, self.ref_point_idx = self.choose_traj(state)
        qp = OSQP()
        x_dot, u_star = self.compute_clf(
            state,
            self.demo_trajs[self.ref_traj_idx][self.ref_point_idx],
            qp,
            self.model,
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
        omega = self.quat_ds.step(q_curr, self.dt)
        
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
