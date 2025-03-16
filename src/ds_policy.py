import os
import sys
import logging
import time

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "DS-Policy/src/",
    )
)
from neural_ode import NeuralODE
import numpy as np
import torch
from jaxopt import OSQP
import jax
import jax.numpy as jnp
from scipy import sparse  # Add this import for sparse matrices
import json
from scipy.spatial.transform import Rotation as R

from train_neural_ode import train
from ds_utils import quat_mult, Quaternion

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "se3_lpvds/src/quaternion_ds",
    )
)
from src.gmm_class import gmm_class
from src.quat_class import quat_class
from src.util.process_tools import (
    pre_process,
    compute_output,
    extract_state,
    rollout_list,
)


qp = OSQP()


@jax.jit
def qp_solve(Q_opt, G_opt, h_opt):
    return qp.run(params_obj=(Q_opt, jnp.zeros(3)), params_ineq=(G_opt, h_opt)).params


class DSPolicy:
    """
    NOTE: train_model/load_model uses a neural ODE for both position and orientation.
          train_pos_model/load_pos_model uses a neural ODE for position only.
          train_quat_model/load_quat_model uses a SE3 LPVDS for orientation only.
          Both position and orientation models have to be trained/loaded before get_action
    get_action(state: np.ndarray(n_dims)) -> (dx, dy, dz, droll, dpitch, dyaw)
    """

    def __init__(
        self,
        x: list[np.ndarray],
        x_dot: list[np.ndarray],
        quat: list[np.ndarray],
        omega: list[np.ndarray],
        dt: float = 0.01,
        switch: bool = False,
        lookahead: int = 5,
        demo_traj_probs: np.ndarray = None,
        backtrack_steps: int = 0,
    ):
        """
        Args:
            x: list of position trajectories
            x_dot: list of velocity trajectories
            quat: list of quaternion trajectories
            omega: list of angular velocity trajectories
            dt: Time step
            switch: If True, allows switching between trajectories at runtime. If False, sticks to initially chosen trajectory.
            lookahead: Number of steps to lookahead for reference trajectory selection
            backtrack_steps: Number of steps to backtrack on collision, TODO: not used
        """
        self.x_dim = 3
        self.r_dim = 4
        self.dt = dt
        self.x = x
        self.x_dot = x_dot
        self.quat = quat
        self.omega = omega
        self.switch = switch
        self.lookahead = lookahead
        self.backtrack_steps = backtrack_steps

        self.demo_traj_probs = (
            np.ones(len(x)) if demo_traj_probs is None else demo_traj_probs
        )

        self.ref_traj_idx = None
        self.ref_point_idx = None

        # models
        self.model = None
        self.model_reverse = None
        self.pos_model = None
        self.pos_model_reverse = None
        self.quat_model = None
        (
            self.p_in,
            self.q_in,
            self.p_out,
            self.q_out,
            self.p_init,
            self.q_init,
            self.p_att,
            self.q_att,
        ) = self.quad_data_preprocess(
            [np.concatenate([x[i], quat[i]], axis=-1) for i in range(len(x))]
        )

    def load_model(self, model_path: str):
        model_name = os.path.basename(model_path)
        width_str = model_name.split("width")[1].split("_")[0]
        depth_str = model_name.split("depth")[1].split("_")[0]
        width_size = int(width_str)
        depth = int(depth_str)
        self.model = NeuralODE(self.x_dim + self.r_dim, width_size, depth)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def train_model(
        self,
        save_path: str,
        batch_size: int = 1,
        lr_strategy: tuple = (1e-3, 1e-4, 1e-5),
        steps_strategy: tuple = (5000, 5000, 5000),
        length_strategy: tuple = (0.4, 0.7, 1),
        plot: bool = True,
        print_every: int = 100,
    ):
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
        depth_str = save_path.split("depth")[1].split("_")[0]
        width_size = int(width_str)
        depth = int(depth_str)
        self.model = train(
            [
                np.concatenate([self.x[i], self.quat[i]], axis=-1)
                for i in range(len(self.x))
            ],
            [
                np.concatenate([self.x_dot[i], self.omega[i]], axis=-1)
                for i in range(len(self.x_dot))
            ],
            save_path,
            batch_size=batch_size,
            lr_strategy=lr_strategy,
            steps_strategy=steps_strategy,
            length_strategy=length_strategy,
            width_size=width_size,
            depth=depth,
            plot=plot,
            print_every=print_every,
        )
        self.model.eval()

    def load_pos_model(self, pos_model_path: str):
        pos_model_name = os.path.basename(pos_model_path)
        width_str = pos_model_name.split("width")[1].split("_")[0]
        depth_str = pos_model_name.split("depth")[1].split(".")[0]
        width_size = int(width_str)
        depth = int(depth_str)
        self.pos_model = NeuralODE(self.x_dim, width_size, depth)
        self.pos_model.load_state_dict(torch.load(pos_model_path, weights_only=True))
        self.pos_model.eval()

        # pos_backtrack_model_name = os.path.basename(pos_backtrack_model_path)
        # width_str = pos_backtrack_model_name.split("width")[1].split("_")[0]
        # depth_str = pos_backtrack_model_name.split("depth")[1].split(".")[0]
        # width_size = int(width_str)
        # depth = int(depth_str)
        # self.pos_backtrack_model = NeuralODE(self.x_dim, width_size, depth)
        # self.pos_backtrack_model.load_state_dict(torch.load(pos_backtrack_model_path, weights_only=True))
        # self.pos_backtrack_model.eval()

    def train_pos_model(
        self,
        save_path: str,
        batch_size: int = 1,
        lr_strategy: tuple = (1e-3, 1e-4, 1e-5),
        steps_strategy: tuple = (5000, 5000, 5000),
        length_strategy: tuple = (0.4, 0.7, 1),
        plot: bool = True,
        print_every: int = 100,
    ):
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
            self.demo_pos,
            self.demo_vel,
            save_path,
            data_size=self.x_dim,
            batch_size=batch_size,
            lr_strategy=lr_strategy,
            steps_strategy=steps_strategy,
            length_strategy=length_strategy,
            width_size=width_size,
            depth=depth,
            plot=plot,
            print_every=print_every,
        )
        self.pos_model.eval()
        # self.pos_backtrack_model = train(
        #     [np.flip(pos, axis=0) for pos in self.demo_pos],
        #     [np.flip(vel, axis=0) for vel in self.demo_vel],
        #     save_path+"_backtrack",
        #     data_size=self.x_dim,
        #     batch_size=batch_size,
        #     lr_strategy=lr_strategy,
        #     steps_strategy=steps_strategy,
        #     length_strategy=length_strategy,
        #     width_size=width_size,
        #     depth=depth,
        #     plot=plot,
        # )
        # self.pos_backtrack_model.eval()

    def model_forward(self, state: np.ndarray):
        if self.model is None:  # pos_model and quat_model are separate
            with torch.no_grad():
                x_dot = self.pos_model.forward(state[: self.x_dim]).numpy()

            q_curr = R.from_quat(state[self.x_dim :])

            _, _, omega = self.quat_model._step(q_curr, self.dt)

            return np.concatenate([x_dot, omega])
        else:
            with torch.no_grad():
                return self.model.forward(state).numpy()

    def compute_vel(self, x: np.ndarray):
        """
        Compute velocities from trajectories
        NOTE: not used
        """
        x_dot = np.diff(x, axis=0) / self.dt
        x_dot = np.vstack([x_dot, x_dot[-1]])

        return x_dot

    def train_quat_model(self, save_path: str, k_init: int = 10):
        quat_obj = quat_class(
            self.q_in,
            self.q_out,
            self.q_att,
            self.dt,
            K_init=k_init,
            output_path=save_path,
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

        self.quat_model = quat_class(
            self.q_in, self.q_out, self.q_att, self.dt, K_init=K
        )

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

    def quad_data_preprocess(self, trajs: list[np.ndarray]) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
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

    def get_action(
        self,
        state: np.ndarray,
        clf: bool = True,
        alpha_V: float = 20.0,
        lookahead: int = None,
    ) -> np.ndarray:
        """
        Args:
            state: position + quaternion
            clf: if True, apply CLF
            alpha_V: CLF parameter
            lookahead: Number of steps to lookahead for reference trajectory selection, if None, use self.lookahead, otherwise update self.lookahead
        Returns:
            action: np.ndarray(dx, dy, dz, droll, dpitch, dyaw)
        """
        if lookahead is not None:
            self.lookahead = lookahead

        action_from_model = self.model_forward(
            state
        )  # action from model, with no CLF/CBF
        if clf:
            action = self.apply_clf(state, action_from_model, alpha_V)
        else:
            action = action_from_model
        return action

    def apply_clf(
        self,
        current_state: np.ndarray,
        action_from_model: np.ndarray,
        alpha_V: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.ref_traj_idx, self.ref_point_idx = self.choose_ref(
            current_state[: self.x_dim], self.switch
        )
        ref_x = self.x[self.ref_traj_idx][self.ref_point_idx]
        ref_quat = self.quat[self.ref_traj_idx][self.ref_point_idx]
        ref_state = np.concatenate([ref_x, ref_quat])

        if (
            self.model is None
        ):  # pos_model and quat_model are separate, only apply to pos_model
            current_x = current_state[: self.x_dim]
            f_x = action_from_model[: self.x_dim]
            with torch.no_grad():
                f_xref = jnp.array(self.pos_model.forward(ref_x))

            error = jnp.array(current_x) - jnp.array(ref_x)

            # CLF (Control Lyapunov Function)
            V = jnp.sum(jnp.square(error)) / 2

            # Compute CLF derivative terms
            s = jnp.dot(error, f_x - f_xref)

            # QP parameters for CLF-based control
            Q_opt = jnp.eye(3)  # Cost on control input
            G_opt = 2 * error[:3].reshape(1, -1)  # CLF derivative terms
            h_opt = jnp.array([-alpha_V * V - 2 * s])  # CLF constraint

            u_star = qp_solve(Q_opt, G_opt, h_opt).primal.reshape(-1)

            action = action_from_model
            action[:3] = f_x[:3] + u_star

            return action

        else:  # unified model NOTE: not good
            quat = jnp.array(current_state[3:]).reshape((-1, 1))
            # Scaling factors for position and quaternion errors in the cost function
            scale_q = 1
            scale_p = 1

            # Construct the cost matrix for the QP
            Q_opt = jnp.vstack(
                (
                    jnp.hstack((scale_p * jnp.eye(3), jnp.zeros((3, 3)))),
                    jnp.hstack((jnp.zeros((3, 3)), scale_q * jnp.eye(3))),
                )
            )

            # Compute error between current and reference state
            err = jnp.array(current_state) - jnp.array(ref_state)
            # Normalize quaternion error
            err = err.at[3:].set(err[3:] / jnp.linalg.norm(err[3:]))
            err_pos = err[:3]  # Position error
            err_quat = err[3:]  # Quaternion error

            # Lyapunov function (quadratic in the error)
            V = jnp.sum(jnp.square(2 * err)) / 4
            # Get nominal dynamics from the learned model
            vel = jnp.array(action_from_model).reshape((-1, 1))

            vel_pos = vel[:3]  # Position dynamics

            # Angular velocity from the model
            vel_ang = vel[3:].reshape((-1, 1))
            vel_ang_quat = jnp.vstack(
                (0.0, vel_ang)
            )  # Prepend 0 for quaternion multiplication

            # Get reference dynamics TODO
            with torch.no_grad():
                vel_ref = jnp.array(self.model.forward(ref_state)).reshape((-1, 1))
            vel_pos_ref = vel_ref[:3].reshape((-1, 1))
            vel_ang_ref = vel_ref[3:].reshape((-1, 1))
            # vel_pos_ref = jnp.array(
            #     self.x_dot[self.ref_traj_idx][self.ref_point_idx]
            # ).reshape((-1, 1))
            # vel_ang_ref = jnp.array(
            #     self.omega[self.ref_traj_idx][self.ref_point_idx]
            # ).reshape((-1, 1))
            vel_ang_quat_ref = jnp.vstack(
                (0.0, vel_ang_ref)
            )  # Prepend 0 for quaternion multiplication

            # Compute the CLF derivative terms
            s_p = jnp.vdot(err_pos, vel_pos - vel_pos_ref)  # Position term
            s_q = jnp.vdot(
                err_quat / 2.0,
                quat_mult(vel_ang_quat[:, 0], quat).reshape((-1, 1))
                - quat_mult(vel_ang_quat_ref[:, 0], ref_quat.reshape((-1, 1))).reshape(
                    (-1, 1)
                ),
            )  # Quaternion term

            s = s_p + s_q  # Combined CLF derivative

            # Construct quaternion matrix for quaternion dynamics
            Q = jnp.array(
                [
                    [quat[0, 0], quat[1, 0], quat[2, 0], quat[3, 0]],
                    [-quat[1, 0], quat[0, 0], quat[3, 0], -quat[2, 0]],
                    [-quat[2, 0], -quat[3, 0], quat[0, 0], quat[1, 0]],
                    [-quat[3, 0], quat[2, 0], -quat[1, 0], quat[0, 0]],
                ]
            )
            Q2 = Q[:, 1:]  # 4 x 3
            Q2_minus = -Q2
            Q2_1 = jnp.vstack((Q2_minus[0, :], -Q2_minus[1:, :]))

            # Construct the CLF constraint for the QP
            G_opt = 2 * jnp.hstack((err_pos.T, (err_quat.T / 2.0) @ Q2_1)).reshape(
                1, -1
            )  # Ensure shape (1,6)
            h_opt = jnp.array([-alpha_V * V - 2 * s]).reshape(-1)  # Ensure shape (1,)

            # Initialize QP solver
            qp = OSQP()

            # Solve the QP to find optimal control inputs
            sol = qp.run(
                params_obj=(Q_opt, jnp.zeros(Q_opt.shape[0])),
                params_ineq=(G_opt, h_opt),
            ).params

            ustar = sol.primal.reshape((-1, 1))

            # Extract position and angular velocity control inputs
            u_pos = ustar[:3]
            u_ang_quat = jnp.vstack((0.0, ustar[3:]))

            # Compute state derivatives with the control inputs
            p_dot = 0.1 * vel_pos + u_pos  # Position derivative
            q_dot = 0.5 * quat_mult(
                vel_ang_quat[:, 0] + u_ang_quat[:, 0], quat
            ).reshape((-1, 1))

            q_dot_Quat = Quaternion(q_dot[0][0], q_dot[1:].reshape(-1))
            q_dot_Quat.normalize()
            # Combine into full state derivative
            x_dot_cmd = np.concatenate(
                (p_dot.reshape(-1), q_dot_Quat.axis_angle().reshape(-1))
            )

            return x_dot_cmd

    def choose_ref(self, x_state: np.ndarray, switch: bool = False) -> tuple[int, int]:
        """
        Choose ref_traj_idx and ref_point_idx based on state.
        Criteria:
            - for each demo trajectory i, compute smallest distance to x_state among all its points: closest_distances[i], corresponding point index: closest_indices[i]
            - score[i] = log(p[i]) - closest_distances[i] TODO: this could be modified
            - prob_dist = softmax(score)
            - choose traj_idx = argmax(prob_dist), point_idx = closest_indices[traj_idx]
        Args:
            x_state: np.ndarray(x_dim)
            switch: bool, if False, sticks to the current trajectory
        Returns:
            ref_traj_idx: int
            point_idx within the traj: int
        """
        if switch or self.ref_traj_idx is None:
            # Calculate distances from current state to each point in each trajectory
            closest_indices = []
            closest_distances = []

            # Iterate through each trajectory
            for traj in self.x:
                distances = np.sqrt(np.sum((traj - x_state) ** 2, axis=1))
                closest_idx = np.argmin(distances)
                min_distance = distances[closest_idx]
                closest_indices.append(closest_idx)
                closest_distances.append(min_distance)

            scores = 10 * np.log(self.demo_traj_probs) - np.array(closest_distances)
            prob_dist = np.exp(scores) / np.sum(np.exp(scores))

            # Select the trajectory with the highest probability
            traj_idx = np.argmax(prob_dist)
            # Get the corresponding closest point index for the selected trajectory
            point_idx = closest_indices[traj_idx]
        else:  # stick to the current trajectory
            traj_idx = self.ref_traj_idx
            distances = np.sqrt(
                np.sum(
                    (self.x[self.ref_traj_idx] - x_state) ** 2,
                    axis=1,
                )
            )
            point_idx = np.argmin(distances)

        point_idx = min(point_idx + self.lookahead, self.x[traj_idx].shape[0] - 1)

        return int(traj_idx), int(point_idx)

    def update_demo_traj_probs(
        self, x_state: np.ndarray, radius: float, penalty: float, lookahead: int = None
    ):
        """
        Update the trajectory probabilities based on proximity to the current state.

        Args:
            x_state: np.ndarray - Current position state (x, y, z)
            radius: float - Threshold distance to consider a trajectory point as "close"
            penalty: float - Factor to reduce probability for trajectories with no close points
            lookahead: int - Number of steps to lookahead for reference trajectory selection, if None, use self.lookahead, otherwise update self.lookahead
        Returns:
            None - Updates self.demo_traj_probs in-place
        """
        if lookahead is not None:
            self.lookahead = lookahead

        for i, traj in enumerate(self.x):
            distances = np.sqrt(np.sum((traj - x_state) ** 2, axis=1))
            if np.any(distances < radius):
                self.demo_traj_probs[i] *= penalty

        self.ref_traj_idx, self.ref_point_idx = self.choose_ref(x_state, True)
