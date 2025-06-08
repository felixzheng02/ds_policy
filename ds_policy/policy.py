import logging
import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, List, Union, Callable
import matplotlib.pyplot as plt
from predicators.settings import CFG

from .neural_ode.neural_ode import NeuralODE
import numpy as np
import torch
from jaxopt import OSQP
import jax
import jax.numpy as jnp
import json
from scipy.spatial.transform import Rotation as R

from .neural_ode.train_neural_ode import train
from .so3_lpvds.quat_class import quat_class
from .so3_lpvds.process_tools import (
    pre_process,
    compute_output,
    extract_state,
    rollout_list,
)
from .se3_lpvds.src.se3_class import se3_class
from .se3_lpvds.src.gmm_class import gmm_class
from .se3_lpvds.src.util import process_tools, plot_tools

qp = OSQP()


@jax.jit
def qp_solve(Q_opt, G_opt, h_opt):
    return qp.run(params_obj=(Q_opt, jnp.zeros(Q_opt.shape[0])), params_ineq=(G_opt, h_opt)).params


# Configuration classes for DSPolicy
@dataclass
class PositionModelConfig:
    """Configuration for position model."""
    mode: Literal["none", "avg", "neural_ode"]
    
    # Neural ODE specific parameters
    load_path: Optional[str] = None
    width: int = 128
    depth: int = 3
    save_path: Optional[str] = None
    batch_size: int = 100
    device: str = "cpu"
    lr_strategy: Tuple[float, float, float] = (1e-3, 1e-4, 1e-5)
    epoch_strategy: Tuple[int, int, int] = (10, 10, 10)
    length_strategy: Tuple[float, float, float] = (0.4, 0.7, 1)
    plot: bool = True
    print_every: int = 10
    
    def __post_init__(self):
        if self.mode == "neural_ode" and self.load_path is None and self.save_path is None:
            raise ValueError("For neural_ode mode, either load_path or save_path must be provided")


@dataclass
class QuaternionModelConfig:
    """Configuration for quaternion model."""
    mode: Literal["none", "simple", "so3_lpvds"]
    
    # SO3-LPVDS specific parameters
    load_path: Optional[str] = None
    save_path: Optional[str] = None
    k_init: int = 10
    
    def __post_init__(self):
        if self.mode == "so3_lpvds" and self.load_path is None and self.save_path is None:
            raise ValueError("For so3_lpvds mode, either load_path or save_path must be provided")


@dataclass
class UnifiedModelConfig:
    """Configuration for unified position and orientation model."""
    mode: Literal["neural_ode", "se3_lpvds"]
    
    # ================================================
    # Neural ODE specific parameters
    # ================================================
    load_path: Optional[str] = None
    width: int = 256
    depth: int = 5
    save_path: Optional[str] = None
    batch_size: int = 100
    device: str = "cpu"
    lr_strategy: Tuple[float, float, float] = (1e-3, 1e-4, 1e-5)
    epoch_strategy: Tuple[int, int, int] = (50, 50, 50)
    length_strategy: Tuple[float, float, float] = (0.4, 0.7, 1)
    plot: bool = True
    print_every: int = 10
    
    # ================================================
    # SE3-LPVDS specific parameters
    # ================================================
    K_candidates: List[int] = field(default_factory=lambda: [1])
    
    # PD control parameters (for SE3-LPVDS near target)
    enable_simple_ds_near_target: bool = False
    simple_ds_pos_threshold: float = 0.05
    simple_ds_ori_threshold: float = 0.1
    K_pos: float = 1.0
    K_ori: float = 1.0

    # Resampling parameters
    attractor_resampling_mode: str = "Gaussian" # This determines how to resample the attractor. Could be "random", "nearest", "Gaussian"
    
    # def __post_init__(self):
    #     if self.load_path is None and self.save_path is None:
    #         raise ValueError(f"For {self.mode} mode, either load_path or save_path must be provided")


class DSPolicy:
    """
    Public methods:
        __init__: Initialize the DS policy with demonstration data and model configuration
        get_action: Generate control action for a given state
        update_demo_traj_probs: Update trajectory probabilities based on collision detection
        reset: Reset the trajectory probabilities to the initial values
        resample: Resample the attractor and adjust the policy accordingly
    """

    def __init__(
        self,
        x: list[np.ndarray],
        x_dot: list[np.ndarray],
        quat: list[np.ndarray],
        omega: list[np.ndarray],
        gripper: list[np.ndarray],
        pos_config: Optional[PositionModelConfig] = None,
        quat_config: Optional[QuaternionModelConfig] = None,
        unified_config: Optional[UnifiedModelConfig] = None,
        dt: float = 0.01,
        switch: bool = False,
        lookahead: int = 5,
        demo_traj_probs: np.ndarray = None,
        backtrack_steps: int = 0,
        relative_cluster_attractor: np.ndarray = None,
        **kwargs
    ):
        """
        Initialize the Dynamical System Policy with demonstration data and model configuration.
        
        Args:
            x: List of position trajectories from demonstrations
            x_dot: List of velocity trajectories from demonstrations
            quat: List of quaternion trajectories from demonstrations
            omega: List of angular velocity trajectories from demonstrations
            gripper: List of gripper trajectories from demonstrations
            pos_config: Configuration for position-only model
            quat_config: Configuration for orientation-only model
            unified_config: Configuration for unified position+orientation model
            dt: Time step for simulation and prediction
            switch: If True, allows switching between trajectories at runtime
            lookahead: Number of steps to lookahead for reference trajectory selection
            demo_traj_probs: Prior probabilities for each demonstration trajectory
            backtrack_steps: Number of steps to backtrack on collision
            **kwargs: Additional keyword arguments
        """
        # Validate configuration parameters
        if unified_config is not None and (pos_config is not None or quat_config is not None):
            raise ValueError("Cannot provide both unified_config and individual model configs")
        if unified_config is None and (pos_config is None or quat_config is None):
            raise ValueError("Must provide either unified_config or both pos_config and quat_config")

        self.dt = dt
        self.x = x
        self.x_dot = x_dot
        self.quat = quat
        self.omega = omega
        self.gripper = gripper
        self.switch = switch
        self.lookahead = lookahead
        self.backtrack_steps = backtrack_steps

        self.demo_traj_probs = np.ones(len(x)) if demo_traj_probs is None else demo_traj_probs
        self.demo_traj_scores = None
        self.ref_traj_idx = None
        self.ref_point_idx_lookahead = None
        self.ref_point_idx_no_lookahead = None

        # Initialize models to None
        self.model = None  # unified model
        self.pos_model = None
        self.quat_model = None

        # Initialize model flags
        self.pos_none = False
        self.pos_avg = False
        self.quat_none = False
        self.quat_simple = False
        self.se3_lpvds = False

        self.relative_cluster_attractor = relative_cluster_attractor

        # Process data for quat model
        (
            self.p_in,
            self.q_in,
            self.p_out,
            self.q_out,
            self.p_init,
            self.q_init,
            self.p_att,
            self.q_att,
        ) = self._quat_data_preprocess(
            [np.concatenate([x[i], quat[i]], axis=-1) for i in range(len(x))]
        )

        # Configure models
        self._configure_models(pos_config, quat_config, unified_config)
        self._validate_model_setup()

    def get_action(
        self,
        state: np.ndarray,
        clf: bool = True,
        alpha_V: float = 20.0,
        lookahead: int = None,
        backtrack: bool = False,
    ) -> np.ndarray:
        """
        Generate a control action for the given state.
        
        Computes a velocity command based on the current state, using the configured models.
        Can optionally apply Control Lyapunov Function (CLF) constraints for stability.
        
        Args:
            state: Current state vector [position (3), quaternion (4)] NOTE: quaternion needs to be in xyzw format
            clf: If True, apply Control Lyapunov Function constraints
            alpha_V: CLF convergence rate parameter
            lookahead: Steps to look ahead when selecting reference point (updates self.lookahead if provided)
            backtrack: If True, generate a backtracking action instead of a forward one
            
        Returns:
            action: Control action [linear velocity (3), angular velocity (3), gripper (1)]
        """
        if lookahead is not None:
            self.lookahead = lookahead

        self.ref_traj_idx, self.ref_point_idx_lookahead, self.ref_point_idx_no_lookahead = self._choose_ref(
            state[: 3], self.switch
        )

        if backtrack:
            # Use the negative demonstrated velocity at the reference point
            backtrack_vel_pos = -self.x_dot[self.ref_traj_idx][self.ref_point_idx_no_lookahead]
            backtrack_vel_ang = -self.omega[self.ref_traj_idx][self.ref_point_idx_no_lookahead]

            gripper_action = np.zeros(1)

            return np.concatenate([backtrack_vel_pos, backtrack_vel_ang, gripper_action])

        if self.se3_lpvds:
            # Check if near target and PD control is enabled
            p_curr = state[:3] + self.pos_shift
            q_curr = R.from_quat(state[3:]) * self.r_shift

            pos_dist = np.linalg.norm(p_curr - self.pos_att)
            q_err = self.r_att * q_curr.inv()
            ori_angle = np.linalg.norm(q_err.as_rotvec())

            if self.enable_simple_ds_near_target and pos_dist < self.simple_ds_pos_threshold and ori_angle < self.simple_ds_ori_threshold:
                # Use PD controller
                p_error = self.pos_att - p_curr
                ori_error_vec = q_err.as_rotvec()

                action_pos = self.K_pos * p_error 
                action_ang = self.K_ori * ori_error_vec

                gripper_action = np.zeros(1) # Assuming open/close is handled elsewhere when near target
                return np.concatenate([action_pos, action_ang, gripper_action])
            else:
                # Use SE3-LPVDS model
                p_next, q_next, gamma, v, w = self.model.step(p_curr, q_curr, self.dt)
                
                # apply modulations
                for obj_center, radius in self.modulations:
                    v = spherical_normal_modulation(p_curr, obj_center, radius, v)
                action_pos = v.flatten()
                action_ang = w.flatten()
                gripper_action = np.zeros(1)
                return np.concatenate([action_pos, action_ang, gripper_action])

        # Original logic for forward action
        action_raw = self._get_action_raw(
            state
        )
        if clf:
            action = self._apply_clf(state, action_raw, alpha_V)
        else:
            action = action_raw
        if len(self.gripper) > 0:
            gripper_action = self.gripper[self.ref_traj_idx][self.ref_point_idx_no_lookahead]
        else:
            gripper_action = np.zeros(1)
        return np.concatenate([action, gripper_action])

    def resample(self, state: np.ndarray = None):
        if self.se3_lpvds:
            new_pos_att, new_R_att = self.se3_lpvds_attractor_generator.sample()
            self.pos_shift = self.pos_att - new_pos_att
            self.r_shift = self.r_att * new_R_att.inv()
            if state is not None:
                self._add_modulation_point(state, radius=0.2)
        else:
            if state is not None:
                self._update_demo_traj_probs(state, "ref_point", 0.8)
        return

    def _shift_trajs(self, pos_att: np.ndarray, R_att: R):
        pos_shifted = []
        for l in range(len(self.x)):
            p_diff = pos_att - self.x[l][-1, :]
            pos_shifted.append(p_diff.reshape(1, -1) + self.x[l])

        R_shifted = []
        R_list = [R.from_quat(quat_arr) for quat_arr in self.quat]
        for l in range(len(self.quat)):
            R_diff = R_att * R_list[l][-1].inv()
            R_shifted.append([R_diff * r for r in R_list[l]])

        return pos_shifted, R_shifted

    def _add_modulation_point(self, state: np.ndarray, radius: float):
        """
        Args:
            state: Current state (x, y, z, qx, qy, qz, qw)
            radius: Radius of the modulation
        """
        self.modulations.append((state[:3], radius))

    def _update_demo_traj_probs(
        self, state: np.ndarray, mode: str, penalty: float, traj_threshold: float = 0.1, radius: float = 0.05, angle_threshold: float = np.pi/4, lookahead: int = None
    ):
        """
        Reduce trajectory probabilities by a factor of penalty for trajectories that come close to the current position.
        NOTE: this is not for se3_lpvds, which does not track the reference trajectory
        
        Args:
            state: Current state (x, y, z, qx, qy, qz, qw)
            mode: "trajectory", "cur_point" or "ref_point"
            penalty: Factor to reduce probability for trajectories with close points
            traj_threshold: Threshold distance to consider a trajectory as "close"
            radius: Threshold distance to consider a trajectory point as "close"
            angle_threshold: Threshold angle to consider a trajectory point as "close"
            lookahead: Steps to look ahead when selecting reference point (updates self.lookahead if provided)
        """
        if lookahead is not None:
            self.lookahead = lookahead
        if mode == "cur_point":
            for i in range(len(self.x)):
                distances = np.sqrt(np.sum((self.x[i] - state[:3]) ** 2, axis=1))
                angles = 2 * np.arccos(np.abs(np.sum(self.quat[i] * state[3:], axis=1)))
                mask = np.logical_and(distances < radius, angles < angle_threshold)
                if np.any(mask):
                    self.demo_traj_probs[i] *= penalty
        elif mode == "ref_point":
            for i in range(len(self.x)):
                cur_ref_x = self.x[self.ref_traj_idx][self.ref_point_idx_lookahead]
                cur_ref_quat = self.quat[self.ref_traj_idx][self.ref_point_idx_lookahead]
                distances = np.sqrt(np.sum((self.x[i] - cur_ref_x) ** 2, axis=1))
                angles = 2 * np.arccos(np.abs(np.sum(self.quat[i] * cur_ref_quat, axis=1)))
                mask = np.logical_and(distances < radius, angles < angle_threshold)
                if np.any(mask):
                    self.demo_traj_probs[i] *= penalty
        elif mode == "trajectory":
            # Get the currently followed trajectory
            current_traj_idx = self.ref_traj_idx
            if current_traj_idx is None:
                return

            current_traj = self.x[current_traj_idx]

            # Compute Hausdorff distances between current trajectory and all other trajectories
            for i in range(len(self.x)):
                if i == current_traj_idx:
                    continue  # Skip the current trajectory

                # Compute the pairwise distances between points in current_traj and other trajectory
                # For each point in current_traj, find its distance to each point in other trajectory
                # Shape: (len(current_traj), len(other_traj))
                other_traj = self.x[i]

                # Reshape for broadcasting: (n_points_A, 1, 3) - (1, n_points_B, 3)
                pairwise_distances = np.sqrt(np.sum(
                    (current_traj[:, np.newaxis, :] - other_traj[np.newaxis, :, :]) ** 2, 
                    axis=2
                ))

                # Directed Hausdorff h(A,B): max_{a in A} min_{b in B} d(a,b)
                h_AB = np.max(np.min(pairwise_distances, axis=1))

                # Directed Hausdorff h(B,A): max_{b in B} min_{a in A} d(b,a)
                h_BA = np.max(np.min(pairwise_distances, axis=0))

                # Hausdorff distance is the maximum of h(A,B) and h(B,A)
                hausdorff_dist = max(h_AB, h_BA)

                # If the Hausdorff distance is below the threshold, reduce the probability
                if hausdorff_dist < traj_threshold:  # radius parameter is used as threshold
                    self.demo_traj_probs[i] *= penalty
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # NOTE: this always penalizes the currently followed trajectory, forces it to at least switch to some other close trajectory
        self.demo_traj_probs[self.ref_traj_idx] *= penalty

        # Normalize the probabilities to keep the highest probability at 1
        if len(self.demo_traj_probs) > 0:
            max_prob = np.max(self.demo_traj_probs)
            if max_prob > 0:  # Avoid division by zero
                self.demo_traj_probs = self.demo_traj_probs / max_prob

        self.ref_traj_idx, self.ref_point_idx_lookahead, self.ref_point_idx_no_lookahead = self._choose_ref(
            state[: 3], True # NOTE: this is necessary! Forces resampling even self.switch is False. _choose_ref does not modify self.switch
        )

    def init_demo_traj_scores(self, x_state):
        self._choose_ref(x_state, True)

    def reset(self):
        self.demo_traj_probs = np.ones(len(self.x))
        self.ref_traj_idx = None
        self.ref_point_idx_lookahead = None

    def _load_node(self, model_path: str, input_dim: int, output_dim: int) -> NeuralODE:
        """
        Load a pre-trained Neural ODE model from disk.
        
        Args:
            model_path: Path to the saved model file
            input_dim: Dimension of the input to the model
            
        Returns:
            loaded_model: Loaded Neural ODE model
        """
        model_name = os.path.basename(model_path)
        width_str = model_name.split("width")[1].split("_")[0]
        depth_str = model_name.split("depth")[1].split("_")[0]
        width_size = int(width_str)
        depth = int(depth_str)

        model = NeuralODE(input_dim, output_dim, width_size, depth)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to('cpu')
        model.eval()

        return model

    def _train_node(
        self,
        input_data: list[np.ndarray],
        output_data: list[np.ndarray],
        save_path: str,
        device: str = "cpu",
        batch_size: int = 1,
        lr_strategy: tuple = (1e-3, 1e-4, 1e-5),
        epoch_strategy: tuple = (100, 100, 100),
        length_strategy: tuple = (0.4, 0.7, 1),
        plot: bool = True,
        print_every: int = 100,
    ) -> NeuralODE:
        """
        Train a Neural ODE model on demonstration data.
        
        Args:
            input_data: List of input trajectories for training
            output_data: List of output trajectories for training
            save_path: Path to save the trained model
            device: Device to run the training on
            batch_size: Number of trajectories to process in each batch
            lr_strategy: Learning rates for each training phase
            epoch_strategy: Number of epochs for each training phase
            length_strategy: Fraction of trajectories to use in each phase
            plot: Whether to plot training progress
            print_every: Print training metrics every N steps
            
        Returns:
            trained_model: Trained Neural ODE model
        """
        width_str = save_path.split("width")[1].split("_")[0]
        depth_str = save_path.split("depth")[1].split("_")[0]
        width_size = int(width_str)
        depth = int(depth_str)

        model = train(
            input_data,
            output_data,
            save_path,
            device=device,
            batch_size=batch_size,
            lr_strategy=lr_strategy,
            epoch_strategy=epoch_strategy,
            length_strategy=length_strategy,
            width_size=width_size,
            depth=depth,
            plot=plot,
            print_every=print_every,
        )
        model.to('cpu')
        model.eval()

        return model

    def _get_action_raw(self, state: np.ndarray):
        """
        Generate a raw control action for the given state without CLF constraints.
        
        Uses one of three approaches based on configuration:
        1. Velocity averaging for position + SO3-LPVDS for orientation
        2. Neural ODE for position + SO3-LPVDS for orientation
        3. Unified Neural ODE for both position and orientation
        
        Args:
            state: Current state vector [position (3), quaternion (4)]
            
        Returns:
            action: Raw control action [linear velocity (3), angular velocity (3)]
        """
        if self.model is not None:
            with torch.no_grad():
                return self.model.forward(state).numpy()

        if self.pos_none:
            action_pos = np.zeros(3)
        elif self.pos_avg:
            # Find all points from self.x whose distance is close to current state
            x_curr = state[:3]
            close_points_velocities = []

            # Define a distance threshold for "close" points
            distance_threshold = 0.1  # This can be adjusted as needed

            # Iterate through each trajectory
            for traj_idx, traj in enumerate(self.x):
                # Calculate distances from current state to each point in trajectory
                distances = np.sqrt(np.sum((traj - x_curr) ** 2, axis=1))

                # Find indices of close points
                close_indices = np.where(distances < distance_threshold)[0]

                # Add corresponding velocities to our collection
                for idx in close_indices:
                    close_points_velocities.append(self.x_dot[traj_idx][idx])

            # If no close points found, use the model directly
            if len(close_points_velocities) == 0:
                action_pos = np.zeros(3)
            else:
                # Average the velocities of close points
                action_pos = np.mean(np.array(close_points_velocities), axis=0)
        elif self.pos_model is not None:
            with torch.no_grad():
                action_pos = self.pos_model.forward(state[:3]).numpy()
        else:
            raise ValueError("Cannot get position action")

        if self.quat_none:
            action_ang = np.zeros(3)
        elif self.quat_simple: # NOTE: this implements a PD controller for the orientation error
            cur_quat = R.from_quat(state[3:])
            ref_quat = R.from_quat(self.quat[self.ref_traj_idx][self.ref_point_idx_lookahead])
            R_err = ref_quat * cur_quat.inv()
            action_ang = R_err.as_rotvec()
            # R_err = cur_quat.inv() * ref_quat
            # # Extract quaternion [x, y, z, w] from error rotation
            # e_quat = R_err.as_quat()
            # e_vec = e_quat[:3]  # vector part
            # e0 = e_quat[3]   # scalar part
            # # Sign of scalar part (to ensure shortest rotation direction)
            # sign_e0 = 1.0 if e0 >= 0.0 else -1.0
            # # Proportional term (factor of 2 often used from quaternion kinematics)
            # action_ang = 2.0 * sign_e0 * e_vec
        elif self.quat_model is not None:
            with torch.no_grad():
                action_ang = self.quat_model._step(R.from_quat(state[3:]), self.dt)[2]
        else:
            raise ValueError("Cannot get orientation action")

        return np.concatenate([action_pos, action_ang])

    def _compute_vel(self, x: np.ndarray):
        """
        Compute velocities from trajectory positions using finite differences.
        
        Args:
            x: Trajectory positions
            
        Returns:
            x_dot: Computed velocities
        """
        x_dot = np.diff(x, axis=0) / self.dt
        x_dot = np.vstack([x_dot, x_dot[-1]])

        return x_dot

    def _train_so3_lpvds(self, save_path: str, k_init: int = 10):
        """
        Train an SO3-LPVDS model for quaternion orientation control.
        
        Args:
            save_path: Path to save the trained model
            k_init: Initial number of Gaussian components for the model
        """
        q_rotations = [[R.from_quat(quat) for quat in quat_traj] for quat_traj in self.quat]
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

    def _load_so3_lpvds(self, model_path: str):
        """
        Load a pre-trained SO3-LPVDS model for quaternion orientation control.
        
        Args:
            model_path: Path to the saved model file
        """
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

    def _quat_data_preprocess(self, trajs: list[np.ndarray]) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Preprocess quaternion trajectory data for SO3-LPVDS training.
        
        Converts raw demonstration data into the format required by the SO3-LPVDS model.
        
        Args:
            trajs: List of trajectories containing position and quaternion data
            
        Returns:
            Tuple containing processed position and orientation data:
            (p_in, q_in, p_out, q_out, p_init, q_init, p_att, q_att)
        """
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

    def _apply_clf(
        self,
        current_state: np.ndarray,
        action_raw: np.ndarray,
        alpha_V: float,
    ) -> np.ndarray:
        """
        Apply Control Lyapunov Function (CLF) constraints to ensure stability.
        
        Uses quadratic programming to find the minimal adjustment to the model's
        output that satisfies stability constraints.
        
        Args:
            current_state: Current state vector [position (3), quaternion (4)]
            action_from_model: Raw action from the model
            alpha_V: CLF convergence rate parameter
            
        Returns:
            action: Modified action that satisfies CLF constraints
        """
        ref_x = self.x[self.ref_traj_idx][self.ref_point_idx_lookahead]
        ref_quat = self.quat[self.ref_traj_idx][self.ref_point_idx_lookahead]
        ref_state = np.concatenate([ref_x, ref_quat])

        if (
            self.quat_model is not None # quat_model already applied clf, only need to apply clf to position
            or self.quat_simple # quat_simple does not need clf
        ):  
            current_x = current_state[: 3]
            f_x = action_raw[: 3]
            if self.ref_point_idx_no_lookahead == self.x[self.ref_traj_idx].shape[0] - 1: # TODO: this is different from the original implementation
                f_xref = 0
            elif self.pos_none or self.pos_avg:
                f_xref = self.x_dot[self.ref_traj_idx][self.ref_point_idx_lookahead]
            else:
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

            action = action_raw
            action[:3] = f_x[:3] + u_star

            return action

        else:  # apply clf to both NOTE: doesn't work
            # Get reference dynamics
            if self.ref_point_idx_no_lookahead == self.x[self.ref_traj_idx].shape[0] - 1: # TODO: this is different from the original implementation
                vel_pos_ref = np.zeros(3)
                vel_ang_ref = np.zeros(3)
            elif self.model is not None:
                with torch.no_grad():
                    vel_ref = jnp.array(self.model.forward(ref_state))
                    vel_pos_ref = vel_ref[:3]
                    vel_ang_ref = vel_ref[3:]
            else:
                if self.pos_none or self.pos_avg:
                    vel_pos_ref = self.x_dot[self.ref_traj_idx][self.ref_point_idx_lookahead]
                else:
                    with torch.no_grad():
                        vel_pos_ref = jnp.array(self.pos_model.forward(ref_x))
                if self.quat_none:
                    vel_ang_ref = jnp.zeros(3)
                else:
                    with torch.no_grad():
                        vel_ang_ref = jnp.array(self.quat_model._step(ref_quat, self.dt)[2])
            action_ref_pos = vel_pos_ref.reshape((-1,1))
            action_ref_euler = vel_ang_ref.reshape((-1,1))
            action_ref_quat = jnp.vstack((0.0, action_ref_euler))
            action_ref = jnp.vstack((action_ref_pos, action_ref_quat))

            cur_state = jnp.array(current_state).reshape((-1,1))
            cur_quat = cur_state[3:]

            # Get the reference state at the chosen index
            ref_state = jnp.array(ref_state).reshape((-1,1)) # dim x 1

            # Extract reference quaternion
            ref_quat = ref_state[3:]

            # Scaling factors for position and quaternion errors in the cost function
            scale_q = 1
            scale_p = 1

            # Construct the cost matrix for the QP
            Q_opt = jnp.vstack((jnp.hstack((scale_p*jnp.eye(3), jnp.zeros((3,3)))), jnp.hstack((jnp.zeros((3,3)), scale_q*jnp.eye(3)))))

            # Compute error between current and reference state
            err = cur_state - ref_state
            # Normalize quaternion error
            err = err.at[3:].set(err[3:]/jnp.linalg.norm(err[3:]))
            err_pos = cur_state[:3] - ref_state[:3]  # Position error
            err_quat = (R.from_quat(ref_quat) * R.from_quat(cur_quat).inv()).as_quat() # Quaternion error

            # Lyapunov function (quadratic in the error)
            V = jnp.sum(jnp.square(2*err))/4

            # Get nominal dynamics from the learned model
            action_raw = action_raw.reshape((-1,1))
            action_raw_pos = action_raw[:3]  # Position dynamics

            # Angular velocity from the model
            action_raw_euler = action_raw[3:].reshape((-1,1))
            action_raw_quat = jnp.vstack((0.0, action_raw_euler))  # Prepend 0 for quaternion multiplication

            # Compute the CLF derivative terms
            s_p = jnp.vdot(err_pos, action_raw_pos - action_ref_pos)  # Position term
            s_q = jnp.vdot(err_quat/2.0, quat_mult(action_raw_quat[:, 0], cur_quat[:, 0]).reshape((-1,1)) - quat_mult(action_ref_quat[:, 0], ref_quat[:, 0]).reshape((-1,1)))  # Quaternion term

            s = s_p + s_q  # Combined CLF derivative

            # Construct quaternion matrix for quaternion dynamics
            Q = jnp.array([[cur_quat[0, 0], cur_quat[1, 0], cur_quat[2, 0], cur_quat[3, 0]],
                            [-cur_quat[1, 0], cur_quat[0,0], cur_quat[3, 0], -cur_quat[2, 0]],
                            [-cur_quat[2, 0], -cur_quat[3, 0], cur_quat[0, 0], cur_quat[1, 0]],
                            [-cur_quat[3, 0], cur_quat[2, 0], -cur_quat[1, 0], cur_quat[0, 0]]])
            Q2 = Q[:, 1:] # 4 x 3
            Q2_minus = -Q2
            Q2_1 = jnp.vstack((Q2_minus[0, :], -Q2_minus[1:,:]))

            # Construct the CLF constraint for the QP
            G_opt = 2*jnp.hstack((err_pos.T, (err_quat.T/2.0) @ Q2_1))
            h_opt = jnp.array([-alpha_V*V - 2*s])

            # qp = OSQP()
            # sol = qp.run(params_obj=(Q_opt,jnp.zeros(Q_opt.shape[0])), params_ineq=(G_opt, h_opt)).params
            # ustar = sol.primal.reshape((-1,1))

            ustar = qp_solve(Q_opt, G_opt, h_opt).primal.reshape((-1,1))

            # # Extract position and angular velocity control inputs
            # u_p_x = ustar[:3]
            # u_w_x = jnp.vstack((0.0, ustar[3:]))

            # # Compute state derivatives with the control inputs
            # action_pos = action_raw_pos + u_p_x  # Position derivative
            # action_euler = action_raw_euler + ustar[3:]
            return np.array(action_raw + ustar).flatten()

    def _choose_ref(self, x_state: np.ndarray, switch: bool = False) -> tuple[int, int, int]:
        """
        Choose reference trajectory and point based on current position.
        
        Criteria:
            - for each demo trajectory i, compute smallest distance to x_state among all its points: closest_distances[i], corresponding point index: closest_indices[i]
            - score[i] = log(p[i]) - closest_distances[i] NOTE: this could be modified
            - prob_dist = softmax(score)
            - traj_idx = argmax(prob_dist), point_idx = closest_indices[traj_idx]
        
        Args:
            x_state: Current position state (x, y, z)
            switch: If True, allows switching between trajectories NOTE: this doesn't modify self.switch
            
        Returns:
            ref_traj_idx: Index of the selected reference trajectory
            ref_point_idx_lookahead: Index of the selected point within the trajectory with lookahead
            ref_point_idx_no_lookahead: Index of the selected point within the trajectory without lookahead
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

            scores = np.log(self.demo_traj_probs) - np.array(closest_distances)
            self.demo_traj_scores = np.exp(scores) / np.sum(np.exp(scores))

            # Select the trajectory with the highest probability
            # If multiple trajectories have the same maximum probability, choose one randomly
            max_prob = np.max(self.demo_traj_scores)
            max_indices = np.where(self.demo_traj_scores == max_prob)[0]
            if len(max_indices) > 1:
                # Multiple trajectories have the same maximum probability, choose randomly
                traj_idx = np.random.choice(max_indices)
            else:
                traj_idx = max_indices[0]

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

        point_idx_no_lookahead = point_idx
        point_idx_lookahead = min(point_idx + self.lookahead, self.x[traj_idx].shape[0] - 1)

        return int(traj_idx), int(point_idx_lookahead), int(point_idx_no_lookahead)

    def _configure_models(
        self, 
        pos_config: Optional[PositionModelConfig], 
        quat_config: Optional[QuaternionModelConfig], 
        unified_config: Optional[UnifiedModelConfig]
    ):
        """
        Configure models based on the provided configuration objects.
        
        Args:
            pos_config: Configuration for position model
            quat_config: Configuration for quaternion model
            unified_config: Configuration for unified model
        """
        if unified_config is not None:
            self._configure_unified_model(unified_config)
            return

        if pos_config is not None:
            self._configure_pos_model(pos_config)

        if quat_config is not None:
            self._configure_quat_model(quat_config)

    def _configure_pos_model(self, config: PositionModelConfig):
        """
        Configure the position model based on its configuration.
        
        Args:
            config: Position model configuration
        """
        if config.mode == "none":
            self.pos_none = True
        elif config.mode == "avg":
            self.pos_avg = True
        elif config.mode == "neural_ode":
            if config.load_path is not None:
                self.pos_model = self._load_node(config.load_path, 3, 3)
            else:
                self.pos_model = self._train_node(
                    self.x,
                    self.x_dot,
                    config.save_path,
                    device=config.device,
                    batch_size=config.batch_size,
                    lr_strategy=config.lr_strategy,
                    epoch_strategy=config.epoch_strategy,
                    length_strategy=config.length_strategy,
                    plot=config.plot,
                    print_every=config.print_every
                )

    def _configure_quat_model(self, config: QuaternionModelConfig):
        """
        Configure the quaternion model based on its configuration.
        
        Args:
            config: Quaternion model configuration
        """
        if config.mode == "none":
            self.quat_none = True
        elif config.mode == "simple":
            self.quat_simple = True
        elif config.mode == "so3_lpvds":
            if config.load_path is not None:
                self._load_so3_lpvds(config.load_path)
            else:
                self._train_so3_lpvds(
                    save_path=config.save_path,
                    k_init=config.k_init
                )

    def _configure_unified_model(self, config: UnifiedModelConfig):
        """
        Configure the unified model based on its configuration.
        
        Args:
            config: Unified model configuration
        """
        if config.mode == "neural_ode":
            if config.load_path is not None:
                self.model = self._load_node(config.load_path, 7, 6)
            else:
                # Create combined input and output data for unified model
                input_data = [
                    np.concatenate([self.x[i], self.quat[i]], axis=-1)
                    for i in range(len(self.x))
                ]
                output_data = [
                    np.concatenate([self.x_dot[i], self.omega[i]], axis=-1)
                    for i in range(len(self.x_dot))
                ]

                self.model = self._train_node(
                    input_data,
                    output_data,
                    config.save_path,
                    device=config.device,
                    batch_size=config.batch_size,
                    lr_strategy=config.lr_strategy,
                    epoch_strategy=config.epoch_strategy,
                    length_strategy=config.length_strategy,
                    plot=config.plot,
                    print_every=config.print_every
                )
        elif config.mode == "se3_lpvds":
            self.se3_lpvds = True
            p_raw = self.x # list of np.ndarray
            q_raw = [[R.from_quat(quat) for quat in quat_traj] for quat_traj in self.quat] # list of list of Rotation

            end_pts = [(p_traj[-1], q_traj[-1]) for p_traj, q_traj in zip(p_raw, q_raw)]
            self.se3_lpvds_attractor_generator = self.SE3LVPDSAttractorGenerator(end_pts, mode=config.attractor_resampling_mode)
            # p_att, q_att = self.se3_lpvds_attractor_generator.sample()
            # p_in, q_in = self._shift_trajs(p_att, q_att) # same length as self.x, self.quat, but shifted by attractor
            # p_in = process_tools._smooth_pos(p_in)
            t_raw = [np.linspace(0, len(p_traj) * self.dt, len(p_traj)) for p_traj in p_raw]
            p_in, q_in, t_raw, p_att, q_att = process_tools.pre_process(p_raw, q_raw, t_raw, shift=False, opt="savgol")

            if self.relative_cluster_attractor is not None:
                p_att = self.relative_cluster_attractor[0:3]
                q_att = R.from_quat(self.relative_cluster_attractor[3:])

            self.pos_att: np.ndarray = p_att
            self.r_att: R = q_att
            self.pos_shift = np.zeros(3)
            self.r_shift = R.identity()

            # Truncate last part of the trajectories to avoid unstable dynamics
            # truncate_percent = 0.05
            # for i in range(len(p_in)):
            #     keep_points = int((1 - truncate_percent) * len(p_in[i]))
            #     p_in[i] = p_in[i][:keep_points]
            #     q_in[i] = q_in[i][:keep_points]

            K_candidates = config.K_candidates
            self.train_se3_lpvds(p_in, q_in, p_att, q_att, self.dt, K_candidates, visualize=False)

            # Store PD parameters from config
            self.enable_simple_ds_near_target = config.enable_simple_ds_near_target
            if self.enable_simple_ds_near_target:  
                self.simple_ds_pos_threshold = config.simple_ds_pos_threshold
                self.simple_ds_ori_threshold = config.simple_ds_ori_threshold
                self.K_pos = config.K_pos
                self.K_ori = config.K_ori

            self.modulations: list[tuple[np.ndarray, float]] = [] # (position, radius)

    def train_se3_lpvds(self, p_in, q_in, p_att, q_att, dt, K_candidates, visualize=False):
        t_in = [[j*dt for j in range(len(p_in[i]))] for i in range(len(p_in))] # list of list of float
        p_out, q_out = process_tools.compute_output(p_in, q_in, t_in)
        p_in_roll, q_in_roll, p_out_roll, q_out_roll = process_tools.rollout_list(p_in, q_in, p_out, q_out) # Use rolled versions
        smallest_reconstruction_error = float('inf')
        # self.model = se3_class(p_in_roll, q_in_roll, p_out_roll, q_out_roll, p_att, q_att, dt, K_candidates[0])
        # self.model.begin()
        for i, K in enumerate(K_candidates):
            model = se3_class(p_in_roll, q_in_roll, p_out_roll, q_out_roll, p_att, q_att, dt, K)  
            try:
                model.begin()
            except Exception as e:
                print(f"Omitting K={K} due to error: {e}")
                continue
            reconstruction_error = model.compute_reconstruction_error() # TODO
            if reconstruction_error < smallest_reconstruction_error:
                smallest_reconstruction_error = reconstruction_error
                self.model = model
        print(f"Best K: {K_candidates[i]}")
        if visualize:
            plot_tools.plot_gmm(p_in_roll, self.model.gmm)

    def _validate_model_setup(self):
        """
        Validate that the models are properly configured.
        
        Raises:
            ValueError: If models are not properly configured
        """
        if self.model is not None:
            # Unified model is configured, no need for separate models
            return

        if not self.pos_none and not self.pos_avg and self.pos_model is None:
            raise ValueError("Either set pos_config.mode to 'none' or 'avg' or configure a position model")

        if not self.quat_none and not self.quat_simple and self.quat_model is None:
            raise ValueError("Either set quat_config.mode to 'none' or 'simple' or configure a quaternion model")

    def plot_position_vector_field(
        self,
        save_path: str = None,
    ):
        """
        Visualize the vector field of the trained model along with training data.
        """
        pos_to_vel = lambda x: self.get_action(x)[:3]
        demo_trajs_flat = np.concatenate(self.x, axis=0)

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
                    vel = pos_to_vel(np.concatenate([pos, quat]))
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

        for traj in self.x:
            ax.plot3D(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                "b-",
                linewidth=1,
            )

        if self.se3_lpvds:
            ax.scatter(self.model.p_att[0], self.model.p_att[1], self.model.p_att[2], "g-", linewidth=1)

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

    def compute_reconstruction_error(self) -> tuple[float, float]:
        """
        Compute the reconstruction error of the trained model, using the training data.

        Returns:
            error: float, the total reconstruction error
            avg_error: float, the average reconstruction error
        """
        error = 0
        total_pts = 0
        for i in range(len(self.x)):
            total_pts += len(self.x[i])
            for j in range(len(self.x[i])):
                action = self.get_action(np.concatenate([self.x[i][j], self.quat[i][j]]), clf=True, alpha_V=20.0, lookahead=5)
                x_dot_pred = action[:3]
                omega_pred = action[3:6]
                error += np.linalg.norm(self.x_dot[i][j] - x_dot_pred) + np.linalg.norm(self.omega[i][j] - omega_pred)
        return error, error / total_pts

    class SE3LVPDSAttractorGenerator:
        def __init__(self, end_pts: list[tuple[np.ndarray, R]], mode: str):
            """
            Args:
                end_pts: List of tuples (np.ndarray, Rotation)
                mode: Mode of the attractor generator
            """
            self.end_pts = end_pts
            self.mode = mode
            if mode == "Gaussian":
                if len(self.end_pts) == 1:
                    logging.warning("Only one end point provided. Gaussian not fit.")
                else:
                    R_mean = R.from_quat([r.as_quat() for _, r in self.end_pts]).mean()
                    self.gaussian = gmm_class(np.array([p for p, _ in self.end_pts]), [r for _, r in self.end_pts], R_mean, K_init=1)
                    self.gaussian.fit()
                

        def sample(self, state: Optional[np.ndarray] = None) -> tuple[np.ndarray, R]:
            if self.mode == "random":
                random_idx = np.random.randint(0, len(self.end_pts))
                return self.end_pts[random_idx]
            elif self.mode == "nearest":
                pass
            elif self.mode == "Gaussian":
                p, q = self.gaussian.sample()
                return (p, R.from_quat(q))
            else:
                raise ValueError(f"Invalid mode: {self.mode}")


def gamma_spherical(point: np.ndarray, object_center: np.ndarray, radius: float) -> tuple[float, np.ndarray]:
    """
    Calculates the gamma function and its gradient for a spherical obstacle in 3D.

    Args:
        point: A 3-element numpy array representing the point [x, y, z].
        object_center: A 3-element numpy array representing the center [cx, cy, cz] of the sphere.
        radius: The radius of the sphere.

    Returns:
        A tuple containing:
            - gamma (float): The value of the gamma function Γ(point) = ||point - center||² - radius² + 1.
            - gradient_gamma (np.ndarray): A 3-element numpy array representing the gradient ∇Γ(point) = [2dx, 2dy, 2dz].
    """
    # Ensure inputs are numpy arrays
    point = np.asarray(point)
    object_center = np.asarray(object_center)

    # Compute displacement vector from obstacle center
    displacement = point - object_center
    dx, dy, dz = displacement[0], displacement[1], displacement[2]

    # Squared distance from point to obstacle center
    dist_sq = dx**2 + dy**2 + dz**2

    # Gamma function: Γ(point) = ||point - center||² - radius² + 1
    gamma = dist_sq - radius**2 + 1

    # Gradient of Γ(point): ∇Γ(point) = 2 * displacement = [2dx, 2dy, 2dz]
    gradient_gamma = 2 * displacement

    return gamma, gradient_gamma

def normal_modulation(gamma: float, gradient_gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the modulation matrix for obstacle avoidance in 3D using the normal vector.

    Args:
        gamma: Scalar value of the boundary function Γ(x).
        gradient_gamma: 3D vector representing the gradient ∇Γ(x).

    Returns:
        A tuple containing:
            - D (np.ndarray): 3x3 diagonal matrix of eigenvalues.
            - E (np.ndarray): 3x3 matrix representing the eigenbasis.
            - M (np.ndarray): 3x3 modulation matrix.
    """
    # Ensure gradient is a numpy array
    gradient_gamma = np.asarray(gradient_gamma)

    # Initialize outputs to identity matrices (fallback for zero gradient)
    D = np.eye(3)
    E = np.eye(3)
    M = np.eye(3)

    # Calculate the norm of the gradient
    grad_norm = np.linalg.norm(gradient_gamma)

    # Avoid division by zero if gradient is zero
    if grad_norm < 1e-8:
        # If gradient is zero, no modulation is applied (return identity matrices)
        # This might happen far from the obstacle or exactly at the center
        return D, E, M

    # Normalize the gradient to get the normal vector 'n'
    n = gradient_gamma / grad_norm

    # Construct the eigenbasis E = [n, e1, e2]
    # Find a vector 't' not parallel to 'n'
    if abs(n[0]) < 0.9:  # If n is not aligned with x-axis
        t = np.array([1.0, 0.0, 0.0])
    else: # If n is aligned or nearly aligned with x-axis, use y-axis
        t = np.array([0.0, 1.0, 0.0])

    # Create the first tangent vector e1 using cross product and normalize
    e1 = np.cross(n, t)
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-8: # Should not happen if t is chosen correctly
        # Fallback if cross product is zero (n and t were parallel)
        # This case is unlikely with the above check for t
        # We can try another axis like [0, 0, 1] or just return identity
        t = np.array([0.0, 0.0, 1.0])
        e1 = np.cross(n, t)
        e1_norm = np.linalg.norm(e1)
        if e1_norm < 1e-8: return D, E, M # Final fallback
    e1 = e1 / e1_norm

    # Create the second tangent vector e2 using cross product (already normalized)
    e2 = np.cross(n, e1)
    # Optional check: np.linalg.norm(e2) should be close to 1

    # Eigenbasis matrix E (columns are n, e1, e2)
    E = np.column_stack((n, e1, e2))

    # Calculate eigenvalues
    # Avoid division by zero or very large values if gamma is close to zero
    if abs(gamma) < 1e-8:
        # Handle case where gamma is very small (deep inside obstacle)
        # Option 1: Set large modulation (e.g., project velocity onto tangent plane)
        lambda_n = -1000.0 # Large negative value to strongly repel
        lambda_e = 1.0     # Allow tangential motion
        # Option 2: Return identity (less aggressive)
        # return np.eye(3), np.eye(3), np.eye(3)
    else:
        lambda_n = 1.0 - 1.0 / gamma  # Eigenvalue for normal direction
        lambda_e = 1.0 + 1.0 / gamma  # Eigenvalue for tangential directions

    # Eigenvalue matrix D
    D = np.diag([lambda_n, lambda_e, lambda_e])

    # Modulation matrix M = E * D * E^T
    # Since E is orthogonal (orthonormal columns), E^T = E^-1
    M = E @ D @ E.T

    return D, E, M

def spherical_normal_modulation(points: np.ndarray, object_center: np.ndarray, radius: float, v: np.ndarray) -> np.ndarray:
    gamma, gradient_gamma = gamma_spherical(points, object_center, radius)
    D, E, M = normal_modulation(gamma, gradient_gamma)
    # Apply modulation to velocity
    if gamma >= 1:
        v_modulated = M @ v
    else:
        repulsion_gain = 1.0
        repulsive_dir = gradient_gamma / np.linalg.norm(gradient_gamma)
        xd_rep = repulsion_gain * repulsive_dir
        v_modulated = xd_rep
    return v_modulated