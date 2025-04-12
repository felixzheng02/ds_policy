import os

from .neural_ode.neural_ode import NeuralODE
import numpy as np
import torch
from jaxopt import OSQP
import jax
import jax.numpy as jnp
import json
from scipy.spatial.transform import Rotation as R

from .neural_ode.train_neural_ode import train
from .ds_utils import quat_mult, Quaternion, quat_to_euler, euler_to_quat
from .so3_lpvds.gmm_class import gmm_class
from .so3_lpvds.quat_class import quat_class
from .so3_lpvds.process_tools import (
    pre_process,
    compute_output,
    extract_state,
    rollout_list,
)


qp = OSQP()


@jax.jit
def qp_solve(Q_opt, G_opt, h_opt):
    return qp.run(params_obj=(Q_opt, jnp.zeros(Q_opt.shape[0])), params_ineq=(G_opt, h_opt)).params


class DSPolicy:
    """
    Public methods:
        __init__: Initialize the DS policy with demonstration data and model configuration
        get_action: Generate control action for a given state
        update_demo_traj_probs: Update trajectory probabilities based on collision detection
        reset: Reset the trajectory probabilities to the initial values
    Private methods (should not be called externally):
        _load_node: Load a pre-trained Neural ODE model from disk
        _train_node: Train a Neural ODE model on demonstration data
        _get_action_raw: Generate a raw control action for the given state without CLF constraints
        _compute_vel: Compute velocities from trajectory positions using finite differences
        _train_so3_lpvds: Train an SO3-LPVDS model for quaternion orientation control
        _load_so3_lpvds: Load a pre-trained SO3-LPVDS model for quaternion orientation control
        _quat_data_preprocess: Preprocess the quaternion data for the SO3-LPVDS model
        _apply_clf: Apply the Control Lyapunov Function (CLF) constraints to the control action
        _choose_ref: Choose the reference trajectory and point based on the current position
        _configure_models: Configure the models based on the provided configuration dictionary
        _validate_model_setup: Validate that the models are properly configured
    """

    def __init__(
        self,
        x: list[np.ndarray],
        x_dot: list[np.ndarray],
        quat: list[np.ndarray],
        omega: list[np.ndarray],
        gripper: list[np.ndarray],
        model_config: dict,
        dt: float = 0.01,
        switch: bool = False,
        lookahead: int = 5,
        demo_traj_probs: np.ndarray = None,
        backtrack_steps: int = 0,
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
            model_config: Configuration dictionary for models with structure:
                - unified_model (dict, optional): Configuration for unified position+orientation model
                    - load_path (str, optional): Path to load pre-trained model
                    - [training parameters if not loading]
                - pos_model (dict, optional): Configuration for position-only model
                    - special_mode (str, optional): "none", "avg"
                    - load_path (str, optional): Path to load pre-trained model
                    - [training parameters if not loading]
                - quat_model (dict, optional): Configuration for orientation-only model
                    - special_mode (str, optional): "none"
                    - load_path (str, optional): Path to load pre-trained model
                    - [training parameters if not loading]
            dt: Time step for simulation and prediction
            switch: If True, allows switching between trajectories at runtime
            lookahead: Number of steps to lookahead for reference trajectory selection
            demo_traj_probs: Prior probabilities for each demonstration trajectory
            backtrack_steps: Number of steps to backtrack on collision
            **kwargs: Additional keyword arguments
        """
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

        self.pos_none = False
        self.pos_avg = False
        self.quat_none = False
        self.quat_simple = False
        self._configure_models(model_config)
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
    
    def update_demo_traj_probs(
        self, state: np.ndarray, mode: str, penalty: float, traj_threshold: float = 0.1, radius: float = 0.05, angle_threshold: float = np.pi/4, lookahead: int = None
    ):
        """
        Reduce trajectory probabilities by a factor of penalty for trajectories that come close to the current position.
        
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
            err_quat = quat_mult(ref_quat.flatten(), R.from_quat(cur_quat.flatten()).inv().as_quat()).reshape((-1,1)) # Quaternion error
            
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

    def _configure_models(self, model_config):
        """
        Configure models based on the provided configuration dictionary.
        
        Sets up the appropriate models for position and orientation control
        based on the configuration provided during initialization.
        
        Args:
            model_config: Dictionary containing model configurations
        """
        # Configure unified model if specified
        if 'unified_model' in model_config:
            unified_config = model_config['unified_model']
            
            # Check if loading pre-trained model
            if 'load_path' in unified_config:
                self.model = self._load_node(unified_config['load_path'], 7, 6)
            else:
                # Training configuration
                width = unified_config.get('width', 128)
                depth = unified_config.get('depth', 3)
                save_path = unified_config.get('save_path', f'./models/unified_model_width{width}_depth{depth}.pt')
                batch_size = unified_config.get('batch_size', 1)
                device = unified_config.get('device', 'cpu')
                lr_strategy = unified_config.get('lr_strategy', (1e-3, 1e-4, 1e-5))
                epoch_strategy = unified_config.get('epoch_strategy', (100, 100, 100))
                length_strategy = unified_config.get('length_strategy', (0.4, 0.7, 1))
                plot = unified_config.get('plot', True)
                print_every = unified_config.get('print_every', 100)
                
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
                    save_path,
                    device=device,
                    batch_size=batch_size,
                    lr_strategy=lr_strategy,
                    epoch_strategy=epoch_strategy,
                    length_strategy=length_strategy,
                    plot=plot,
                    print_every=print_every
                )
            return
        
        # Configure position model if specified
        if 'pos_model' in model_config:
            pos_config = model_config['pos_model']
            # Check for use_avg mode
            self.pos_none = pos_config.get('special_mode', None) == 'none'
            self.pos_avg = pos_config.get('special_mode', None) == 'avg'
            
            if not self.pos_none and not self.pos_avg:
                if 'load_path' in pos_config:
                    self.pos_model = self._load_node(pos_config['load_path'], 3, 3)
                else:
                    # Training configuration
                    width = pos_config.get('width', 128)
                    depth = pos_config.get('depth', 3)
                    save_path = pos_config.get('save_path', f'./models/pos_model_width{width}_depth{depth}.pt')
                    device = pos_config.get('device', 'cpu')
                    batch_size = pos_config.get('batch_size', 1)
                    lr_strategy = pos_config.get('lr_strategy', (1e-3, 1e-4, 1e-5))
                    epoch_strategy = pos_config.get('epoch_strategy', (100, 100, 100))
                    length_strategy = pos_config.get('length_strategy', (0.4, 0.7, 1))
                    plot = pos_config.get('plot', True)
                    print_every = pos_config.get('print_every', 100)
                    
                    self.pos_model = self._train_node(
                        self.x,
                        self.x_dot,
                        save_path,
                        device=device,
                        batch_size=batch_size,
                        lr_strategy=lr_strategy,
                        epoch_strategy=epoch_strategy,
                        length_strategy=length_strategy,
                        plot=plot,
                        print_every=print_every
                    )
        
        # Configure quaternion model if specified
        if 'quat_model' in model_config:
            quat_config = model_config['quat_model']

            self.quat_none = quat_config.get('special_mode', None) == 'none'
            if self.quat_none:
                return
            self.quat_simple = quat_config.get('special_mode', None) == 'simple'
            if self.quat_simple:
                return
            
            # Check if loading pre-trained model
            if 'load_path' in quat_config:
                self._load_so3_lpvds(quat_config['load_path'])
            else:
                # Training configuration
                save_path = quat_config.get('save_path', './models/quat_model.json')
                k_init = quat_config.get('k_init', 10)
                
                self._train_so3_lpvds(
                    save_path=save_path,
                    k_init=k_init
                )

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
            raise ValueError("Either set special_mode to 'none' or 'avg' or configure a position model")
        
        if not self.quat_none and not self.quat_simple and self.quat_model is None:
            raise ValueError("Either set special_mode to 'none' or configure a quaternion model")