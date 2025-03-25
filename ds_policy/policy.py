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
from .ds_utils import quat_mult, Quaternion
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
    return qp.run(params_obj=(Q_opt, jnp.zeros(3)), params_ineq=(G_opt, h_opt)).params


class DSPolicy:
    """
    Public methods:
        __init__: Initialize the DS policy with demonstration data and model configuration
        get_action: Generate control action for a given state
        update_demo_traj_probs: Update trajectory probabilities based on collision detection
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
            model_config: Configuration dictionary for models with structure:
                - unified_model (dict, optional): Configuration for unified position+orientation model
                    - load_path (str, optional): Path to load pre-trained model
                    - [training parameters if not loading]
                - pos_model (dict, optional): Configuration for position-only model
                    - special_mode (str, optional): "none", "avg"
                    - load_path (str, optional): Path to load pre-trained model
                    - [training parameters if not loading]
                - quat_model (dict, optional): Configuration for orientation-only model
                    - load_path (str, optional): Path to load pre-trained model
                    - [training parameters if not loading]
            dt: Time step for simulation and prediction
            switch: If True, allows switching between trajectories at runtime
            lookahead: Number of steps to lookahead for reference trajectory selection
            demo_traj_probs: Prior probabilities for each demonstration trajectory
            backtrack_steps: Number of steps to backtrack on collision
            **kwargs: Additional keyword arguments
        """
        if x[0].shape[-1] == 3: # transformed to goal frame
            self.node_input_size = 3
            self.node_output_size = 3
        elif x[0].shape[-1] == 6: # raw data from world_frame
            self.node_input_size = 6
            self.node_output_size = 3
        else:
            raise ValueError(f"Unexpected input size: {x[0].shape[-1]}")
        self.dt = dt
        self.x = x
        self.x_dot = x_dot
        self.quat = quat
        self.omega = omega
        self.switch = switch
        self.lookahead = lookahead
        self.backtrack_steps = backtrack_steps

        self.demo_traj_probs = np.ones(len(x)) if demo_traj_probs is None else demo_traj_probs
        self.ref_traj_idx = None
        self.ref_point_idx = None

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

        self.none = False
        self.avg = False
        self._configure_models(model_config)
        self._validate_model_setup()

    def get_action(
        self,
        state: np.ndarray,
        clf: bool = True,
        alpha_V: float = 20.0,
        lookahead: int = None,
    ) -> np.ndarray:
        """
        Generate a control action for the given state.
        
        Computes a velocity command based on the current state, using the configured models.
        Can optionally apply Control Lyapunov Function (CLF) constraints for stability.
        
        Args:
            state: Current state vector [position (3), quaternion (4)]
            clf: If True, apply Control Lyapunov Function constraints
            alpha_V: CLF convergence rate parameter
            lookahead: Steps to look ahead when selecting reference point (updates self.lookahead if provided)
            
        Returns:
            action: Control action [linear velocity (3), angular velocity (3)]
        """
        if lookahead is not None:
            self.lookahead = lookahead

        action_raw = self._get_action_raw(
            state
        )
        if clf:
            action = self._apply_clf(state, action_raw, alpha_V)
        else:
            action = action_raw
        return action
    
    def update_demo_traj_probs(
        self, x_state: np.ndarray, radius: float, penalty: float, lookahead: int = None
    ):
        """
        Reduce trajectory probabilities by a factor of penalty for trajectories that come close to the current position.
        
        Args:
            x_state: Current position state (x, y, z)
            radius: Threshold distance to consider a trajectory point as "close"
            penalty: Factor to reduce probability for trajectories with close points
            lookahead: Steps to look ahead when selecting reference point (updates self.lookahead if provided)
        """
        if lookahead is not None:
            self.lookahead = lookahead

        for i, traj in enumerate(self.x):
            distances = np.sqrt(np.sum((traj - x_state) ** 2, axis=1))
            if np.any(distances < radius):
                self.demo_traj_probs[i] *= penalty

        self.ref_traj_idx, self.ref_point_idx = self._choose_ref(x_state, True)

    def _load_node(self, model_path: str) -> NeuralODE:
        """
        Load a pre-trained Neural ODE model from disk.
        
        Args:
            model_path: Path to the saved model file
        Returns:
            loaded_model: Loaded Neural ODE model
        """
        model_name = os.path.basename(model_path)
        width_str = model_name.split("width")[1].split("_")[0]
        depth_str = model_name.split("depth")[1].split("_")[0]
        width_size = int(width_str)
        depth = int(depth_str)
        
        model = NeuralODE(self.node_input_size, self.node_output_size, width_size, depth)
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
        if self.none:
            q_curr = R.from_quat(state[3:])
            _, _, omega = self.quat_model._step(q_curr, self.dt)            
            return np.concatenate([np.zeros(3), omega])
        
        elif self.avg:
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
                x_dot = np.zeros(3)
            else:
                # Average the velocities of close points
                x_dot = np.mean(np.array(close_points_velocities), axis=0)
            
            # Get orientation part
            q_curr = R.from_quat(state[3:])
            _, _, omega = self.quat_model._step(q_curr, self.dt)            

            return np.concatenate([x_dot, omega])

        elif self.model is None:  # pos_model and quat_model are separate
            with torch.no_grad():
                x_dot = self.pos_model.forward(state[:3]).numpy()

            q_curr = R.from_quat(state[3:])

            _, _, omega = self.quat_model._step(q_curr, self.dt)

            return np.concatenate([x_dot, omega])
        else:
            with torch.no_grad():
                return self.model.forward(state).numpy()

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
        action_from_model: np.ndarray,
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
        self.ref_traj_idx, self.ref_point_idx = self._choose_ref(
            current_state[:3], self.switch
        )
        ref_x = self.x[self.ref_traj_idx][self.ref_point_idx]
        ref_quat = self.quat[self.ref_traj_idx][self.ref_point_idx]
        ref_state = np.concatenate([ref_x, ref_quat])

        if (
            self.model is None
        ):  # using so3-lpvds for orientation control, only need to apply clf to position
            current_x = current_state[:3]
            f_x = action_from_model[:3]
            if self.ref_point_idx == self.x[self.ref_traj_idx].shape[0] - 1: # TODO: this is different from the original implementation
                f_xref = 0
            elif self.none or self.avg:
                f_xref = self.x_dot[self.ref_traj_idx][self.ref_point_idx]
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

    def _choose_ref(self, x_state: np.ndarray, switch: bool = False) -> tuple[int, int]:
        """
        Choose reference trajectory and point based on current position.
        
        Selects the most appropriate demonstration trajectory and point within it
        to serve as a reference for the control system.
        
        Args:
            x_state: Current position state (x, y, z)
            switch: If True, allows switching between trajectories
            
        Returns:
            ref_traj_idx: Index of the selected reference trajectory
            ref_point_idx: Index of the selected point within the trajectory
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
                self.model = self._load_node(unified_config['load_path'])
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
            self.none = pos_config.get('special_mode', None) == 'none'
            self.avg = pos_config.get('special_mode', None) == 'avg'
            
            if not self.none and not self.avg:
                if 'load_path' in pos_config:
                    self.pos_model = self._load_node(pos_config['load_path'])
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
        if self.none or self.avg:
            # When using none or averaging, we still need quaternion model
            if self.quat_model is None:
                raise ValueError("Quaternion model is required when using velocity averaging mode")
            return
        
        if self.model is not None:
            # Unified model is configured, no need for separate models
            return
        
        if self.pos_model is None or self.quat_model is None:
            # Both position and quaternion models are required when not using unified model
            missing = []
            if self.pos_model is None:
                missing.append("position")
            if self.quat_model is None:
                missing.append("quaternion")
            raise ValueError(f"Missing required models: {', '.join(missing)}. Either configure a unified model or both position and quaternion models.")
