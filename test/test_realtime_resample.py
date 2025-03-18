import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ds_policy import DSPolicy
from src.load_tools import load_data
from src.ds_utils import quat_to_euler

from test_ds_policy import Simulator, Animator


class TestRealtimeResample:
    """
    Test class for demonstrating real-time trajectory resampling with obstacle avoidance.
    
    This class simulates a robot following a trajectory while dynamically updating
    trajectory probabilities when encountering obstacles or constraints.
    """
    
    def __init__(self, ds_policy):
        """
        Initialize the test with a DS policy.
        
        Args:
            ds_policy: A configured DSPolicy instance
        """
        self.ds_policy = ds_policy
        self.demo_traj_probs_history = []

    def test(self, init_state: np.ndarray, save_dir: str, n_steps: list[int]):
        """
        Run a multi-segment test with trajectory resampling between segments.
        
        Args:
            init_state: Initial state (position + euler angles)
            save_dir: Directory to save simulation results
            n_steps: List of steps for each simulation segment
            
        Returns:
            demo_traj_probs_history: History of trajectory probabilities
        """
        k = len(n_steps)

        simulators = [Simulator(self.ds_policy) for _ in range(k)]

        self.demo_traj_probs_history.append(self.ds_policy.demo_traj_probs)

        cur_state = init_state
        for i in range(k):
            cur_save_path = os.path.join(save_dir, f"simulator_{i}.npz")
            traj, _, _ = simulators[i].simulate(
                cur_state, cur_save_path, n_steps[i], clf=True, alpha_V=30, lookahead=5
            )
            cur_state = traj[-1]
            if i < k - 1:
                self.resample(cur_state)

        # Create directory for save_path if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, "demo_traj_probs_history.npz"),
            demo_traj_probs_history=self.demo_traj_probs_history,
            allow_pickle=True,
        )

        return self.demo_traj_probs_history

    def resample(self, cur_state: np.ndarray):
        """
        Update trajectory probabilities based on current state.
        
        Args:
            cur_state: Current state (position + euler angles)
        """
        self.ds_policy.update_demo_traj_probs(
            x_state=cur_state[:3], radius=0.05, penalty=0.1
        )
        self.demo_traj_probs_history.append(self.ds_policy.demo_traj_probs)


class TestRealtimeResampleAnimator(Animator):
    """
    Specialized animator for visualizing trajectory resampling.
    
    This animator changes the visibility of demonstration trajectories
    based on their changing probabilities during the simulation.
    """
    
    def __init__(
        self,
        traj: np.ndarray,
        demo_trajs: list[np.ndarray],
        ref_traj_indices: list[int],
        demo_traj_probs_history: list,
    ):
        """
        Initialize the resampling animator.
        
        Args:
            traj: Trajectory to animate
            demo_trajs: Demonstration trajectories
            ref_traj_indices: Reference trajectory indices for each frame
            demo_traj_probs_history: History of trajectory probabilities
        """
        super().__init__(traj, demo_trajs, ref_traj_indices)
        self.demo_traj_probs_history = demo_traj_probs_history
        self.current_step_index = 0
        self.step_boundaries = np.cumsum(
            [0] + n_steps
        )  # Track where each simulation segment begins
        self.demo_line_alpha = 1

    def update(self, frame):
        """
        Update animation for the given frame.
        
        Updates trajectory lines and changes the visibility of demo trajectories
        based on their probabilities at the current simulation segment.
        
        Args:
            frame: Current animation frame
        
        Returns:
            Updated graphical elements
        """
        (
            self.generated_line,
            self.ref_line,
            self.start_point,
            self.arrow_container[0],
            self.orientation_label,
        ) = super().update(frame)

        # If frame is 0, we're restarting the animation, so reset alpha values
        if frame == 0:
            self.current_step_index = 0
            # Reset all demo lines to original alpha
            for line in self.demo_lines:
                line.set_alpha(self.demo_line_alpha)

        # Determine which step we're in and corresponding probability distribution
        step_idx = np.searchsorted(self.step_boundaries, frame, side="right") - 1
        if step_idx != self.current_step_index and step_idx < len(
            self.demo_traj_probs_history
        ):
            self.current_step_index = step_idx

            # Update colors based on current probability distribution
            probs = self.demo_traj_probs_history[step_idx]
            for i, prob in enumerate(probs):
                self.demo_lines[i].set_alpha(max(0.05, prob * self.demo_line_alpha))

        return (
            self.generated_line,
            self.ref_line,
            self.start_point,
            self.arrow_container[0],
            self.orientation_label,
            self.demo_lines,
        )


if __name__ == "__main__":
    save_dir = "DSPolicy/data/test_realtime_resample"
    n_steps = [20, 20, 20]
    option = "move_towards"
    x, x_dot, quat, omega = load_data("custom", option)
    
    if True:  # Set to False to skip simulation and only animate existing results
        # Define model configuration using the new structure
        model_config = {
            'pos_model': {
                'load_path': f"DSPolicy/models/mlp_width128_depth3_{option}.pt"
            },
            'quat_model': {
                'save_path': f"DSPolicy/models/quat_model_{option}.json",
                'k_init': 10
            }
        }
        
        # Initialize DS policy with the new model_config parameter
        ds_policy = DSPolicy(
            x=x, 
            x_dot=x_dot, 
            quat=quat, 
            omega=omega, 
            model_config=model_config,
            dt=0.01, 
            switch=False, 
            lookahead=5
        )

        # Set up random initial state
        rng = np.random.default_rng(seed=3)
        init_pos_x = rng.uniform(low=-0.3, high=0.3)
        init_pos_y = rng.uniform(low=-0.4, high=-0.1)
        init_pos_z = rng.uniform(low=-0.3, high=0.3)
        quat_rng = np.random.RandomState(seed=4)
        init_euler = R.random(random_state=quat_rng).as_euler("xyz", degrees=False)
        init_state = np.concatenate(
            [np.array([init_pos_x, init_pos_y, init_pos_z]), init_euler]
        )

        # Run the resampling test
        test_realtime_resample = TestRealtimeResample(ds_policy)
        demo_traj_probs_history = test_realtime_resample.test(
            # init_state=np.concatenate([x[0][0], quat_to_euler(quat[0][0])]),
            init_state=init_state,
            save_dir=save_dir,
            n_steps=n_steps,
        )

    # Load and visualize results
    traj_all = []
    demo_trajs = [np.concatenate([x[i], quat[i]], axis=-1) for i in range(len(x))]
    ref_traj_indices_all = []
    
    # Load each segment's data
    for i in range(len(n_steps)):
        save_path = os.path.join(save_dir, f"simulator_{i}.npz")
        data = np.load(save_path, allow_pickle=True)
        traj = data["traj"]
        ref_traj_indices = data["ref_traj_indices"]
        traj_all.append(traj)
        ref_traj_indices_all.append(ref_traj_indices)
    
    # Load trajectory probability history
    demo_traj_probs_history = np.load(
        os.path.join(save_dir, "demo_traj_probs_history.npz"), allow_pickle=True
    )["demo_traj_probs_history"]

    # Combine all segments for visualization
    traj_all = np.concatenate(traj_all, axis=0)
    ref_traj_indices_all = list(np.concatenate(ref_traj_indices_all, axis=0))
    
    # Create and run the animator
    animator = TestRealtimeResampleAnimator(
        traj_all, demo_trajs, ref_traj_indices_all, demo_traj_probs_history
    )
    animator.animate(None, interval=300)
