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
    def __init__(self, ds_policy):
        self.ds_policy = ds_policy
        self.demo_traj_probs_history = []

    def test(self, init_state: np.ndarray, save_dir: str, n_steps: int):

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

        np.savez(
            os.path.join(save_dir, "demo_traj_probs_history.npz"),
            demo_traj_probs_history=self.demo_traj_probs_history,
            allow_pickle=True,
        )

        return self.demo_traj_probs_history

    def resample(self, cur_state: np.ndarray):
        self.ds_policy.update_demo_traj_probs(
            x_state=cur_state[:3], radius=0.1, penalty=0.1
        )
        self.demo_traj_probs_history.append(self.ds_policy.demo_traj_probs)


class TestRealtimeResampleAnimator(Animator):
    def __init__(
        self,
        traj: np.ndarray,
        demo_trajs: list[np.ndarray],
        ref_traj_indices: list[int],
        demo_traj_probs_history: list,
    ):
        super().__init__(traj, demo_trajs, ref_traj_indices)
        self.demo_traj_probs_history = demo_traj_probs_history
        self.current_step_index = 0
        self.step_boundaries = np.cumsum(
            [0] + n_steps
        )  # Track where each simulation segment begins
        self.demo_line_alpha = 1

    def animate(self, save_path: str = None, interval: int = 100):
        super().animate(save_path, interval)

    def update(self, frame):
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
    save_dir = "DS-Policy/data/test_realtime_resample"
    n_steps = [10, 10, 10]
    x, x_dot, quat, omega = load_data("custom")
    if True:
        ds_policy = DSPolicy(
            x,
            x_dot,
            quat,
            omega,
            dt=0.01,
            switch=False,
            lookahead=5,
            demo_traj_probs=None,
        )
        ds_policy.load_pos_model(
            pos_model_path="DS-Policy/models/mlp_width128_depth3.pt"
        )
        ds_policy.train_quat_model(
            save_path="DS-Policy/models/quat_model.json", k_init=10
        )

        rng = np.random.default_rng(seed=3)
        init_pos_x = rng.uniform(low=-0.3, high=0.3)
        init_pos_y = rng.uniform(low=-0.4, high=-0.1)
        init_pos_z = rng.uniform(low=-0.3, high=0.3)
        quat_rng = np.random.RandomState(seed=4)
        init_euler = R.random(random_state=quat_rng).as_euler("xyz", degrees=False)
        init_state = np.concatenate(
            [np.array([init_pos_x, init_pos_y, init_pos_z]), init_euler]
        )

        test_realtime_resample = TestRealtimeResample(ds_policy)
        demo_traj_probs_history = test_realtime_resample.test(
            init_state=np.concatenate([x[0][0], quat_to_euler(quat[0][0])]),
            save_dir=save_dir,
            n_steps=n_steps,
        )

    traj_all = []
    demo_trajs = [np.concatenate([x[i], quat[i]], axis=-1) for i in range(len(x))]
    ref_traj_indices_all = []
    for i in range(len(n_steps)):
        save_path = os.path.join(save_dir, f"simulator_{i}.npz")
        data = np.load(save_path, allow_pickle=True)
        traj = data["traj"]
        ref_traj_indices = data["ref_traj_indices"]
        traj_all.append(traj)
        ref_traj_indices_all.append(ref_traj_indices)
    demo_traj_probs_history = np.load(
        os.path.join(save_dir, "demo_traj_probs_history.npz"), allow_pickle=True
    )["demo_traj_probs_history"]

    traj_all = np.concatenate(traj_all, axis=0)
    ref_traj_indices_all = list(np.concatenate(ref_traj_indices_all, axis=0))
    animator = TestRealtimeResampleAnimator(
        traj_all, demo_trajs, ref_traj_indices_all, demo_traj_probs_history
    )
    animator.animate(None, interval=300)
