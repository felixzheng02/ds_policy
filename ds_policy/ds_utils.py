import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline as RS
import jax.numpy as jnp
import math


def load_data(task: str, option: str, finger: bool = False, transform_to_object_of_interest_frame: bool = True, debug_on: bool = False):
    path = task
    if task == "OpenSingleDoor":
        object_of_interest = "handle"
        if option == "OpenSingleDoor_MoveTowards_option":
            seg_num_int = 0
        elif option == "OpenSingleDoor_MoveAway_option":
            seg_num_int = 1
        elif option == "reach_behind_and_pull":
            seg_num_int = 2
    elif task == "PnPCounterToCab":
        if option == "PnPCounterToCab_Pick_option":
            seg_num_int = 0
            object_of_interest = "object"
        elif option == "PnPCounterToCab_Place_option":
            seg_num_int = 1
            object_of_interest = "bottom"
    else:
        raise ValueError(f"Invalid option: {option}")
    input_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "demo_data",
        # "smoothing_window_21_test_saving_contact"
        path,
    )
    seg_num = str(seg_num_int).zfill(2)
    traj_files = [
        f
        for f in os.listdir(input_path)
        if f.endswith("_eef_traj.npy") and f"_seg_{seg_num}_" in f
    ]
    L = len(traj_files)

    x = []
    x_dot = []
    quat_traj_all = []
    ang_vel_traj_all = []
    gripper_traj = []

    for l in range(L):
        demo_num = str(l).zfill(2)

        # Check if eef_traj file exists, if not, continue to next iteration
        if not os.path.exists(os.path.join(input_path, f"demo_{demo_num}_seg_{seg_num}_eef_traj.npy")):
            continue
        
        eef_traj = np.load(
            os.path.join(input_path, f"demo_{demo_num}_seg_{seg_num}_eef_traj.npy")
        )
        object_of_interest_traj = np.load(
            os.path.join(
                input_path, f"demo_{demo_num}_seg_{seg_num}_{object_of_interest}_traj.npy"
            )
        )
        if finger:
            finger_dist_traj = np.load(
                os.path.join(
                    input_path, f"demo_{demo_num}_seg_{seg_num}_finger_dist_traj.npy"
                )
            )
        contact_traj = np.load(
            os.path.join(
                input_path, f"demo_{demo_num}_seg_{seg_num}_contact_traj.npy"
            ),
            allow_pickle=True,
        )
        # Extract positions and rotation matrices
        eef_pos = eef_traj[:, :3]
        eef_quat = eef_traj[:, 3:]
        object_of_interest_pos = object_of_interest_traj[:, :3]
        object_of_interest_quat = object_of_interest_traj[:, 3:]
        eef_rot = np.array([R.from_quat(q).as_matrix() for q in eef_quat])

        if transform_to_object_of_interest_frame:
            object_of_interest_rot = np.array([R.from_quat(q).as_matrix() for q in object_of_interest_quat])

            pos_traj, rot_traj = transform_frame(eef_pos, eef_rot, object_of_interest_pos, object_of_interest_rot)
        else:
            pos_traj = eef_pos
            rot_traj = eef_rot

        dt = 1 / 60
        quat_traj_scipy = R.from_matrix(rot_traj)
        quat_traj = quat_traj_scipy.as_quat()
        vel_traj, ang_vel_traj = compute_vel_traj(pos_traj, rot_traj, dt)
        # convert quat to euler angle plot derivative as arrows to make sure it is correct
        # plot euler_traj and ang_vel_traj_euler
        if debug_on:
            euler_traj = quat_traj_scipy.as_euler("xyz", degrees=False)
            ang_vel_traj_euler = np.diff(euler_traj, axis=0) / dt
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(2, 3, figsize=(12, 8))
            axs[0, 0].plot(ang_vel_traj[:, 0])
            axs[0, 0].set_title("Angular Velocity X")
            axs[0, 0].set_ylabel("Angular Velocity (rad/s)")
            axs[0, 1].plot(ang_vel_traj[:, 1])
            axs[0, 1].set_title("Angular Velocity Y")
            axs[0, 2].plot(ang_vel_traj[:, 2])
            axs[0, 2].set_title("Angular Velocity Z")

            axs[1, 0].plot(ang_vel_traj_euler[:, 0])
            axs[1, 0].set_title("Euler Angular Velocity X")
            axs[1, 0].set_xlabel("Time Step")
            axs[1, 0].set_ylabel("Angular Velocity (rad/s)")
            axs[1, 1].plot(ang_vel_traj_euler[:, 1])
            axs[1, 1].set_title("Euler Angular Velocity Y")
            axs[1, 1].set_xlabel("Time Step")
            axs[1, 2].plot(ang_vel_traj_euler[:, 2])
            axs[1, 2].set_title("Euler Angular Velocity Z")
            axs[1, 2].set_xlabel("Time Step")

            plt.tight_layout()
            plt.show()

        # Store trajectories
        # if l == 0:
        #     x = [pos_traj]
        #     x_dot = [vel_traj]
        #     quat_traj_all = [quat_traj]
        #     ang_vel_traj_all = [ang_vel_traj]
        # else:
        if transform_to_object_of_interest_frame:
            pos_traj_final = pos_traj
            quat_traj_final = quat_traj
        else:
            pos_traj_final = np.concatenate([eef_pos, object_of_interest_pos], axis=1)
            quat_traj_final = np.concatenate([eef_quat, object_of_interest_quat], axis=1)
        x.append(pos_traj_final)
        x_dot.append(vel_traj)
        quat_traj_all.append(quat_traj_final)
        ang_vel_traj_all.append(ang_vel_traj)

        # gripper control
        if finger:
            gripper_threshold = 0.1 # threshold to distinguish open and close
            cur_gripper_traj = finger_dist_traj < gripper_threshold
            cur_gripper_traj = cur_gripper_traj.astype(float)  # Convert boolean to float
            cur_gripper_traj[cur_gripper_traj == 0] = -1  # Change 0 (open) to -1
            gripper_traj.append(cur_gripper_traj)


    return x, x_dot, quat_traj_all, ang_vel_traj_all, gripper_traj


def transform_frame(origin_pos: np.ndarray, origin_rot: np.ndarray, target_pos: np.ndarray, target_rot: np.ndarray):
    """
    Args:
        origin_pos: np.ndarray, shape (N, 3)
        origin_rot: np.ndarray, shape (N, 3, 3)
        target_pos: np.ndarray, shape (N, 3)
        target_rot: np.ndarray, shape (N, 3, 3)
    Returns:
        pos_traj: np.ndarray, shape (N, 3), transformed position
        rot_traj: np.ndarray, shape (N, 3, 3), transformed rotation
    """
    rel_pos = origin_pos - target_pos

    # Transform relative positions to initial handle frame
    pos_traj = np.zeros_like(rel_pos)

    rot_traj = np.zeros_like(target_rot)
    for i in range(len(rel_pos)):
        # Transform to initial handle frame
        pos_traj[i] = target_rot[i].T @ rel_pos[i]
        rot_traj[i] = target_rot[i].T @ origin_rot[i]

    return pos_traj, rot_traj

def compute_vel_traj(pos_traj: np.ndarray, rot_traj: np.ndarray, dt: float):
    """
    Args:
        pos_traj: np.ndarray, shape (N, 3)
        rot_traj: np.ndarray, shape (N, 3, 3)
        dt: float, time step
    Returns:
        translational_vel_traj: np.ndarray, shape (N, 3), translational velocity
        angular_vel_traj: np.ndarray, shape (N, 3), angular velocity
    """
    quat_traj_scipy = R.from_matrix(rot_traj)
    quat_traj = quat_traj_scipy.as_quat()

    translational_vel_traj = np.diff(pos_traj, axis=0) / dt
    translational_vel_traj = np.vstack([translational_vel_traj, translational_vel_traj[-1]])

    ang_vel_traj = np.zeros((len(quat_traj_scipy), 3))
    time_vec = np.arange(len(quat_traj_scipy)) * dt
    rs = RS(time_vec, quat_traj_scipy)
    ang_vel_traj = rs(time_vec, 1)

    return translational_vel_traj, ang_vel_traj