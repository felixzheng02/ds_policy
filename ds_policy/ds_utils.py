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
        if option == "move_towards":
            seg_num_int = 0
        elif option == "move_away":
            seg_num_int = 1
        elif option == "reach_behind_and_pull":
            seg_num_int = 2
    elif task == "PnPCounterToCab":
        object_of_interest = "object"
        if option == "PnPCounterToCab_Pick_option":
            seg_num_int = 0
        elif option == "PnPCounterToCab_Place_option":
            seg_num_int = 1
    else:
        raise ValueError(f"Invalid option: {option}")
    input_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "demo_data",
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
            # Save initial handle pose
            object_of_interest_pos_init = object_of_interest_pos[0]
            object_of_interest_rot_init = object_of_interest_rot[0]

            # Compute relative position in world frame
            rel_pos_world = eef_pos - object_of_interest_pos_init

            # Transform relative positions to initial handle frame
            pos_traj = np.zeros_like(rel_pos_world)

            rot_traj = np.zeros_like(eef_rot)
            for i in range(len(rel_pos_world)):
                # Transform to initial handle frame
                pos_traj[i] = object_of_interest_rot_init.T @ rel_pos_world[i]
                rot_traj[i] = object_of_interest_rot_init.T @ eef_rot[i]
        else:
            pos_traj = eef_pos
            rot_traj = eef_rot

        # Convert rotation matrices to quaternions
        quat_traj_scipy = R.from_matrix(rot_traj)
        quat_traj = quat_traj_scipy.as_quat()
        # Compute velocities
        dt = 1 / 60
        vel_traj = np.diff(pos_traj, axis=0) / dt
        vel_traj = np.vstack([vel_traj, vel_traj[-1]])

        # compute angular velocity
        # do finite difference of quat_traj
        ang_vel_traj = np.zeros((len(quat_traj_scipy), 3))
        time_vec = np.arange(len(quat_traj_scipy)) * dt
        rs = RS(time_vec, quat_traj_scipy)
        ang_vel_traj = rs(time_vec, 1)
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


def quat_to_euler(quat):
    return R.from_quat(quat).as_euler("xyz", degrees=False)


def euler_to_quat(euler):
    return R.from_euler("xyz", euler, degrees=False).as_quat()


def quat_mult(q1, q2):
    q1_s = q1[0]
    q2_s = q2[0]
    q1_v = q1[1:].reshape((-1, 1))
    q2_v = q2[1:].reshape((-1, 1))
    scalar = q1_s * q2_s - q1_v.T @ q2_v
    skew = jnp.array([[0, -q1[3], q1[2]], [q1[3], 0, -q1[1]], [-q1[2], q1[1], 0]])
    vector = q1_s * q2_v + q2_s * q1_v + skew @ q2_v
    q_result = jnp.concatenate((scalar, vector), axis=0).flatten()
    return q_result


class Quaternion:
    def __init__(self, scalar=1, vec=[0, 0, 0]):
        self.q = np.array([scalar, 0.0, 0.0, 0.0])
        self.q[1:4] = vec

    def normalize(self):
        self.q = self.q / np.linalg.norm(self.q)

    def scalar(self):
        return self.q[0]

    def vec(self):
        return self.q[1:4]

    def axis_angle(self):
        theta = 2 * math.acos(self.scalar())
        vec = self.vec()
        if np.linalg.norm(vec) == 0:
            return np.zeros(3)
        vec = vec / np.linalg.norm(vec)
        return vec * theta

    def euler_angles(self):
        phi = math.atan2(
            2 * (self.q[0] * self.q[1] + self.q[2] * self.q[3]),
            1 - 2 * (self.q[1] ** 2 + self.q[2] ** 2),
        )
        theta = math.asin(2 * (self.q[0] * self.q[2] - self.q[3] * self.q[1]))
        psi = math.atan2(
            2 * (self.q[0] * self.q[3] + self.q[1] * self.q[2]),
            1 - 2 * (self.q[2] ** 2 + self.q[3] ** 2),
        )
        return np.array([phi, theta, psi])

    def from_axis_angle(self, a):
        angle = np.linalg.norm(a)
        if angle != 0:
            axis = a / angle
        else:
            axis = np.array([1, 0, 0])
        self.q[0] = math.cos(angle / 2)
        self.q[1:4] = axis * math.sin(angle / 2)
        # self.normalize()

    def from_rotm(self, R):
        theta = math.acos((np.trace(R) - 1) / 2)
        omega_hat = (R - np.transpose(R)) / (2 * math.sin(theta))
        omega = np.array([omega_hat[2, 1], -omega_hat[2, 0], omega_hat[1, 0]])
        self.q[0] = math.cos(theta / 2)
        self.q[1:4] = omega * math.sin(theta / 2)
        self.normalize()

    def inv(self):
        q_inv = Quaternion(self.scalar(), -self.vec())
        q_inv.normalize()
        return q_inv

    # Implement quaternion multiplication
    def __mul__(self, other):
        t0 = (
            self.q[0] * other.q[0]
            - self.q[1] * other.q[1]
            - self.q[2] * other.q[2]
            - self.q[3] * other.q[3]
        )
        t1 = (
            self.q[0] * other.q[1]
            + self.q[1] * other.q[0]
            + self.q[2] * other.q[3]
            - self.q[3] * other.q[2]
        )
        t2 = (
            self.q[0] * other.q[2]
            - self.q[1] * other.q[3]
            + self.q[2] * other.q[0]
            + self.q[3] * other.q[1]
        )
        t3 = (
            self.q[0] * other.q[3]
            + self.q[1] * other.q[2]
            - self.q[2] * other.q[1]
            + self.q[3] * other.q[0]
        )
        retval = Quaternion(t0, [t1, t2, t3])
        return retval

    def __str__(self):
        return str(self.scalar()) + ", " + str(self.vec())


# q = Quaternion()
# R = np.array([[1, 0, 0],[0, 0.707, -0.707],[0, 0.707, 0.707]])
# q.from_rotm(R)
# print q
