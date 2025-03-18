import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
import jax.numpy as jnp
import math


def load_data(path, regex=None):
    if regex is None:
        traj_files = sorted(glob.glob(os.path.join(path, "*")))
    else:
        traj_files = sorted(glob.glob(os.path.join(path, regex)))

    # Load each file and store in a list
    trajs = []
    for file in traj_files:
        try:
            traj = np.load(file)
            trajs.append(traj)
        except Exception as e:
            print(f"Error loading {os.path.basename(file)}: {str(e)}")

    return trajs


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
