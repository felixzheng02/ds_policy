import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation as R

from .util import optimize_tools, quat_tools
from .gmm_class import gmm_class


# Helper function to write data to a JSON file
def write_json(data, path):
    """Writes a dictionary to a JSON file with indentation."""
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


# Helper function to compute angular velocity
def compute_ang_vel(q_k, q_kp1, dt):
    """ Computes the angular velocity between two orientations in the fixed frame.

    Args:
        q_k (Rotation): Orientation at time step k.
        q_kp1 (Rotation): Orientation at time step k+1.
        dt (float): Time difference between steps.

    Returns:
        np.ndarray: Angular velocity vector.
    """
    # dq = q_k.inv() * q_kp1    # from q_k to q_kp1 in body frame
    dq = q_kp1 * q_k.inv()    # from q_k to q_kp1 in fixed frame

    dq = dq.as_rotvec()
    w  = dq / dt

    return w


# Class implementing the SE(3) Locally Parameterized Vector Field Stable Dynamical System (LPV-DS)
class se3_class:
    def __init__(self, p_in:np.ndarray, q_in:list, p_out:np.ndarray, q_out:list, p_att:np.ndarray, q_att:R, dt:float, K_init:int) -> None:
        """ Initializes the SE(3) LPV-DS model with training data and parameters.

        Parameters:
        ----------
            p_in (np.ndarray):      [M, N] NumPy array of POSITION INPUT

            q_in (list):            M-length List of Rotation objects for ORIENTATION INPUT

            p_out (np.ndarray):     [M, N] NumPy array of POSITION OUTPUT

            q_out (list):           M-length List of Rotation objects for ORIENTATION OUTPUT

            p_att (np.ndarray):     [1, N] NumPy array of POSITION ATTRACTOR

            q_att (Rotation):       Single Rotation object for ORIENTATION ATTRACTOR
            
            dt (float):             TIME DIFFERENCE in differentiating ORIENTATION

            K_init (int):           Initial Number of Gaussian Components for GMM

            M:                      Number of Observations

            N:                      Observation dimension (assuming 3D position)
        """
        # Standardize quaternion signs (ensure scalar part w is negative)
        standardized_q_in = []
        for q in q_in:
            quat = q.as_quat()
            if quat[3] > 0:
                standardized_q_in.append(R.from_quat(-quat))
            else:
                standardized_q_in.append(q)

        standardized_q_out = []
        for q in q_out:
            quat = q.as_quat()
            if quat[3] > 0:
                standardized_q_out.append(R.from_quat(-quat))
            else:
                standardized_q_out.append(q)

        q_att_quat = q_att.as_quat()
        if q_att_quat[3] > 0:
            q_att = R.from_quat(-q_att_quat)

        # store parameters
        self.p_in  = p_in
        self.q_in  = standardized_q_in

        self.p_out = p_out
        self.q_out = standardized_q_out

        self.p_att = p_att
        self.q_att = q_att

        self.dt = dt
        self.K_init = K_init
        self.M = len(q_in)
        self.N = 7 # Dimension of state (3 pos + 4 quat)

        # simulation parameters
        self.tol = 10E-3
        self.max_iter = 5000


        # define output path
        file_path           = os.path.dirname(os.path.realpath(__file__))  
        self.output_path    = os.path.join(os.path.dirname(file_path), 'output.json')


    def _cluster(self):
        """ Clusters the input data using Gaussian Mixture Models (GMM).

        Fits a GMM to the input position and orientation data (p_in, q_in) relative
        to the orientation attractor (q_att). Stores the resulting responsibilities (gamma),
        number of components (K), and the GMM object itself.
        """
        gmm = gmm_class(self.p_in, self.q_in, self.q_att, self.K_init)  

        self.gamma    = gmm.fit()  # K by M
        self.K        = gmm.K
        self.gmm      = gmm


    def _optimize(self):
        """ Optimizes the linear dynamics matrices for position and orientation.

        Computes the optimal A_pos and A_ori matrices for each Gaussian component
        based on the input/output data, attractor points, and GMM responsibilities.
        It handles the dual cover of quaternions by optimizing separately for q and -q.
        """
        A_pos = optimize_tools.optimize_pos(self.p_in, self.p_out, self.p_att, self.gamma)
        A_ori = optimize_tools.optimize_ori(self.q_in, self.q_out, self.q_att, self.gamma)

        q_in_dual   = [R.from_quat(-q.as_quat()) for q in self.q_in]
        q_out_dual  = [R.from_quat(-q.as_quat()) for q in self.q_out]
        q_att_dual =  R.from_quat(-self.q_att.as_quat())
        A_ori_dual = optimize_tools.optimize_ori(q_in_dual, q_out_dual, q_att_dual, self.gamma)

        self.A_pos = np.concatenate((A_pos, A_pos), axis=0)
        self.A_ori = np.concatenate((A_ori, A_ori_dual), axis=0)


    def begin(self):
        """ Performs the main training pipeline: clustering, optimization, and logging. """
        self._cluster()
        self._optimize()
        self._logOut()


    def sim(self, p_init, q_init, step_size):
        """ Simulates the learned SE(3) LPV-DS from an initial pose.

        Args:
            p_init (np.ndarray): Initial position.
            q_init (Rotation): Initial orientation.
            step_size (float): Integration step size for the simulation.

        Returns:
            tuple: Contains the trajectory of positions, orientations, GMM responsibilities,
                   linear velocities, and angular velocities during the simulation.
        """
        p_test = [p_init.reshape(1, -1)]
        q_test = [q_init]

        gamma_test = []

        v_test = []
        w_test = []

        i = 0
        while np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec()) >= self.tol or np.linalg.norm((p_test[-1] - self.p_att)) >= self.tol:
            if i > self.max_iter:
                print("Exceed max iteration")
                break
            
            p_in  = p_test[i]
            q_in  = q_test[i]

            p_next, q_next, gamma, v, w = self.step(p_in, q_in, step_size)

            p_test.append(p_next)        
            q_test.append(q_next)        
            gamma_test.append(gamma[:, 0])
            v_test.append(v)
            w_test.append(w)

            i += 1

        return np.vstack(p_test), q_test, np.array(gamma_test), v_test, w_test
        

    def step(self, p_in, q_in, step_size):
        """ Performs one integration step of the SE(3) LPV-DS.

        Calculates the desired linear velocity (v) and angular velocity (w)
        at the current pose (p_in, q_in) based on the learned dynamics (A_pos, A_ori)
        and the GMM responsibilities (gamma). It considers both the primary
        and dual quaternion covers. Then, integrates forward in time using the
        specified step_size.

        Args:
            p_in (np.ndarray): Current position.
            q_in (Rotation): Current orientation.
            step_size (float): Integration step size.

        Returns:
            tuple: Contains the next position (p_next), next orientation (q_next),
                   GMM responsibilities (gamma) for the current step, calculated
                   linear velocity (v), and calculated angular velocity (w).
        """

        # read parameters
        A_pos = self.A_pos  # (2K, N, N)
        A_ori = self.A_ori  # (2K, N, N)

        p_att = self.p_att.reshape(1, -1)
        q_att = self.q_att

        K     = self.K
        gmm   = self.gmm
        

        # compute gamma
        gamma = gmm.logProb(p_in, q_in)   # gamma value 


        # compute output
        v     = np.zeros((3, 1))
        p_diff  = p_in - p_att
        
        q_out_att = np.zeros((4, 1))
        q_diff  = quat_tools.riem_log(q_att, q_in)
        for k in range(K):
            v         += gamma[k, 0] * A_pos[k] @ p_diff.T
            q_out_att += gamma[k, 0] * A_ori[k] @ q_diff.T
        q_out_body = quat_tools.parallel_transport(q_att, q_in, q_out_att.T)
        q_out_q    = quat_tools.riem_exp(q_in, q_out_body) 
        q_out      = R.from_quat(q_out_q.reshape(4,))
        w          = compute_ang_vel(q_in, q_out, self.dt)  


        # dual cover
        q_att_dual = R.from_quat(-q_att.as_quat())
        q_out_att_dual = np.zeros((4, 1))
        q_diff_dual  = quat_tools.riem_log(q_att_dual, q_in)
        for k in range(K):
            v              += gamma[self.K+k, 0] * A_pos[self.K+k] @ p_diff.T
            q_out_att_dual += gamma[self.K+k, 0] * A_ori[self.K+k] @ q_diff_dual.T
        q_out_body_dual = quat_tools.parallel_transport(q_att_dual, q_in, q_out_att_dual.T)
        q_out_q_dual    = quat_tools.riem_exp(q_in, q_out_body_dual) 
        q_out_dual      = R.from_quat(q_out_q_dual.reshape(4,))
        w               += compute_ang_vel(q_in, q_out_dual, self.dt)  


        # propagate forward
        p_next     = p_in + v.T * step_size
        q_next     = R.from_rotvec(w * step_size) * q_in  #compose in world frame
        # q_next     = q_in * R.from_rotvec(w * step_size)   #compose in body frame


        return p_next, q_next, gamma, v, w
    

    def _logOut(self, *args):
        """ Logs the learned model parameters to a JSON file.

        Saves the GMM parameters (Priors, Mu, Sigma), the optimized dynamics matrices
        (A_pos, A_ori), attractor information, and other relevant parameters into a
        JSON file. Allows specifying an output directory.

        Args:
            *args: Optional argument tuple. If provided, the first element is used
                   as the output directory path. Otherwise, uses the default output path.
        """

        Prior = self.gmm.Prior
        Mu    = self.gmm.Mu
        Mu_rollout = [np.hstack((p_mean, q_mean.as_quat())) for (p_mean, q_mean) in Mu]
        Sigma = self.gmm.Sigma

        Mu_arr      = np.zeros((2 * self.K, self.N)) 
        Sigma_arr   = np.zeros((2 * self.K, self.N, self.N), dtype=np.float32)

        for k in range(2 * self.K):
            Mu_arr[k, :] = Mu_rollout[k]
            Sigma_arr[k, :, :] = Sigma[k]

        json_output = {
            "name": "SE3-LPVDS",

            "K": self.K,
            "M": 7,
            "Prior": Prior,
            "Mu": Mu_arr.ravel().tolist(),
            "Sigma": Sigma_arr.ravel().tolist(),

            'A_pos': self.A_pos.ravel().tolist(),
            'A_ori': self.A_ori.ravel().tolist(),
            'att_pos': self.p_att.ravel().tolist(),
            'att_ori': self.q_att.as_quat().ravel().tolist(),
            "dt": self.dt,
            "gripper_open": 0
        }

        if len(args) == 0:
            write_json(json_output, self.output_path)
        else:
            write_json(json_output, os.path.join(args[0], '1.json'))

    def compute_reconstruction_error(self):
        error = 0
        total_pts = self.p_in.shape[0]
        for i in range(total_pts):
            p_in = self.p_in[i]
            q_in = self.q_in[i]
            p_out = self.p_out[i]
            q_out = self.q_out[i]
            p_dot_pred, q_dot_pred, gamma, v, w = self.step(p_in, q_in, 1)
            error += np.linalg.norm(p_out - p_dot_pred) + np.rad2deg(np.linalg.norm(quat_tools.riem_log(q_out, q_dot_pred)))
        return error / total_pts