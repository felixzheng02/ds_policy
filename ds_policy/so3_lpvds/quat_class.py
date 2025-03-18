import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation as R

from . import optimize_tools, quat_tools
from .gmm_class import gmm_class


def write_json(data, path):
    """
    Utility function to write data to a JSON file with formatting
    
    Parameters:
    ----------
    data : dict
        The data to be written to the JSON file
    path : str
        The path where the JSON file will be saved
    """
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def compute_ang_vel(q_k, q_kp1, dt=0.01):
    """ 
    Compute angular velocity between two quaternion orientations
    
    Parameters:
    ----------
    q_k : Rotation
        The initial orientation (as scipy Rotation object)
    q_kp1 : Rotation
        The final orientation (as scipy Rotation object)
    dt : float
        Time difference between orientations
        
    Returns:
    -------
    w : ndarray
        Angular velocity vector
    """

    # dq = q_k.inv() * q_kp1    # from q_k to q_kp1 in body frame
    dq = q_kp1 * q_k.inv()    # from q_k to q_kp1 in fixed frame

    # Convert to rotation vector (axis-angle representation)
    dq = dq.as_rotvec() 
    # Scale by time to get angular velocity
    w  = dq / dt # this is roll, pitch, yaw
    
    return w


class quat_class:
    """
    Quaternion Dynamical System (DS) implementation using Gaussian Mixture Models (GMM)
    
    This class implements a dynamical system for orientation control using quaternions.
    It first fits a GMM to the input data, then optimizes DS parameters to match
    the observed trajectories while ensuring stability towards the attractor.
    
    The implementation uses a dual quaternion cover approach to handle the double
    cover property of quaternions (q and -q represent the same orientation).
    """
    def __init__(self, q_in:list, q_out:list, q_att:R, dt, K_init:int, output_path:str = None) -> None:
        """
        Initialize the quaternion dynamical system
        
        Parameters:
        ----------
            q_in (list):            M-length List of Rotation objects for ORIENTATION INPUT
                                    (demonstration orientations)

            q_out (list):           M-length List of Rotation objects for ORIENTATION OUTPUT
                                    (next orientations in demonstrations)

            q_att (Rotation):       Single Rotation object for ORIENTATION ATTRACTOR
                                    (the goal orientation)

            dt:                     TIME DIFFERENCE in differentiating ORIENTATION
                                    (time step used for computing angular velocities)
            
            K_init:                 Initial number of GAUSSIAN COMPONENTS
                                    (for the GMM clustering)

            M:                      OBSERVATION size (number of demonstration points)

            N:                      OBSERVATION dimension (quaternions have 4 dimensions)
        """

        # Store the demonstration data and parameters
        self.q_in  = q_in
        self.q_out = q_out
        self.q_att = q_att

        self.dt = dt
        self.K_init = K_init
        self.M = len(q_in)
        self.N = 4  # quaternions are 4-dimensional

        # Simulation convergence parameters
        self.tol = 10E-3      # Tolerance for convergence to attractor
        self.max_iter = 5000  # Maximum iterations for simulation

        # Define output path for storing model parameters
        # file_path           = os.path.dirname(os.path.realpath(__file__))  
        # self.output_path    = os.path.join(os.path.dirname(file_path), 'output_ori.json')
        self.output_path = output_path if output_path is not None else 'quat_model.json'


    def _cluster(self):
        """
        Cluster the orientation data using GMM
        
        This method fits a Gaussian Mixture Model to the input orientations
        to identify regions in orientation space with similar dynamics.
        """
        # Create GMM model with input orientations and attractor
        gmm = gmm_class(self.q_in, self.q_att, self.K_init)  

        # Fit the GMM and store the results
        self.gamma = gmm.fit()  # (K, M) - Responsibility matrix
        self.K = gmm.K          # Final number of Gaussian components
        self.gmm = gmm          # Store the GMM model


    def _optimize(self):
        """
        Optimize the dynamical system parameters
        
        This fits linear dynamics matrices for each Gaussian component
        to match the demonstrated trajectories while ensuring stability.
        Uses a dual quaternion cover approach to handle the double cover property.
        """
        # Optimize dynamics matrices using the input/output orientation pairs
        A_ori = optimize_tools.optimize_ori(self.q_in, self.q_out, self.q_att, self.gamma)

        # Create dual quaternion representation (handling the double cover property)
        # The double cover means that q and -q represent the same orientation
        q_in_dual   = [R.from_quat(-q.as_quat()) for q in self.q_in]
        q_out_dual  = [R.from_quat(-q.as_quat()) for q in self.q_out]
        q_att_dual =  R.from_quat(-self.q_att.as_quat())
        
        # Optimize dynamics matrices for the dual quaternions
        A_ori_dual = optimize_tools.optimize_ori(q_in_dual, q_out_dual, q_att_dual, self.gamma)

        # Combine original and dual dynamics matrices
        self.A_ori = np.concatenate((A_ori, A_ori_dual), axis=0)


    def begin(self):
        """
        Initialize the dynamical system model
        
        This method performs the complete model initialization:
        1. Clustering the orientation data
        2. Optimizing the dynamics parameters
        """
        self._cluster()    # Cluster the data using GMM
        self._optimize()   # Optimize the dynamics matrices
        # self._logOut()   # Disabled: Save model parameters to file


    def sim(self, q_init, step_size, steps=None):
        """
        Simulate the dynamical system from an initial orientation
        
        This method simulates the quaternion dynamical system trajectory starting from 
        an initial orientation until convergence to the attractor (goal orientation).
        At each time step, it computes the next orientation and stores trajectory information
        that can be used for analysis and visualization.
        
        The outputs of this function can be directly visualized using:
        - plot_omega(): To visualize the angular velocity components over time
        - plot_gamma(): To visualize the GMM responsibility values over time
        - plot_gmm(): To visualize the spatial clustering of the demonstration data
            
        Parameters:
        ----------
        q_init : Rotation
            Initial orientation for simulation (as scipy Rotation object)
        step_size : float
            Time step for numerical integration
            
        Returns:
        -------
        q_test : list
            List of orientations (Rotation objects) along the simulated trajectory
        gamma_test : ndarray, shape (steps, 2*K)
            GMM component responsibilities at each step:
            - rows: time steps
            - columns: responsibility values for each Gaussian component (including dual cover)
        omega_test : ndarray, shape (steps, 3)
            Angular velocities at each step:
            - rows: time steps
            - columns: [ω_x, ω_y, ω_z] components of angular velocity
        """
        if steps is None:
            steps = self.max_iter
        # Initialize lists to store trajectory information
        q_test = [q_init]      # List of orientations
        gamma_test = []        # GMM responsibility values
        omega_test = []        # Angular velocities

        i = 0
        # Continue until we reach the attractor (within tolerance)
        # using the geodesic distance on SO(3) to measure convergence
        while np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec()) >= self.tol:
            if i > self.max_iter:
                print("Exceed max iteration")
                break
            if i > steps:
                break
            
            q_in = q_test[i]  # Current orientation

            # Compute next orientation and store results
            q_next, gamma, omega = self._step(q_in, step_size)

            q_test.append(q_next)        
            gamma_test.append(gamma[:, 0])
            omega_test.append(omega)

            i += 1

        return q_test, np.array(gamma_test), np.array(omega_test)
        

    def _step(self, q_in, step_size):
        """ 
        Integrate the dynamical system forward by one time step
        
        This implements the quaternion-based dynamical system using the
        learned GMM and dynamics matrices. It applies both the original
        and dual dynamics to handle the quaternion double cover property.
        
        Parameters:
        ----------
        q_in : Rotation
            Current orientation
        step_size : float
            Time step for numerical integration
            
        Returns:
        -------
        q_next : Rotation
            Next orientation after one step
        gamma : ndarray
            GMM component responsibilities
        omega : ndarray
            Angular velocity
        """
        # Extract stored parameters
        A_ori = self.A_ori  # (2K, N, N) - Dynamics matrices for each component
        q_att = self.q_att  # Attractor orientation
        K     = self.K      # Number of Gaussian components
        gmm   = self.gmm    # GMM model

        # Compute GMM component responsibilities for current orientation
        gamma = gmm.logProb(q_in)   # (2K, 1)

        # --- First cover (original quaternion representation) ---
        q_out_att = np.zeros((4, 1))
        # Compute logarithmic map from attractor to current orientation (on manifold)
        q_diff = quat_tools.riem_log(q_att, q_in)
        # Weighted sum of dynamics matrices applied to the orientation difference
        for k in range(K):
            q_out_att += gamma[k, 0] * A_ori[k] @ q_diff.T
        # Parallel transport the result from attractor frame to current orientation frame
        q_out_body = quat_tools.parallel_transport(q_att, q_in, q_out_att.T)
        # Apply exponential map to get the next orientation
        q_out_q = quat_tools.riem_exp(q_in, q_out_body) 
        q_out = R.from_quat(q_out_q.reshape(4,))
        # Compute angular velocity from current to next orientation
        omega = compute_ang_vel(q_in, q_out, self.dt)  

        # --- Dual cover (negated quaternion representation) ---
        q_att_dual = R.from_quat(-q_att.as_quat())
        q_out_att_dual = np.zeros((4, 1))
        # Repeat the same process with dual quaternions
        q_diff_dual = quat_tools.riem_log(q_att_dual, q_in)
        for k in range(K):
            q_out_att_dual += gamma[self.K+k, 0] * A_ori[self.K+k] @ q_diff_dual.T
        q_out_body_dual = quat_tools.parallel_transport(q_att_dual, q_in, q_out_att_dual.T)
        q_out_q_dual = quat_tools.riem_exp(q_in, q_out_body_dual) 
        q_out_dual = R.from_quat(q_out_q_dual.reshape(4,))
        # Add the contribution from the dual representation
        omega += compute_ang_vel(q_in, q_out_dual, self.dt)  
        
        # Propagate forward using the computed angular velocity
        q_next = R.from_rotvec(omega * step_size) * q_in  # Compose in world frame
        # q_next = q_in * R.from_rotvec(w_out * step_size)   # Compose in body frame (alternative)

        return q_next, gamma, omega
            

    def _logOut(self, *args): 
        """
        Export the learned dynamical system parameters to a JSON file
        
        Parameters:
        ----------
        *args : str, optional
            If provided, the first argument is used as the output directory path
        """
        # Extract GMM parameters
        Prior = self.gmm.Prior
        Mu = self.gmm.Mu
        Mu_rollout = [q_mean.as_quat() for q_mean in Mu]
        Sigma = self.gmm.Sigma

        # Convert to arrays for JSON serialization
        Mu_arr = np.zeros((2 * self.K, self.N)) 
        Sigma_arr = np.zeros((2 * self.K, self.N, self.N), dtype=np.float32)

        for k in range(2 * self.K):
            Mu_arr[k, :] = Mu_rollout[k]
            Sigma_arr[k, :, :] = Sigma[k]

        # Create JSON structure with all model parameters
        json_output = {
            "name": "Quaternion-DS",

            "K": self.K,              # Number of Gaussian components
            "M": 4,                   # Quaternion dimension
            "Prior": Prior,           # GMM priors
            "Mu": Mu_arr.ravel().tolist(),        # GMM means (flattened)
            "Sigma": Sigma_arr.ravel().tolist(),  # GMM covariances (flattened)

            'A_ori': self.A_ori.ravel().tolist(),  # Dynamics matrices (flattened)
            'att_ori': self.q_att.as_quat().ravel().tolist(),  # Attractor orientation
            "dt": self.dt,            # Time step
            "gripper_open": 0         # Default gripper state
        }

        # Save to the specified path or default path
        if len(args) == 0:
            write_json(json_output, self.output_path)
        else:
            write_json(json_output, os.path.join(args[0], '1.json'))
