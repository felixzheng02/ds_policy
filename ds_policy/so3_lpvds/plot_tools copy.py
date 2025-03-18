import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.spatial.transform import Rotation as R
from .quat_tools import *
import random


# font = {'family' : 'Times New Roman',
#          'size'   : 18
#          }
# mpl.rc('font', **font)


def plot_omega(omega_test):
    """
    Plot the angular velocity components over simulation time.
    
    This function visualizes the three components of angular velocity (ω_x, ω_y, ω_z)
    across the simulation trajectory. These represent the rotational velocities around
    each axis obtained from the quaternion dynamical system simulation.
    
    Parameters:
    ----------
    omega_test : ndarray, shape (M, 3)
        Angular velocity values from simulation, where:
        - M is the number of time steps
        - Each row contains the 3 components of angular velocity [ω_x, ω_y, ω_z]
        
    Returns:
    -------
    None. Displays the plot with three subplots, one for each angular velocity component.
    """
    # Get dimensions of angular velocity data
    M, N = omega_test.shape
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    # Define colors for each component (x,y,z)
    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    # Plot each component of angular velocity in a separate subplot
    for k in range(3):
        axs[k].scatter(np.arange(M), omega_test[:, k], s=5, color=colors[k])
        # axs[k].set_ylim([0, 1])
    
    # Add labels
    axs[0].set_title("Angular Velocity Components Over Time")
    axs[2].set_xlabel("Simulation Step")
    axs[0].set_ylabel("ω_x (rad/s)")
    axs[1].set_ylabel("ω_y (rad/s)")
    axs[2].set_ylabel("ω_z (rad/s)")


def plot_gamma(gamma_arr, **argv):
    """
    Plot the GMM responsibility values over simulation time.
    
    This function visualizes how the responsibility values (γ) for each Gaussian component
    in the mixture model change throughout the simulation. These values represent 
    the probability that each data point belongs to a specific component/cluster.
    
    Parameters:
    ----------
    gamma_arr : ndarray, shape (M, K)
        GMM responsibility values from simulation, where:
        - M is the number of time steps
        - K is the number of Gaussian components
        - Each row contains the responsibility values for all components at that time step
        
    **argv : dict
        Optional arguments:
        - title : str
            Custom title for the plot
            
    Returns:
    -------
    None. Displays the plot with K subplots, one for each Gaussian component.
    """
    # Get dimensions of responsibility values
    M, K = gamma_arr.shape

    # Create subplots, one for each Gaussian component
    fig, axs = plt.subplots(K, 1, figsize=(12, 8))

    # Define colors for each component
    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    # If there's only one component, make axes indexable
    if K == 1:
        axs = [axs]
        
    # Plot each component's responsibility values
    for k in range(K):
        axs[k].scatter(np.arange(M), gamma_arr[:, k], s=5, color=colors[k])
        axs[k].set_ylim([0, 1])
        axs[k].set_ylabel(f"γ_{k}")
    
    # Add appropriate title and labels
    if "title" in argv:
        axs[0].set_title(argv["title"])
    else:
        axs[0].set_title(r"Gaussian Component Responsibilities $\gamma(\cdot)$ over Time")
    
    axs[-1].set_xlabel("Simulation Step")


def plot_gmm(p_in, gmm):
    """
    Visualize the GMM clustering results in 3D space.
    
    This function plots the clustered data points in 3D space, with colors representing
    their cluster assignments, and visualizes the principal axes of each Gaussian component.
    
    Parameters:
    ----------
    p_in : ndarray, shape (M, 3)
        3D position data points (e.g., quaternion data projected into 3D)
        
    gmm : gmm_class
        Gaussian Mixture Model object containing:
        - assignment_arr: Cluster assignments for each data point
        - K: Number of Gaussian components
        - gaussian_list: Parameters of each Gaussian component
        
    Returns:
    -------
    None. Displays a 3D plot showing clustered data and Gaussian component orientations.
    """
    # Get cluster assignments and number of components
    label = gmm.assignment_arr
    K     = gmm.K

    # Define colors for data point clusters
    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    # Map colors to data points based on their cluster assignment
    color_mapping = np.take(colors, label)

    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    # Plot data points colored by cluster assignment
    ax.scatter(p_in[:, 0], p_in[:, 1], p_in[:, 2], 'o', color=color_mapping[:], s=1, alpha=0.4, label="Demonstration")

    # Colors for principal axes visualization (colorblind-friendly)
    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB

    # Calculate scale for principal axes visualization
    x_min, x_max = ax.get_xlim()
    scale = (x_max - x_min)/4
    
    # For each cluster, visualize its principal axes
    for k in range(K):
        # Get points in this cluster
        label_k = np.where(label == k)[0]
        p_in_k = p_in[label_k, :]
        loc = np.mean(p_in_k, axis=0)  # Calculate cluster center

        # Get rotation matrix for this cluster's orientation
        r = gmm.gaussian_list[k]["mu"]
        
        # Draw principal axes (x, y, z) for this cluster
        for j, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis), colors)):
            line = np.zeros((2, 3))
            line[1, j] = scale
            line_rot = r.apply(line)
            line_plot = line_rot + loc
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c, linewidth=1)


    # Configure axes appearance
    ax.axis('equal')
    ax.set_xlabel(r'$\xi_1$', labelpad=20)
    ax.set_ylabel(r'$\xi_2$', labelpad=20)
    ax.set_zlabel(r'$\xi_3$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


