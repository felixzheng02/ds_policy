import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate
import functools as ft
import cvxpy as cp
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from scipy.integrate import odeint


class NeuralODE(nn.Module):
    """Neural ODE function for 3D translational dynamics.

    This class defines a neural network that learns to predict translational and rotational velocities from positions and quaternions.
    The model takes in a 7D state vector and outputs 7D vector field.

    Structure:
    - Input: 7D state vector [position (3D), quaternion (4D)]
    - MLP: Maps state to velocities
    - Output: 7D vector field [translational velocity (3D), rotational velocity (4D)]
    """

    def __init__(self, data_size: int, width_size: int, depth: int, **kwargs):
        """Initialize the model.

        Args:
            data_size: Dimension of input state (3 for position)
            width_size: Width of hidden layers
            depth: Number of hidden layers
        """
        super(NeuralODE, self).__init__()

        if data_size == 3:  # only position
            output_size = 3
        elif data_size == 7:  # position and quaternion
            output_size = 6  # x_dot, omega
        else:
            raise ValueError(f"Invalid data size: {data_size}")

        # Create a list to hold all layers
        layers = []

        # Input layer
        layers.append(nn.Linear(data_size, width_size))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(width_size, width_size))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(width_size, output_size))

        # Combine all layers into a sequential model
        self.mlp = nn.Sequential(*layers)

        # Apply orthogonal initialization to all linear layers
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

    def forward(self, x):
        """Predict velocity directly from position.

        Args:
            x: Current state vector(s). Can be of either 3D or 7D.

        Returns:
            3D or 6D velocity
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        return self.mlp(x)


class NeuralODEWrapper(nn.Module):
    """
    Wrapper for NeuralODE for simulation.
    TODO: not used
    """

    def __init__(self, data_size, width_size, depth, **kwargs):
        super(NeuralODEWrapper, self).__init__()
        self.neural_ode = NeuralODE(data_size, width_size, depth)

    def predict_velocity(self, x: np.ndarray):
        """Predict velocity directly from position without ODE integration."""
        return self.neural_ode.forward(x)

    def generate_trajectory(self, ts, y0):
        """
        Generate a trajectory by solving the ODE.

        Args:
            ts: Time points to evaluate trajectory at
            y0: Initial 3D position

        Returns:
            Position trajectory evaluated at specified timepoints
        """
        # Convert inputs to torch tensors if they aren't already
        if not isinstance(ts, torch.Tensor):
            ts = torch.tensor(ts, dtype=torch.float32)
        if not isinstance(y0, torch.Tensor):
            y0 = torch.tensor(y0, dtype=torch.float32)

        # Make sure y0 has the right shape
        if y0.ndim == 1:
            y0 = y0.unsqueeze(0)  # Add batch dimension

        # Solve ODE
        solution = odeint(
            self.neural_ode,
            y0,
            ts,
            method="dopri5",
            rtol=1e-3,
            atol=1e-6,
            options={"max_num_steps": 4000},
        )

        return solution

    def forward(self, ts, y0):
        """Default to trajectory generation when called directly."""
        return self.generate_trajectory(ts, y0)
