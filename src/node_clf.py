import time

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax
from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate

import jax.tree_util as jtu
import functools as ft

import cvxpy as cp
import plotly.graph_objects as go
import numpy as np
from matplotlib.lines import Line2D


class Func_rot(eqx.Module):
    """Neural ODE function for 3D translational dynamics.
    
    This class defines a neural network that learns to predict velocities from positions.
    The model takes in a 3D position vector and outputs 3D velocity.
    
    Structure:
    - Input: 3D state vector [position (3D)]
    - MLP: Maps state to velocities 
    - Output: 3D vector field [translational velocity (3D)]
    """
    mlp: eqx.nn.MLP  # Neural network to learn the dynamics

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        """Initialize the model.
        
        Args:
            data_size: Dimension of input state (3 for position)
            width_size: Width of hidden layers
            depth: Number of hidden layers
            key: Random key for initialization
        """
        super().__init__(**kwargs)
        initializer = jnn.initializers.orthogonal()
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )
        # Initialize weights using orthogonal initialization
        model_key = key
        key_weights = jrandom.split(model_key, depth+1)

        for i in range(depth+1):
            where = lambda m: m.layers[i].weight
            shape = self.mlp.layers[i].weight.shape
            self.mlp = eqx.tree_at(where, self.mlp, replace=initializer(key_weights[i], shape, dtype=jnp.float32))

    @eqx.filter_jit
    def predict_velocity(self, x):
        """Predict velocity directly from position.
        
        Args:
            x: Current 3D position(s). Can be a single vector (3,), batch (N, 3), 
               or multiple trajectories (L, M, 3) where L is number of trajectories
               and M is points per trajectory
            
        Returns:
            3D velocity vector(s) with same shape as input
        """
        # Convert to jnp array if not already
        x = jnp.asarray(x)
        
        # Handle different input shapes
        if x.ndim == 1:
            # Single position vector (3,) -> add batch dimension (1, 3)
            pred = self.mlp(x)
            return pred.reshape(-1)  # Back to (3,)
        elif x.ndim == 2:
            # Batch of positions (N, 3) -> use as is
            return jax.vmap(self.mlp)(x)
        elif x.ndim == 3:
            # Multiple trajectories (L, M, 3) -> reshape to (L*M, 3), apply model, then reshape back
            L, M, N = x.shape
            x_flat = x.reshape(-1, N)
            pred_flat = jax.vmap(self.mlp)(x_flat)
            return pred_flat.reshape(L, M, N)
        else:
            raise ValueError(f"Input must be 1D, 2D or 3D array, got shape {x.shape}")

    @eqx.filter_jit
    def __call__(self, t, y, args):
        """Compute the vector field at current state (used for ODE solving).
        
        Args:
            t: Current time (not used)
            y: Current 3D position
            args: Additional arguments (not used)
            
        Returns:
            3D velocity vector
        """
        return self.predict_velocity(y)


class NeuralODE_rot(eqx.Module):
    """
    Neural ODE model for 3D trajectories.

    This class wraps the velocity prediction network and provides methods for both:
    1. Direct velocity prediction
    2. Trajectory generation through ODE integration

    Parameters:
        data_size: Dimension of input state (3 for position)
        width_size: Width of hidden layers in Func network
        depth: Number of hidden layers in Func network
        key: JAX random key for initialization
    """
    func_rot: Func_rot

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func_rot = Func_rot(data_size, width_size, depth, key=key)

    @eqx.filter_jit
    def predict_velocity(self, x):
        """Predict velocity directly from position without ODE integration."""
        return self.func_rot.predict_velocity(x)

    @eqx.filter_jit
    def generate_trajectory(self, ts, y0):
        """
        Generate a trajectory by solving the ODE.
        
        Args:
            ts: Time points to evaluate trajectory at
            y0: Initial 3D position
            
        Returns:
            Position trajectory evaluated at specified timepoints
        """
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func_rot),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=4000,
        )
        return solution.ys

    def __call__(self, ts, y0):
        """Default to trajectory generation when called directly."""
        return self.generate_trajectory(ts, y0)