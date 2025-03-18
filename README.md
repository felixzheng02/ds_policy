# DS Policy

A dynamical system policy for robot control, providing stable and adaptive motion generation using neural ordinary differential equations (Neural ODEs) and control Lyapunov functions (CLFs).

## Features

- Generate robot control actions using dynamical systems
- Train models on demonstration data
- Support for both position and orientation control
- Stability guarantees through CLF constraints
- Ability to switch between reference trajectories at runtime

## Installation

### From Source

```bash
# Create conda environment
conda create -n <env_name> python=3.10.16
conda activate <env_name>

# Clone the repository
git clone https://github.com/felixzheng02/ds_policy
cd ds_policy

# Install the package
pip install -e . --no-cache-dir
```

## Quick Start

```python
import numpy as np
from ds_policy import DSPolicy

# Sample demonstration data
x = [np.random.rand(100, 3)]  # Position trajectories
x_dot = [np.random.rand(100, 3)]  # Velocity trajectories
quat = [np.random.rand(100, 4)]  # Quaternion trajectories
omega = [np.random.rand(100, 3)]  # Angular velocity trajectories

# Configure the model
model_config = {
    'pos_model': {
        # Either use average velocities
        # 'use_avg': True,
        # Or specify load_path to load an existing model
        'load_path': f"ds_policy/models/mlp_width128_depth3.pt",
        # Or provide training parameters if model doesn't exist yet
        # 'width': 128,
        # 'depth': 3,
        # 'save_path': f"ds_policy/models/mlp_width128_depth3.pt",
        # 'batch_size': 100,
        # 'device': "cpu",  # Can be "cpu", "cuda", or "mps"
        # 'lr_strategy': (1e-3, 1e-4, 1e-5),
        # 'epoch_strategy': (10, 10, 10),
        # 'length_strategy': (0.4, 0.7, 1),
        # 'plot': True,
        # 'print_every': 10
    },
    'quat_model': {
        # Either use load_path
        # 'load_path': f"ds_policy/models/quat_model.json",
        # Or specify training parameters
        'save_path': f"ds_policy/models/quat_model.json",
        'k_init': 10
    }
}

# Create a DS policy
policy = DSPolicy(x, x_dot, quat, omega, model_config)

# Generate a control action
state = np.concatenate([np.random.rand(3), np.random.rand(4)])  # [position, quaternion]
action = policy.get_action(state)  # Returns [linear velocity, angular velocity]
```

## Documentation

For more detailed usage and API documentation, please refer to the class docstrings.

## License

[MIT License](LICENSE)