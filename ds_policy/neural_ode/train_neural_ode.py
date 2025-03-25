import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from .neural_ode import NeuralODE


def train(
    x: list[np.ndarray],
    y: list[np.ndarray],
    save_path: str = None,
    device: str = None,
    batch_size: int = 32,
    lr_strategy: tuple = (1e-3, 1e-3, 1e-3),
    epoch_strategy: tuple = (100, 100, 100),
    length_strategy: tuple = (0.4, 0.7, 1),  # TODO: not used
    width_size: int = 64,
    depth: int = 3,
    plot: bool = False,
    print_every: int = 100,
):
    """
    Train the NeuralODE model.

    Args:
        x: Training data. Each array can be of size (N, 3) or (N, 7).
        x_dot: Training velocity data. Each array can be of size (N, 3) or (N, 6).
        save_path: Path to save the trained model
        device: Device to run the training on ("cpu", "cuda", "mps")
        batch_size: Number of samples to process in each training batch
        lr_strategy: Learning rate in each phase
        epoch_strategy: Number of epochs in each phase
        length_strategy: Fraction of the trajectories available for training in each phase
        width_size: Width of hidden layers
        depth: Number of hidden layers
        plot: Whether to plot the training progress
        print_every: Print the training progress every print_every epochs
    """
    # Set default device if none provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else 
                              "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    input_size = x[0].shape[-1]
    output_size = y[0].shape[-1]
    if save_path is None:
        save_path = f"DS-Policy/models/mlp_width{width_size}_depth{depth}_input{input_size}_output{output_size}_unnamed.pt"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    x_flat = np.concatenate(x, axis=0)
    y_flat = np.concatenate(y, axis=0)

    x_flat_tensor = torch.tensor(x_flat, dtype=torch.float32).to(device)
    y_flat_tensor = torch.tensor(y_flat, dtype=torch.float32).to(device)

    # Create dataset and dataloader for batch training
    dataset = TensorDataset(x_flat_tensor, y_flat_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = NeuralODE(input_size, output_size, width_size, depth).to(device)

    # Training loop with curriculum learning
    for phase, (lr, epochs) in enumerate(zip(lr_strategy, epoch_strategy)):
        print(f"\nPhase {phase+1}: Learning rate = {lr}, Epochs = {epochs}")

        # Setup optimizer with learning rate scheduler
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        # Track losses for plotting
        losses = []

        for epoch in range(epochs):
            start_time = time.time()

            # Reset gradients for each batch iteration
            running_loss = 0.0
            num_batches = 0

            # Iterate through mini-batches
            for batch_x, batch_x_dot in dataloader:
                optimizer.zero_grad()

                # Forward pass
                v_pred = model.forward(batch_x)

                # Compute loss
                loss = loss_fn(v_pred, batch_x_dot)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

            # Calculate average loss across all batches
            avg_loss = running_loss / num_batches
            losses.append(avg_loss)

            end_time = time.time()

            if (epoch % print_every) == 0 or epoch == epochs - 1:
                print(
                    f"Epoch: {epoch}, Loss: {avg_loss:.6f}, Computation time: {end_time - start_time:.4f}s"
                )

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    if plot:
        # Plot training loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid(True)
        plt.show()
        plt.close()

    return model


