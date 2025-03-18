import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline as RS


def load_data(input_opt, option=None):
    
    if input_opt == "custom":
        print("\nYou selected custom data.\n")
        if option == "move_towards":
            seg_num_int = 0
        elif option == "move_away":
            seg_num_int = 1
        input_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "custom_data",
            "smoothing_window_21_quat",
        )
        # List all eef trajectory files for first segment
        traj_files = [
            f
            for f in os.listdir(input_path)
            if f.endswith("_eef_traj.npy") and "_seg_00_" in f
        ]
        L = len(traj_files)

        x = []
        x_dot = []

        # Print number of segments and timesteps per demo
        for demo_idx in range(L):
            demo_num = str(demo_idx).zfill(2)
            seg_files = sorted(
                [
                    f
                    for f in os.listdir(input_path)
                    if f"demo_{demo_num}_seg_" in f and f.endswith("_eef_traj.npy")
                ]
            )
            # print(f"Demo {demo_num} has {len(seg_files)} segments:")
            for seg_file in seg_files:
                seg_num = seg_file[12:14]
                traj = np.load(os.path.join(input_path, seg_file))
                contact_traj = np.load(
                    os.path.join(
                        input_path, f"demo_{demo_num}_seg_{seg_num}_contact_traj.npy"
                    ),
                    allow_pickle=True,
                )
                # print(f"  Segment {seg_num}: {len(traj)} timesteps")
                # print(f"  First contact point: {contact_traj[0]}")

        for l in range(L):
            # Load EEF and handle trajectory data for first segment
            demo_num = str(l).zfill(2)
            # Get segment number from user
            seg_num = str(seg_num_int).zfill(2)
            eef_traj = np.load(
                os.path.join(input_path, f"demo_{demo_num}_seg_{seg_num}_eef_traj.npy")
            )
            handle_traj = np.load(
                os.path.join(
                    input_path, f"demo_{demo_num}_seg_{seg_num}_handle_traj.npy"
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
            handle_pos = handle_traj[:, :3]
            handle_rot = handle_traj[:, 3:]
            # quat xyzw

            handle_rot = np.array([R.from_quat(q).as_matrix() for q in handle_rot])
            # Save initial handle pose
            handle_pos_init = handle_pos[0]
            handle_rot_init = handle_rot[0]

            eef_rot = np.array([R.from_quat(q).as_matrix() for q in eef_traj[:, 3:]])

            # Compute relative position in world frame
            rel_pos_world = eef_pos - handle_pos_init

            # Transform relative positions to initial handle frame
            pos_traj = np.zeros_like(rel_pos_world)

            rot_traj = np.zeros_like(eef_rot)
            for i in range(len(rel_pos_world)):
                # Transform to initial handle frame
                pos_traj[i] = handle_rot_init.T @ rel_pos_world[i]
                rot_traj[i] = handle_rot_init.T @ eef_rot[i]

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
            debug_on = False
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
            if l == 0:
                x = [pos_traj]
                x_dot = [vel_traj]
                quat_traj_all = [quat_traj]
                ang_vel_traj_all = [ang_vel_traj]
            else:
                x.append(pos_traj)
                x_dot.append(vel_traj)
                quat_traj_all.append(quat_traj)
                ang_vel_traj_all.append(ang_vel_traj)
        return x, x_dot, quat_traj_all, ang_vel_traj_all

    raise Exception("load_data not implemented")