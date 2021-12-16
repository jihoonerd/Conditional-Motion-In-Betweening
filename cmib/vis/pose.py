import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def project_root_position(position_arr: np.array, file_name: str):
    """
    Take batch of root arrays and porject it on 2D plane

    N: samples
    L: trajectory length
    J: joints

    position_arr: [N,L,J,3]
    """

    root_joints = position_arr[:, :, 0]

    x_pos = root_joints[:, :, 0]
    y_pos = root_joints[:, :, 2]

    fig = plt.figure()

    for i in range(x_pos.shape[1]):

        if i == 0:
            plt.scatter(x_pos[:, i], y_pos[:, i], c="b")
        elif i == x_pos.shape[1] - 1:
            plt.scatter(x_pos[:, i], y_pos[:, i], c="r")
        else:
            plt.scatter(x_pos[:, i], y_pos[:, i], c="k", marker="*", s=1)

    plt.title(f"Root Position: {file_name}")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.xlim((-300, 300))
    plt.ylim((-300, 300))
    plt.grid()
    plt.savefig(f"{file_name}.png", dpi=200)


def plot_single_pose(
    pose,
    frame_idx,
    skeleton,
    save_dir,
    prefix,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    parent_idx = skeleton.parents()

    for i, p in enumerate(parent_idx):
        if i > 0:
            ax.plot(
                [pose[i, 0], pose[p, 0]],
                [pose[i, 2], pose[p, 2]],
                [pose[i, 1], pose[p, 1]],
                c="k",
            )

    x_min = pose[:, 0].min()
    x_max = pose[:, 0].max()

    y_min = pose[:, 1].min()
    y_max = pose[:, 1].max()

    z_min = pose[:, 2].min()
    z_max = pose[:, 2].max()

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("$X$ Axis")

    ax.set_ylim(z_min, z_max)
    ax.set_ylabel("$Y$ Axis")

    ax.set_zlim(y_min, y_max)
    ax.set_zlabel("$Z$ Axis")

    plt.draw()

    title = f"{prefix}: {frame_idx}"
    plt.title(title)
    prefix = prefix
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, prefix + str(frame_idx) + ".png"), dpi=60)
    plt.close()


def plot_pose(
    start_pose,
    inbetween_pose,
    target_pose,
    frame_idx,
    skeleton,
    save_dir,
    prefix,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    parent_idx = skeleton.parents()

    for i, p in enumerate(parent_idx):
        if i > 0:
            ax.plot(
                [start_pose[i, 0], start_pose[p, 0]],
                [start_pose[i, 2], start_pose[p, 2]],
                [start_pose[i, 1], start_pose[p, 1]],
                c="b",
            )
            ax.plot(
                [inbetween_pose[i, 0], inbetween_pose[p, 0]],
                [inbetween_pose[i, 2], inbetween_pose[p, 2]],
                [inbetween_pose[i, 1], inbetween_pose[p, 1]],
                c="k",
            )
            ax.plot(
                [target_pose[i, 0], target_pose[p, 0]],
                [target_pose[i, 2], target_pose[p, 2]],
                [target_pose[i, 1], target_pose[p, 1]],
                c="r",
            )

    x_min = np.min(
        [start_pose[:, 0].min(), inbetween_pose[:, 0].min(), target_pose[:, 0].min()]
    )
    x_max = np.max(
        [start_pose[:, 0].max(), inbetween_pose[:, 0].max(), target_pose[:, 0].max()]
    )

    y_min = np.min(
        [start_pose[:, 1].min(), inbetween_pose[:, 1].min(), target_pose[:, 1].min()]
    )
    y_max = np.max(
        [start_pose[:, 1].max(), inbetween_pose[:, 1].max(), target_pose[:, 1].max()]
    )

    z_min = np.min(
        [start_pose[:, 2].min(), inbetween_pose[:, 2].min(), target_pose[:, 2].min()]
    )
    z_max = np.max(
        [start_pose[:, 2].max(), inbetween_pose[:, 2].max(), target_pose[:, 2].max()]
    )

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("$X$ Axis")

    ax.set_ylim(z_min, z_max)
    ax.set_ylabel("$Y$ Axis")

    ax.set_zlim(y_min, y_max)
    ax.set_zlabel("$Z$ Axis")

    plt.draw()

    title = f"{prefix}: {frame_idx}"
    plt.title(title)
    prefix = prefix
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, prefix + str(frame_idx) + ".png"), dpi=60)
    plt.close()


def plot_pose_with_stop(
    start_pose,
    inbetween_pose,
    target_pose,
    stopover,
    frame_idx,
    skeleton,
    save_dir,
    prefix,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    parent_idx = skeleton.parents()

    for i, p in enumerate(parent_idx):
        if i > 0:
            ax.plot(
                [start_pose[i, 0], start_pose[p, 0]],
                [start_pose[i, 2], start_pose[p, 2]],
                [start_pose[i, 1], start_pose[p, 1]],
                c="b",
            )
            ax.plot(
                [inbetween_pose[i, 0], inbetween_pose[p, 0]],
                [inbetween_pose[i, 2], inbetween_pose[p, 2]],
                [inbetween_pose[i, 1], inbetween_pose[p, 1]],
                c="k",
            )
            ax.plot(
                [target_pose[i, 0], target_pose[p, 0]],
                [target_pose[i, 2], target_pose[p, 2]],
                [target_pose[i, 1], target_pose[p, 1]],
                c="r",
            )

            ax.plot(
                [stopover[i, 0], stopover[p, 0]],
                [stopover[i, 2], stopover[p, 2]],
                [stopover[i, 1], stopover[p, 1]],
                c="indigo",
            )

    x_min = np.min(
        [start_pose[:, 0].min(), inbetween_pose[:, 0].min(), target_pose[:, 0].min()]
    )
    x_max = np.max(
        [start_pose[:, 0].max(), inbetween_pose[:, 0].max(), target_pose[:, 0].max()]
    )

    y_min = np.min(
        [start_pose[:, 1].min(), inbetween_pose[:, 1].min(), target_pose[:, 1].min()]
    )
    y_max = np.max(
        [start_pose[:, 1].max(), inbetween_pose[:, 1].max(), target_pose[:, 1].max()]
    )

    z_min = np.min(
        [start_pose[:, 2].min(), inbetween_pose[:, 2].min(), target_pose[:, 2].min()]
    )
    z_max = np.max(
        [start_pose[:, 2].max(), inbetween_pose[:, 2].max(), target_pose[:, 2].max()]
    )

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("$X$ Axis")

    ax.set_ylim(z_min, z_max)
    ax.set_ylabel("$Y$ Axis")

    ax.set_zlim(y_min, y_max)
    ax.set_zlabel("$Z$ Axis")

    plt.draw()

    title = f"{prefix}: {frame_idx}"
    plt.title(title)
    prefix = prefix
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, prefix + str(frame_idx) + ".png"), dpi=60)
    plt.close()
