import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def plot_pose(
    start_pose, inbetween_pose, target_pose, frame_idx, skeleton, save_dir, pred=True
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    parent_idx = []
    for joint_name in skeleton.skeleton.keys():
        parent = skeleton.skeleton[joint_name]["parent"]
        if parent is not None:  # If joint_name is root
            parent_idx.append(skeleton.joints.index(parent))
        else:
            parent_idx.append(-1)

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

    title = f"Generated: {frame_idx}" if pred else f"Ground Truth {frame_idx}"
    plt.title(title)
    prefix = "pred_" if pred else "gt_"
    plot_tmp_dir = os.path.join(save_dir, "results", "tmp")
    pathlib.Path(plot_tmp_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(plot_tmp_dir, prefix + str(frame_idx) + ".png"), dpi=200)
    plt.close()
