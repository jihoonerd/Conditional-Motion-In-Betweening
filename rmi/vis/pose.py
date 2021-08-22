import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def project_root_position(position_arr: np.array, file_name: str):
    """
    Take batch of root arrays and porject it on 2D plane

    N: samples
    L: trajectory length
    J: joints

    position_arr: [N,L,J,3]
    """

    root_joints = position_arr[:, :, 0]

    x_pos = root_joints[:,:,0]
    y_pos = root_joints[:,:,2]

    fig = plt.figure()

    for i in range(x_pos.shape[1]):
        
        if i == 0:
            plt.scatter(x_pos[:,i], y_pos[:,i], c='b')
        elif i == x_pos.shape[1] - 1:
            plt.scatter(x_pos[:,i], y_pos[:,i], c='r')
        else:
            plt.scatter(x_pos[:,i], y_pos[:,i], c='k', marker='*', s=1)

    plt.title(f"Root Position: {file_name}")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.xlim((-300,300))
    plt.ylim((-300,300))
    plt.grid()
    plt.savefig(f"{file_name}.png", dpi=200)





    

def plot_pose(
    start_pose, inbetween_pose, target_pose, frame_idx, skeleton, save_dir, prefix,
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

    title = f"{prefix}: {frame_idx}"
    plt.title(title)
    prefix = prefix
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, prefix + str(frame_idx) + ".png"), dpi=100)
    plt.close()

def plot_pose_compare2(
    start_pose, inbetween_pose1, inbetween_pose2, target_pose, frame_idx, skeleton, save_dir, pred=True
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
                [inbetween_pose1[i, 0], inbetween_pose1[p, 0]],
                [inbetween_pose1[i, 2], inbetween_pose1[p, 2]],
                [inbetween_pose1[i, 1], inbetween_pose1[p, 1]],
                c="k",
            )
            ax.plot(
                [inbetween_pose2[i, 0], inbetween_pose2[p, 0]],
                [inbetween_pose2[i, 2], inbetween_pose2[p, 2]],
                [inbetween_pose2[i, 1], inbetween_pose2[p, 1]],
                c="g",
            )

            ax.plot(
                [target_pose[i, 0], target_pose[p, 0]],
                [target_pose[i, 2], target_pose[p, 2]],
                [target_pose[i, 1], target_pose[p, 1]],
                c="r",
            )

    x_min = np.min(
        [start_pose[:, 0].min(), inbetween_pose1[:, 0].min(), inbetween_pose2[:, 0].min(), target_pose[:, 0].min()]
    )
    x_max = np.max(
        [start_pose[:, 0].max(), inbetween_pose1[:, 0].max(), inbetween_pose2[:, 0].max(), target_pose[:, 0].max()]
    )

    y_min = np.min(
        [start_pose[:, 1].min(), inbetween_pose1[:, 1].min(), inbetween_pose2[:, 1].min(), target_pose[:, 1].min()]
    )
    y_max = np.max(
        [start_pose[:, 1].max(), inbetween_pose1[:, 1].max(), inbetween_pose2[:, 1].max(), target_pose[:, 1].max()]
    )

    z_min = np.min(
        [start_pose[:, 2].min(), inbetween_pose1[:, 2].min(), inbetween_pose2[:, 2].min(), target_pose[:, 2].min()]
    )
    z_max = np.max(
        [start_pose[:, 2].max(), inbetween_pose1[:, 2].max(), inbetween_pose2[:, 2].max(), target_pose[:, 2].max()]
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


def plot_pose_compare3(
    start_pose, inbetween_pose1, inbetween_pose2, inbetween_pose3, target_pose, frame_idx, skeleton, save_dir, pred=True
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
                [inbetween_pose1[i, 0], inbetween_pose1[p, 0]],
                [inbetween_pose1[i, 2], inbetween_pose1[p, 2]],
                [inbetween_pose1[i, 1], inbetween_pose1[p, 1]],
                c="k",
            )
            ax.plot(
                [inbetween_pose2[i, 0], inbetween_pose2[p, 0]],
                [inbetween_pose2[i, 2], inbetween_pose2[p, 2]],
                [inbetween_pose2[i, 1], inbetween_pose2[p, 1]],
                c="g",
            )
            ax.plot(
                [inbetween_pose3[i, 0], inbetween_pose3[p, 0]],
                [inbetween_pose3[i, 2], inbetween_pose3[p, 2]],
                [inbetween_pose3[i, 1], inbetween_pose3[p, 1]],
                c="c",
            )
            ax.plot(
                [target_pose[i, 0], target_pose[p, 0]],
                [target_pose[i, 2], target_pose[p, 2]],
                [target_pose[i, 1], target_pose[p, 1]],
                c="r",
            )

    x_min = np.min(
        [start_pose[:, 0].min(), inbetween_pose1[:, 0].min(), inbetween_pose2[:, 0].min(), inbetween_pose3[:, 0].min(), target_pose[:, 0].min()]
    )
    x_max = np.max(
        [start_pose[:, 0].max(), inbetween_pose1[:, 0].max(), inbetween_pose2[:, 0].max(), inbetween_pose3[:, 0].max(), target_pose[:, 0].max()]
    )

    y_min = np.min(
        [start_pose[:, 1].min(), inbetween_pose1[:, 1].min(), inbetween_pose2[:, 1].min(), inbetween_pose3[:, 1].min(), target_pose[:, 1].min()]
    )
    y_max = np.max(
        [start_pose[:, 1].max(), inbetween_pose1[:, 1].max(), inbetween_pose2[:, 1].max(), inbetween_pose3[:, 1].max(), target_pose[:, 1].max()]
    )

    z_min = np.min(
        [start_pose[:, 2].min(), inbetween_pose1[:, 2].min(), inbetween_pose2[:, 2].min(), inbetween_pose3[:, 2].min(), target_pose[:, 2].min()]
    )
    z_max = np.max(
        [start_pose[:, 2].max(), inbetween_pose1[:, 2].max(), inbetween_pose2[:, 2].max(), inbetween_pose3[:, 2].max(), target_pose[:, 2].max()]
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


def plot_pose_compare4(
    start_pose, inbetween_pose1, inbetween_pose2, inbetween_pose3, inbetween_pose4, target_pose, frame_idx, skeleton, save_dir, pred=True
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
                [inbetween_pose1[i, 0], inbetween_pose1[p, 0]],
                [inbetween_pose1[i, 2], inbetween_pose1[p, 2]],
                [inbetween_pose1[i, 1], inbetween_pose1[p, 1]],
                c="k",
            )
            ax.plot(
                [inbetween_pose2[i, 0], inbetween_pose2[p, 0]],
                [inbetween_pose2[i, 2], inbetween_pose2[p, 2]],
                [inbetween_pose2[i, 1], inbetween_pose2[p, 1]],
                c="g",
            )
            ax.plot(
                [inbetween_pose3[i, 0], inbetween_pose3[p, 0]],
                [inbetween_pose3[i, 2], inbetween_pose3[p, 2]],
                [inbetween_pose3[i, 1], inbetween_pose3[p, 1]],
                c="c",
            )
            ax.plot(
                [inbetween_pose4[i, 0], inbetween_pose4[p, 0]],
                [inbetween_pose4[i, 2], inbetween_pose4[p, 2]],
                [inbetween_pose4[i, 1], inbetween_pose4[p, 1]],
                c="m",
            )
            ax.plot(
                [target_pose[i, 0], target_pose[p, 0]],
                [target_pose[i, 2], target_pose[p, 2]],
                [target_pose[i, 1], target_pose[p, 1]],
                c="r",
            )

    x_min = np.min(
        [start_pose[:, 0].min(), inbetween_pose1[:, 0].min(), inbetween_pose2[:, 0].min(), inbetween_pose3[:, 0].min(), inbetween_pose4[:, 0].min(), target_pose[:, 0].min()]
    )
    x_max = np.max(
        [start_pose[:, 0].max(), inbetween_pose1[:, 0].max(), inbetween_pose2[:, 0].max(), inbetween_pose3[:, 0].max(), inbetween_pose4[:, 0].max(), target_pose[:, 0].max()]
    )
    y_min = np.min(
        [start_pose[:, 1].min(), inbetween_pose1[:, 1].min(), inbetween_pose2[:, 1].min(), inbetween_pose3[:, 1].min(), inbetween_pose4[:, 1].min(), target_pose[:, 1].min()]
    )
    y_max = np.max(
        [start_pose[:, 1].max(), inbetween_pose1[:, 1].max(), inbetween_pose2[:, 1].max(), inbetween_pose3[:, 1].max(), inbetween_pose4[:, 1].max(), target_pose[:, 1].max()]
    )

    z_min = np.min(
        [start_pose[:, 2].min(), inbetween_pose1[:, 2].min(), inbetween_pose2[:, 2].min(), inbetween_pose3[:, 2].min(), inbetween_pose4[:, 2].min(), target_pose[:, 2].min()]
    )
    z_max = np.max(
        [start_pose[:, 2].max(), inbetween_pose1[:, 2].max(), inbetween_pose2[:, 2].max(), inbetween_pose3[:, 2].max(), inbetween_pose4[:, 2].max(), target_pose[:, 2].max()]
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



def plot_pose_compare5(
    start_pose, inbetween_pose1, inbetween_pose2, inbetween_pose3, inbetween_pose4, inbetween_pose5, target_pose, frame_idx, skeleton, save_dir, pred=True
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
                [inbetween_pose1[i, 0], inbetween_pose1[p, 0]],
                [inbetween_pose1[i, 2], inbetween_pose1[p, 2]],
                [inbetween_pose1[i, 1], inbetween_pose1[p, 1]],
                c="k",
            )
            ax.plot(
                [inbetween_pose2[i, 0], inbetween_pose2[p, 0]],
                [inbetween_pose2[i, 2], inbetween_pose2[p, 2]],
                [inbetween_pose2[i, 1], inbetween_pose2[p, 1]],
                c="g",
            )
            ax.plot(
                [inbetween_pose3[i, 0], inbetween_pose3[p, 0]],
                [inbetween_pose3[i, 2], inbetween_pose3[p, 2]],
                [inbetween_pose3[i, 1], inbetween_pose3[p, 1]],
                c="c",
            )
            ax.plot(
                [inbetween_pose4[i, 0], inbetween_pose4[p, 0]],
                [inbetween_pose4[i, 2], inbetween_pose4[p, 2]],
                [inbetween_pose4[i, 1], inbetween_pose4[p, 1]],
                c="m",
            )
            ax.plot(
                [inbetween_pose5[i, 0], inbetween_pose5[p, 0]],
                [inbetween_pose5[i, 2], inbetween_pose5[p, 2]],
                [inbetween_pose5[i, 1], inbetween_pose5[p, 1]],
                c="y",
            )

            ax.plot(
                [target_pose[i, 0], target_pose[p, 0]],
                [target_pose[i, 2], target_pose[p, 2]],
                [target_pose[i, 1], target_pose[p, 1]],
                c="r",
            )

    x_min = np.min(
        [start_pose[:, 0].min(), inbetween_pose1[:, 0].min(), inbetween_pose2[:, 0].min(), inbetween_pose3[:, 0].min(), inbetween_pose4[:, 0].min(), inbetween_pose5[:,0].min(), target_pose[:, 0].min()]
    )
    x_max = np.max(
        [start_pose[:, 0].max(), inbetween_pose1[:, 0].max(), inbetween_pose2[:, 0].max(), inbetween_pose3[:, 0].max(), inbetween_pose4[:, 0].max(), inbetween_pose5[:,0].max(), target_pose[:, 0].max()]
    )

    y_min = np.min(
        [start_pose[:, 1].min(), inbetween_pose1[:, 1].min(), inbetween_pose2[:, 1].min(), inbetween_pose3[:, 1].min(), inbetween_pose4[:, 1].min(), inbetween_pose5[:,1].min(), target_pose[:, 1].min()]
    )
    y_max = np.max(
        [start_pose[:, 1].max(), inbetween_pose1[:, 1].max(), inbetween_pose2[:, 1].max(), inbetween_pose3[:, 1].max(), inbetween_pose4[:, 1].max(), inbetween_pose5[:, 1].max(), target_pose[:, 1].max()]
    )

    z_min = np.min(
        [start_pose[:, 2].min(), inbetween_pose1[:, 2].min(), inbetween_pose2[:, 2].min(), inbetween_pose3[:, 2].min(), inbetween_pose4[:, 2].min(), inbetween_pose5[:, 2].min(), target_pose[:, 2].min()]
    )
    z_max = np.max(
        [start_pose[:, 2].max(), inbetween_pose1[:, 2].max(), inbetween_pose2[:, 2].max(), inbetween_pose3[:, 2].max(), inbetween_pose4[:, 2].max(), inbetween_pose5[:, 2].max(), target_pose[:, 2].max()]
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
