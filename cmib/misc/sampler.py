import argparse
import os
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import LabelEncoder

from cmib.data.lafan1_dataset import LAFAN1Dataset
from cmib.data.utils import write_json
from cmib.lafan1.utils import quat_ik
from cmib.model.network import TransformerModel
from cmib.model.preprocess import (lerp_input_repr, replace_constant,
                                  slerp_input_repr, vectorize_representation)
from cmib.model.skeleton import (Skeleton, sk_joints_to_remove, sk_offsets, joint_names,
                                sk_parents)
from cmib.vis.pose import plot_pose_with_stop


def test(opt, device):

    save_dir = Path(os.path.join('runs', 'train', opt.exp_name))
    wdir = save_dir / 'weights'
    weights = os.listdir(wdir)
    weights_paths = [wdir / weight for weight in weights]
    latest_weight = max(weights_paths , key = os.path.getctime)
    ckpt = torch.load(latest_weight, map_location=device)
    print(f"Loaded weight: {latest_weight}")

    # Load Skeleton
    skeleton_mocap = Skeleton(offsets=sk_offsets, parents=sk_parents, device=device)
    skeleton_mocap.remove_joints(sk_joints_to_remove)

    # Load LAFAN Dataset
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    lafan_dataset = LAFAN1Dataset(lafan_path=opt.data_path, processed_data_dir=opt.processed_data_dir, train=False, device=device)
    total_data = lafan_dataset.data['global_pos'].shape[0]
    
    # Replace with noise to In-betweening Frames
    from_idx, target_idx = ckpt['from_idx'], ckpt['target_idx'] # default: 9-40, max: 48
    horizon = ckpt['horizon']
    print(f"HORIZON: {horizon}")

    test_idx = []
    for i in range(total_data):
        test_idx.append(i)

    # Compare Input data, Prediction, GT
    save_path = os.path.join(opt.save_path, 'sampler')
    for i in range(len(test_idx)):
        Path(save_path).mkdir(parents=True, exist_ok=True)

        start_pose =  lafan_dataset.data['global_pos'][test_idx[i], from_idx]
        target_pose = lafan_dataset.data['global_pos'][test_idx[i], target_idx]
        gt_stopover_pose = lafan_dataset.data['global_pos'][test_idx[i], from_idx]

        gt_img_path = os.path.join(save_path)
        plot_pose_with_stop(start_pose, target_pose, target_pose, gt_stopover_pose, i, skeleton_mocap, save_dir=gt_img_path, prefix='gt')
        print(f"ID {test_idx[i]}: completed.")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--exp_name', default='slerp_40', help='experiment name')
    parser.add_argument('--data_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH', help='BVH dataset path')
    parser.add_argument('--skeleton_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh', help='path to reference skeleton')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_original/', help='path to save pickled processed data')
    parser.add_argument('--save_path', type=str, default='runs/test', help='path to save model')
    parser.add_argument('--motion_type', type=str, default='jumps', help='motion type')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device("cpu")
    test(opt, device)
