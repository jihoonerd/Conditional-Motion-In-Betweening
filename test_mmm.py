import argparse
import os
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.model.network import TransformerModel
from rmi.model.preprocess import (lerp_input_repr, lerp_reshaped,
                                  vectorize_representation)
from rmi.model.skeleton import (Skeleton, sk_joints_to_remove, sk_offsets,
                                sk_parents)
from rmi.vis.pose import plot_pose_with_stop


def test(opt, device):

    save_dir = Path(os.path.join('runs', 'train', opt.exp_name))
    wdir = save_dir / 'weights'
    weights = os.listdir(wdir)
    weights_paths = [wdir / weight for weight in weights]
    latest_weight = max(weights_paths , key = os.path.getctime)
    ckpt = torch.load(latest_weight, map_location=device)
    print(f"Loaded weight: {latest_weight}")

    if ckpt['preserve_link_train']:
        print("Link Preserving: Training")
    else:
        print("Link Prserving: Post Processing")

    # Load Skeleton
    skeleton_mocap = Skeleton(offsets=sk_offsets, parents=sk_parents, device=device)
    skeleton_mocap.remove_joints(sk_joints_to_remove)

    # Load LAFAN Dataset
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    lafan_dataset = LAFAN1Dataset(lafan_path=opt.data_path, processed_data_dir=opt.processed_data_dir, train=False, target_action=[''], device=device)
    
    # Replace with noise to In-betweening Frames
    from_idx, target_idx = ckpt['from_idx'], ckpt['target_idx'] # default: 9-40, max: 48
    horizon = ckpt['horizon']
    print(f"HORIZON: {horizon}")

    test_idx = []
    for i in range(1, 20):
        test_idx.append(i * 200)

    # Extract dimension from processed data
    pos_dim = lafan_dataset.num_joints * 3
    rot_dim = lafan_dataset.num_joints * 4
    repr_dim = pos_dim + rot_dim

    root_pos = torch.Tensor(lafan_dataset.data['root_p'][:, from_idx:target_idx+1]).to(device)
    local_q = torch.Tensor(lafan_dataset.data['local_q'][:, from_idx:target_idx+1]).to(device)
    local_q_normalized = nn.functional.normalize(local_q, p=2.0, dim=-1)

    # Replace testing inputs
    fixed = 18
    num_clue = 1

    global_pos, global_q = skeleton_mocap.forward_kinematics_with_rotation(local_q_normalized, root_pos)

    global_pos[:,fixed] += torch.Tensor([0,35,0]).expand(global_pos.size(0),lafan_dataset.num_joints,3)

    if ckpt['preserve_link_train']:
        print("USE BONE LENGTH NORMALIZATION...")
        global_pos = skeleton_mocap.convert_to_unit_offset_mat(global_pos)
        global_pose_vec_pos = global_pos.reshape(global_pos.size(0), global_pos.size(1), -1).contiguous()
        global_pose_vec_rot = global_q.reshape(global_q.size(0), global_q.size(1), -1).contiguous()
        global_pose_vec_gt = torch.cat([global_pose_vec_pos, global_pose_vec_rot], dim=2)
        global_pose_vec_input = global_pose_vec_gt.clone().detach()

        root_pos = global_pose_vec_input[:,:,:3]
        link_vec = global_pose_vec_input[:,:,3:pos_dim]
        rot_vec = global_pose_vec_input[:,:,pos_dim:]
        root_lerped = lerp_input_repr(root_pos, fixed)
        link_lerped = lerp_reshaped(link_vec, fixed, 21)
        rot_lerped = lerp_reshaped(rot_vec, fixed, 22)
        pose_interpolated_input = torch.cat([root_lerped, link_lerped, rot_lerped], dim=2)
        input_pos = skeleton_mocap.convert_to_global_pos(pose_interpolated_input[:,:,:pos_dim])

    else:
        global_pose_vec_gt = vectorize_representation(global_pos, global_q)
        global_pose_vec_input = global_pose_vec_gt.clone().detach()
        pose_interpolated_input = lerp_input_repr(global_pose_vec_input, fixed)
        input_pos = pose_interpolated_input[:,:,:pos_dim]
    

    infilling_code = np.zeros((1, horizon))
    infilling_code[0, 1:fixed] = 1
    infilling_code[0, fixed+1:-1] = 1
    infilling_code = torch.tensor(infilling_code, dtype=torch.int, device=device)

    pose_vectorized_input = pose_interpolated_input.permute(1,0,2)

    src_mask = torch.zeros((horizon, horizon), device=device).type(torch.bool)
    src_mask = src_mask.to(device)

    model = TransformerModel(seq_len=ckpt['horizon'], d_model=ckpt['d_model'], nhead=ckpt['nhead'], d_hid=ckpt['d_hid'], nlayers=ckpt['nlayers'], dropout=0.0, out_dim=repr_dim)
    model.load_state_dict(ckpt['transformer_encoder_state_dict'])
    model.eval()

    output = model(pose_vectorized_input, src_mask, infilling_code)

    if ckpt['preserve_link_train']:
        pred_global_pos = output[:,:,:pos_dim].permute(1,0,2)
        pred_global_pos = skeleton_mocap.convert_to_global_pos(pred_global_pos)

        clue = global_pos.clone().detach().reshape(global_pos.size(0), global_pos.size(1), -1)
        clue = skeleton_mocap.convert_to_global_pos(clue)

    else:
        pred_global_pos = output[:,:,:pos_dim].permute(1,0,2).reshape(4474,32,22,3)
        global_pos_unit_vec = skeleton_mocap.convert_to_unit_offset_mat(pred_global_pos)
        pred_global_pos = skeleton_mocap.convert_to_global_pos(global_pos_unit_vec)
        clue = global_pos.clone().detach()
        

    # Compare Input data, Prediction, GT
    for i in range(len(test_idx)):
        save_path = os.path.join(opt.save_path, 'test_' + f'{test_idx[i]}')
        Path(save_path).mkdir(parents=True, exist_ok=True)

        start_pose =  lafan_dataset.data['global_pos'][test_idx[i], from_idx]
        target_pose = lafan_dataset.data['global_pos'][test_idx[i], target_idx]
        stopover_pose = clue[test_idx[i],fixed]
        gt_stopover_pose = lafan_dataset.data['global_pos'][test_idx[i], from_idx + fixed]

        img_aggr_list = []
        for t in range(horizon):
            
            input_img_path = os.path.join(save_path, 'input')
            plot_pose_with_stop(start_pose, input_pos[test_idx[i],t].reshape(lafan_dataset.num_joints, 3).detach().numpy(), target_pose, stopover_pose, t, skeleton_mocap, save_dir=input_img_path, prefix='input')
            pred_img_path = os.path.join(save_path, 'pred_img')
            plot_pose_with_stop(start_pose, pred_global_pos[test_idx[i],t].reshape(lafan_dataset.num_joints, 3).detach().numpy(), target_pose, stopover_pose, t, skeleton_mocap, save_dir=pred_img_path, prefix='pred')
            gt_img_path = os.path.join(save_path, 'gt_img')
            plot_pose_with_stop(start_pose, lafan_dataset.data['global_pos'][test_idx[i], t+from_idx], target_pose, gt_stopover_pose, t, skeleton_mocap, save_dir=gt_img_path, prefix='gt')

            input_img = Image.open(os.path.join(input_img_path, 'input'+str(t)+'.png'), 'r')
            pred_img = Image.open(os.path.join(pred_img_path, 'pred'+str(t)+'.png'), 'r')
            gt_img = Image.open(os.path.join(gt_img_path, 'gt'+str(t)+'.png'), 'r')
            
            img_aggr_list.append(np.concatenate([input_img, pred_img, gt_img.resize(pred_img.size)], 1))

        # Save images
        gif_path = os.path.join(save_path, f'img_{test_idx[i]}.gif')
        imageio.mimsave(gif_path, img_aggr_list, duration=0.1)
        print(f"ID {test_idx[i]}: test completed.")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--exp_name', default='Link-Preserving_wQUAT', help='experiment name')
    parser.add_argument('--data_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH', help='BVH dataset path')
    parser.add_argument('--skeleton_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh', help='path to reference skeleton')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_original/', help='path to save pickled processed data')
    parser.add_argument('--save_path', type=str, default='runs/test', help='path to save model')
    parser.add_argument('--motion_type', type=str, default='walk', help='motion type')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device("cpu")
    test(opt, device)
