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
from cmib.model.skeleton import (Skeleton, joint_names, sk_joints_to_remove,
                                 sk_offsets, sk_parents)
from cmib.vis.pose import plot_pose_with_stop


def test(opt, device):

    save_dir = Path(os.path.join('runs', 'train', opt.exp_name))
    wdir = save_dir / 'weights'
    weights = os.listdir(wdir)

    if opt.weight == 'latest':
        weights_paths = [wdir / weight for weight in weights]
        weight_path = max(weights_paths , key = os.path.getctime)
    else:
        weight_path = wdir / ('train-' + opt.weight + '.pt')
    ckpt = torch.load(weight_path, map_location=device)
    print(f"Loaded weight: {weight_path}")


    # Load Skeleton
    skeleton_mocap = Skeleton(offsets=sk_offsets, parents=sk_parents, device=device)
    skeleton_mocap.remove_joints(sk_joints_to_remove)

    # Load LAFAN Dataset
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    lafan_dataset = LAFAN1Dataset(lafan_path=opt.data_path, processed_data_dir=opt.processed_data_dir, train=False, device=device, window=ckpt['horizon'] + 10 - 1)
    total_data = lafan_dataset.data['global_pos'].shape[0]
    
    # Replace with noise to In-betweening Frames
    from_idx, target_idx = ckpt['from_idx'], ckpt['target_idx'] # default: 9-40, max: 48
    horizon = ckpt['horizon']
    print(f"HORIZON: {horizon}")

    test_idx = [950, 1140, 2100]

    # Extract dimension from processed data
    pos_dim = lafan_dataset.num_joints * 3
    rot_dim = lafan_dataset.num_joints * 4
    repr_dim = pos_dim + rot_dim

    root_pos = torch.Tensor(lafan_dataset.data['root_p'][:, from_idx:target_idx+1]).to(device)
    local_q = torch.Tensor(lafan_dataset.data['local_q'][:, from_idx:target_idx+1]).to(device)
    local_q_normalized = nn.functional.normalize(local_q, p=2.0, dim=-1)

    # Replace testing inputs
    fixed = 0
    global_pos, global_q = skeleton_mocap.forward_kinematics_with_rotation(local_q_normalized, root_pos)

    interpolation = ckpt['interpolation']
    print(f"Interpolation Mode: {interpolation}")

    if interpolation == 'constant':
        global_pose_vec_gt = vectorize_representation(global_pos, global_q)
        global_pose_vec_input = global_pose_vec_gt.clone().detach()
        pose_interpolated_input = replace_constant(global_pose_vec_input, fixed)
        input_pos = pose_interpolated_input[:,:,:pos_dim].detach().numpy()

    elif interpolation == 'slerp':
        global_pose_vec_gt = vectorize_representation(global_pos, global_q)
        global_pose_vec_input = global_pose_vec_gt.clone().detach()
        root_vec = global_pose_vec_input[:,:,:pos_dim]
        rot_vec = global_pose_vec_input[:,:,pos_dim:]
        root_lerped = lerp_input_repr(root_vec, fixed)
        rot_slerped = slerp_input_repr(rot_vec, fixed)
        pose_interpolated_input = torch.cat([root_lerped, rot_slerped], dim=2)
        input_pos = pose_interpolated_input[:,:,:pos_dim].detach().numpy()

    else:
        raise ValueError('Invalid interpolation method')
    
    pose_vectorized_input = pose_interpolated_input.permute(1,0,2)

    src_mask = torch.zeros((horizon, horizon), device=device).type(torch.bool)
    src_mask = src_mask.to(device)

    seq_categories = [x[:-1] for x in lafan_dataset.data['seq_names']]

    le = LabelEncoder()
    le.classes_ = np.load(os.path.join(save_dir, 'le_classes_.npy'))

    target_seq = opt.motion_type
    seq_id = np.where(le.classes_==target_seq)[0]
    conditioning_labels = np.expand_dims((np.repeat(seq_id[0], repeats=len(seq_categories))), axis=1)
    conditioning_labels = torch.Tensor(conditioning_labels).type(torch.int64).to(device)

    model = TransformerModel(seq_len=ckpt['horizon'], d_model=ckpt['d_model'], nhead=ckpt['nhead'], d_hid=ckpt['d_hid'], nlayers=ckpt['nlayers'], dropout=0.05, out_dim=repr_dim)
    model.load_state_dict(ckpt['transformer_encoder_state_dict'])
    model.eval()

    output, _ = model(pose_vectorized_input, src_mask, conditioning_labels)

    pred_global_pos = output[1:,:,:pos_dim].permute(1,0,2).reshape(total_data,horizon-1,22,3)
    global_pos_unit_vec = skeleton_mocap.convert_to_unit_offset_mat(pred_global_pos)
    pred_global_pos = skeleton_mocap.convert_to_global_pos(global_pos_unit_vec).detach().numpy()

    pred_global_rot = output[1:,:,pos_dim:].permute(1,0,2).reshape(total_data,horizon-1,22,4)
    pred_global_rot_normalized = nn.functional.normalize(pred_global_rot, p=2.0, dim=3).detach().numpy()

    clue = global_pos.clone().detach()
        
    # Compare Input data, Prediction, GT
    for i in range(len(test_idx)):
        save_path = os.path.join(opt.save_path, 'test_' + f'{test_idx[i]}')
        Path(save_path).mkdir(parents=True, exist_ok=True)
        pred_json_path = os.path.join(save_path, 'pred_json')
        Path(pred_json_path).mkdir(parents=True, exist_ok=True)
        gt_json_path = os.path.join(save_path, 'gt_json')
        Path(gt_json_path).mkdir(parents=True, exist_ok=True)

        start_pose =  lafan_dataset.data['global_pos'][test_idx[i], from_idx]
        target_pose = lafan_dataset.data['global_pos'][test_idx[i], target_idx]
        stopover_pose = clue[test_idx[i],fixed]
        stopover_rot = global_q[test_idx[i],fixed]
        gt_stopover_pose = lafan_dataset.data['global_pos'][test_idx[i], from_idx + fixed]

        # Replace start/end with gt
        pred_global_pos[test_idx[i], 0] = start_pose

        gpos = pred_global_pos[test_idx[i]]
        grot = pred_global_rot_normalized[test_idx[i]]

        local_quaternion_stopover, local_positions_stopover = quat_ik(stopover_rot.detach().numpy(), stopover_pose.detach().numpy(), parents=skeleton_mocap.parents())
        local_quaternion, local_positions = quat_ik(grot, gpos, parents=skeleton_mocap.parents())

        img_aggr_list = []

        write_json(filename=os.path.join(pred_json_path, f'start.json'), local_q=local_quaternion[0], root_pos=local_positions[0,0], joint_names=joint_names)
        write_json(filename=os.path.join(pred_json_path, f'target.json'), local_q=local_quaternion[-1], root_pos=local_positions[-1,0], joint_names=joint_names)
        write_json(filename=os.path.join(pred_json_path, f'stopover.json'), local_q=local_quaternion_stopover, root_pos=local_positions_stopover[0], joint_names=joint_names)

        write_json(filename=os.path.join(gt_json_path, f'start.json'), local_q=local_quaternion[0], root_pos=local_positions[0,0], joint_names=joint_names)
        write_json(filename=os.path.join(gt_json_path, f'target.json'), local_q=local_quaternion[-1], root_pos=local_positions[-1,0], joint_names=joint_names)
        write_json(filename=os.path.join(gt_json_path, f'stopover.json'), local_q=local_quaternion_stopover, root_pos=local_positions_stopover[0], joint_names=joint_names)

        for t in range(horizon-1):
            
            if opt.plot_image:
                input_img_path = os.path.join(save_path, 'input')
                pred_img_path = os.path.join(save_path, 'pred_img')
                gt_img_path = os.path.join(save_path, 'gt_img')

                plot_pose_with_stop(start_pose, input_pos[test_idx[i],t].reshape(lafan_dataset.num_joints, 3), target_pose, stopover_pose, t, skeleton_mocap, save_dir=input_img_path, prefix='input')
                plot_pose_with_stop(start_pose, pred_global_pos[test_idx[i],t].reshape(lafan_dataset.num_joints, 3), target_pose, stopover_pose, t, skeleton_mocap, save_dir=pred_img_path, prefix='pred')
                plot_pose_with_stop(start_pose, lafan_dataset.data['global_pos'][test_idx[i], t+from_idx], target_pose, gt_stopover_pose, t, skeleton_mocap, save_dir=gt_img_path, prefix='gt')

                input_img = Image.open(os.path.join(input_img_path, 'input'+str(t)+'.png'), 'r')
                pred_img = Image.open(os.path.join(pred_img_path, 'pred'+str(t)+'.png'), 'r')
                gt_img = Image.open(os.path.join(gt_img_path, 'gt'+str(t)+'.png'), 'r')
                
                img_aggr_list.append(np.concatenate([input_img, pred_img, gt_img.resize(pred_img.size)], 1))

            write_json(filename=os.path.join(pred_json_path, f'{t:05}.json'), local_q=local_quaternion[t], root_pos=local_positions[t,0], joint_names=joint_names)
            write_json(filename=os.path.join(gt_json_path, f'{t:05}.json'), local_q=lafan_dataset.data['local_q'][test_idx[i], from_idx + t], root_pos=lafan_dataset.data['global_pos'][test_idx[i], from_idx + t, 0], joint_names=joint_names)

        # Save images
        if opt.plot_image:
            gif_path = os.path.join(save_path, f'img_{test_idx[i]}.gif')
            imageio.mimsave(gif_path, img_aggr_list, duration=0.1)
        print(f"ID {test_idx[i]}: test completed.")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--weight', default='latest')
    parser.add_argument('--exp_name', default='exp', help='experiment name')
    parser.add_argument('--data_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH', help='BVH dataset path')
    parser.add_argument('--skeleton_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh', help='path to reference skeleton')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_80/', help='path to save pickled processed data')
    parser.add_argument('--save_path', type=str, default='runs/test', help='path to save model')
    parser.add_argument('--motion_type', type=str, default='jumps', help='motion type')
    parser.add_argument('--plot_image', type=bool, default=False, help='plot image')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device("cpu")
    test(opt, device)
