import argparse
import os
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn

from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import drop_end_quat
from rmi.model.network import TransformerModel
from rmi.model.preprocess import (lerp_input_repr, replace_constant, slerp_input_repr,
                                  vectorize_representation)
from rmi.model.skeleton import (Skeleton, sk_joints_to_remove, sk_offsets,
                                sk_parents)
from sklearn.preprocessing import LabelEncoder
from rmi.lafan1 import extract


def test(opt, device):

    save_dir = Path(os.path.join('runs', 'train', opt.exp_name))
    Path(os.path.join('cond_bch', opt.gt_motion)).mkdir(parents=True, exist_ok=True)
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

    # Extract stats
    train_actors = ['subject1', 'subject2', 'subject3', 'subject4']
    bvh_folder = os.path.join('ubisoft-laforge-animation-dataset', 'output', 'BVH')
    stats_file = os.path.join(opt.processed_data_dir, 'train_stats.pkl')

    if not os.path.exists(stats_file):
        x_mean, x_std, offsets = extract.get_train_stats(bvh_folder, train_actors)
        with open(stats_file, 'wb') as f:
            pickle.dump({
                'x_mean': x_mean,
                'x_std': x_std,
                'offsets': offsets,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Reusing stats file: ' + stats_file)
        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)
        x_mean = stats['x_mean']
        x_std = stats['x_std']
        offsets = stats['offsets']


    total_data = lafan_dataset.data['global_pos'].shape[0]
    
    # Replace with noise to In-betweening Frames
    from_idx, target_idx = ckpt['from_idx'], ckpt['target_idx'] # default: 9-38, max: 48
    horizon = ckpt['horizon']
    print(f"HORIZON: {horizon}")

    test_idx = []
    for i in range(total_data):
        test_idx.append(i)

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
    global_pos[:,fixed] += torch.Tensor([0,0,0]).expand(global_pos.size(0),lafan_dataset.num_joints,3)

    interpolation = ckpt['interpolation']

    if interpolation == 'constant':
        global_pose_vec_gt = vectorize_representation(global_pos, global_q)
        global_pose_vec_input = global_pose_vec_gt.clone().detach()
        pose_interpolated_input = replace_constant(global_pose_vec_input, fixed)

    elif interpolation == 'slerp':
        global_pose_vec_gt = vectorize_representation(global_pos, global_q)
        global_pose_vec_input = global_pose_vec_gt.clone().detach()
        root_vec = global_pose_vec_input[:,:,:pos_dim]
        rot_vec = global_pose_vec_input[:,:,pos_dim:]
        root_lerped = lerp_input_repr(root_vec, fixed)
        rot_slerped = slerp_input_repr(rot_vec, fixed)
        pose_interpolated_input = torch.cat([root_lerped, rot_slerped], dim=2)

    else:
        raise ValueError('Invalid interpolation method')
    
    pose_vectorized_input = pose_interpolated_input.permute(1,0,2)

    src_mask = torch.zeros((horizon, horizon), device=device).type(torch.bool)
    src_mask = src_mask.to(device)

    seq_categories = [x[:-1] for x in lafan_dataset.data['seq_names']]

    l1_loss = nn.L1Loss()

    le = LabelEncoder()
    le.classes_ = np.load(os.path.join(save_dir, 'le_classes_.npy'))

    model = TransformerModel(seq_len=ckpt['horizon'], d_model=ckpt['d_model'], nhead=ckpt['nhead'], d_hid=ckpt['d_hid'], nlayers=ckpt['nlayers'], dropout=0.0, out_dim=repr_dim)
    model.load_state_dict(ckpt['transformer_encoder_state_dict'])
    model.eval()

    testing_motions = ['walk', 'run', 'jumps', 'ground', 'dance', 'aiming']

    bch_total = []
    for cond_motion in testing_motions:
        l2p = []
        l2q = []

        pred_rot_npss = []

        print(f"GT: {opt.gt_motion}")
        print(f"Condition: {cond_motion}")
        
        bch_out = {}

        bch_out['cond_motion'] = cond_motion
        bch_out['gt_motion'] = opt.gt_motion

        for i in range(len(test_idx)):
            motion_index = np.where(le.classes_ == cond_motion)[0][0]

            conditioning_label = torch.Tensor([[motion_index]]).type(torch.int64).to(device)
            cond_output, _ = model(pose_vectorized_input[:, test_idx[i]:test_idx[i]+1, :], src_mask, conditioning_label)

            output = cond_output

            # TODO: Start-End position replacement
            pred_global_pos = output[1:,:,:pos_dim].permute(1,0,2).reshape(1,horizon-1,22,3)
            global_pos_unit_vec = skeleton_mocap.convert_to_unit_offset_mat(pred_global_pos)
            pred_global_pos = skeleton_mocap.convert_to_global_pos(global_pos_unit_vec).detach().numpy()

            # Replace start/end with gt
            gt_global_pos = lafan_dataset.data['global_pos'][test_idx[i]:test_idx[i]+1, from_idx:target_idx+1].reshape(1, -1, lafan_dataset.num_joints, 3)
            pred_global_pos[0,0] = gt_global_pos[0,0] 
            pred_global_pos[0,-1] = gt_global_pos[0,-1]

            pred_global_rot = output[1:,:,pos_dim:].permute(1,0,2).reshape(1,horizon-1,22,4)
            pred_global_rot_normalized = nn.functional.normalize(pred_global_rot, p=2.0, dim=3)
            gt_global_rot = global_q[test_idx[i]:test_idx[i]+1]
            pred_global_rot_normalized[0,0] = gt_global_rot[0,0]
            pred_global_rot_normalized[0,-1] = gt_global_rot[0,-1]
            pred_rot_npss.append(pred_global_rot_normalized)

            # Normalize for L2P
            normalized_gt_pos = torch.Tensor((lafan_dataset.data['global_pos'][test_idx[i]:test_idx[i]+1, from_idx:target_idx+1].reshape(1, -1, lafan_dataset.num_joints * 3).transpose(0,2,1) - x_mean) / x_std)
            normalized_pred_pos = torch.Tensor((pred_global_pos.reshape(1, -1, lafan_dataset.num_joints * 3).transpose(0,2,1) - x_mean) / x_std)

            l2p.append(torch.mean(torch.norm(normalized_pred_pos[0] - normalized_gt_pos[0], dim=(0))).item())
            l2q.append(torch.mean(torch.norm(pred_global_rot_normalized[0] - global_q[test_idx[i]], dim=(1,2))).item())
        
        l2p_mean = np.mean(l2p)
        l2q_mean = np.mean(l2q)

        print(f"TOTAL TEST DATA: {len(l2p)}")
        print(f"L2P: {l2p_mean}")
        print(f"L2Q: {l2q_mean}")
        print("=================")

        bch_out['L2P'] = l2p_mean
        bch_out['L2Q'] = l2q_mean
        bch_out['TotalData'] = len(l2p)
        bch_total.append(bch_out)
        with open(os.path.join('cond_bch', opt.gt_motion, f'{opt.gt_motion}_bch.pkl'), 'wb') as f:
            pickle.dump(bch_total, f)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--exp_name', default='slerp30_qnorm', help='experiment name')
    parser.add_argument('--data_path', type=str, default='ubisoft-laforge-animation-dataset/output/TEST_WALK', help='BVH dataset path')
    parser.add_argument('--skeleton_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh', help='path to reference skeleton')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_test_walk/', help='path to save pickled processed data')
    parser.add_argument('--save_path', type=str, default='runs/test', help='path to save model')
    parser.add_argument('--gt_motion', type=str, default='walk')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device("cpu")
    test(opt, device)