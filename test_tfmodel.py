import argparse
import os
from pathlib import Path
import numpy as np
import torch
from rmi.model.skeleton import Skeleton, sk_joints_to_remove, sk_offsets, sk_parents
from PIL import Image
from pymo.parsers import BVHParser
import imageio
from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import flip_bvh
from rmi.model.network import TransformerModel
from rmi.model.preprocess import lerp_pose, vectorize_pose, replace_noise
from rmi.vis.pose import plot_pose


def test(opt, device):

    # Prepare Directories
    ckpt_path = Path(opt.ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    # Load Skeleton
    skeleton_mocap = Skeleton(offsets=sk_offsets, parents=sk_parents, device=device)
    skeleton_mocap.remove_joints(sk_joints_to_remove)
    
    # Flip, Load and preprocess data. It utilizes LAFAN1 utilities
    flip_bvh(opt.data_path)

    # Load LAFAN Dataset
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    lafan_dataset = LAFAN1Dataset(lafan_path=opt.data_path, processed_data_dir=opt.processed_data_dir, train=False, target_action=['walk'], device=device, start_seq_length=30, cur_seq_length=30, max_transition_length=30)
    
    # LERP In-betweening Frames
    from_idx, target_idx = 9, 40 # Starting frame: 9, Endframe:40, Inbetween start: 10, Inbetween end: 39
    horizon = target_idx - from_idx + 1
    root_lerped, local_q_lerped = replace_noise(lafan_dataset.data, from_idx=from_idx, target_idx=target_idx)
    contact_init = torch.ones(lafan_dataset.data['contact'].shape) * 0.5

    pose_vectorized_gt = vectorize_pose(lafan_dataset.data['root_p'], lafan_dataset.data['local_q'], lafan_dataset.data['contact'], 96, device)[:,from_idx:target_idx+1,:]
    pose_vectorized_lerp = vectorize_pose(root_lerped, local_q_lerped, contact_init, 96, device)[:,from_idx:target_idx+1,:]

    # Extract dimension from processed data
    root_v_dim = lafan_dataset.root_v_dim
    local_q_dim = lafan_dataset.local_q_dim
    contact_dim = lafan_dataset.contact_dim
    repr_dim = root_v_dim + local_q_dim + contact_dim

    pose_vectorized_gt = pose_vectorized_gt.permute(1,0,2)
    pose_vectorized_lerp = pose_vectorized_lerp.permute(1,0,2)

    src_mask = torch.zeros((horizon, horizon), device=device).type(torch.bool)
    src_mask = src_mask.to(device)

    # fixed_code = 3
    # pose_vectorized_lerp[:,:,repr_dim + fixed_code] = 1

    test_idx = [2,6,7,8,9,10,12,15,17]

    model = TransformerModel(seq_len=horizon, d_model=96, nhead=8, d_hid=1024, nlayers=8, dropout=0.05, out_dim=repr_dim, device=device)
    model.load_state_dict(ckpt['transformer_encoder_state_dict'])
    model.eval()

    output = model(pose_vectorized_lerp, src_mask)

    root_pred = output[:,test_idx,:root_v_dim].permute(1,0,2)
    quat_pred = output[:,test_idx,root_v_dim:root_v_dim+local_q_dim].permute(1,0,2)

    root_fk = root_pred
    quat_fk = quat_pred.reshape(len(test_idx), horizon, lafan_dataset.num_joints, 4)
    quat_fk = quat_fk / torch.norm(quat_fk, dim = -1, keepdim = True)
    pos_pred = skeleton_mocap.forward_kinematics(quat_fk, root_fk)


    quat_lerped = torch.Tensor(local_q_lerped[test_idx, from_idx:target_idx+1])
    quat_lerped = quat_lerped / torch.norm(quat_lerped, dim = -1, keepdim = True)
    pos_lerped = skeleton_mocap.forward_kinematics( 
                                            quat_lerped,
                                            torch.Tensor(root_lerped[test_idx,from_idx:target_idx+1,:]) 
                                            )

    # Compare Lerp, Prediction, GT
    for i in range(len(test_idx)):
        save_path = os.path.join(opt.save_path, 'test_' + f'{test_idx[i]}')
        Path(save_path).mkdir(parents=True, exist_ok=True)

        start_pose =  lafan_dataset.data['global_pos'][test_idx[i], from_idx]
        target_pose = lafan_dataset.data['global_pos'][test_idx[i], target_idx+1]

        img_aggr_list = []
        for t in range(horizon):
            
            # TODO: final frame does not match
            lerp_img_path = os.path.join(save_path, 'input')
            plot_pose(start_pose, pos_lerped[i,t].detach().numpy(), target_pose, t, skeleton_mocap, save_dir=lerp_img_path, prefix='input')
            pred_img_path = os.path.join(save_path, 'pred_img')
            plot_pose(start_pose, pos_pred[i,t].detach().numpy(), target_pose, t, skeleton_mocap, save_dir=pred_img_path, prefix='pred')
            gt_img_path = os.path.join(save_path, 'gt_img')
            plot_pose(start_pose, lafan_dataset.data['global_pos'][test_idx[i], t+from_idx], target_pose, t, skeleton_mocap, save_dir=gt_img_path, prefix='gt')

            lerp_img = Image.open(os.path.join(lerp_img_path, 'input'+str(t)+'.png'), 'r')
            pred_img = Image.open(os.path.join(pred_img_path, 'pred'+str(t)+'.png'), 'r')
            gt_img = Image.open(os.path.join(gt_img_path, 'gt'+str(t)+'.png'), 'r')
            
            img_aggr_list.append(np.concatenate([lerp_img, pred_img, gt_img.resize(pred_img.size)], 1))

        # Save images
        gif_path = os.path.join(save_path, f'img_{test_idx[i]}.gif')
        imageio.mimsave(gif_path, img_aggr_list, duration=0.1)
        print(f"ID {test_idx[i]}: test completed.")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--ckpt_path', type=str, default='train-600.pt', help='weights path')
    parser.add_argument('--data_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH', help='BVH dataset path')
    parser.add_argument('--skeleton_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh', help='path to reference skeleton')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_walk/', help='path to save pickled processed data')
    parser.add_argument('--save_path', type=str, default='runs/test', help='path to save model')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device("cpu")
    test(opt, device)
