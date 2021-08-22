import argparse
import os
from pathlib import Path
import numpy as np
import torch
from kpt.model.skeleton import TorchSkeleton
from PIL import Image
from pymo.parsers import BVHParser
import imageio
from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import flip_bvh
from rmi.model.network import TransformerModel
from rmi.model.preprocess import create_mask, lerp_pose, vectorize_pose
from rmi.vis.pose import plot_pose


def test(opt, device):

    # Prepare Directories
    ckpt_path = Path(opt.ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    # Load Skeleton
    parsed = BVHParser().parse(opt.skeleton_path) # Use first bvh info as a reference skeleton.
    skeleton = TorchSkeleton(skeleton=parsed.skeleton, root_name='Hips', device=device)

    # Flip, Load and preprocess data. It utilizes LAFAN1 utilities
    flip_bvh(opt.data_path)

    # Load LAFAN Dataset
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    lafan_dataset = LAFAN1Dataset(lafan_path=opt.data_path, processed_data_dir=opt.processed_data_dir, train=False, target_action=['walk'], device=device, start_seq_length=30, cur_seq_length=30, max_transition_length=30)
    
    # LERP In-betweening Frames
    from_idx, target_idx = 9, 39
    horizon = target_idx - from_idx
    root_lerped, local_q_lerped = lerp_pose(lafan_dataset.data, from_idx=from_idx, target_idx=target_idx)

    pose_vectorized_gt = vectorize_pose(lafan_dataset.data['root_p'], lafan_dataset.data['local_q'], 96, device)[:,from_idx:target_idx,:]
    pose_vectorized_lerp = vectorize_pose(root_lerped, local_q_lerped, 96, device)[:,from_idx:target_idx,:]

    # Extract dimension from processed data
    root_v_dim = lafan_dataset.root_v_dim
    local_q_dim = lafan_dataset.local_q_dim

    pose_vectorized_gt = pose_vectorized_gt.permute(1,0,2)
    pose_vectorized_lerp = pose_vectorized_lerp.permute(1,0,2)

    src_mask = torch.zeros((horizon, horizon), device=device).type(torch.bool)
    src_mask = src_mask.to(device)

    test_idx = [4,5,6]

    model = TransformerModel(seq_len=horizon, d_model=96, nhead=8, d_hid=1024, nlayers=8, dropout=0.05, out_dim=91, device=device)
    model.load_state_dict(ckpt['transformer_encoder_state_dict'])
    model.eval()

    output = model(pose_vectorized_lerp, src_mask)

    root_pred = output[:,test_idx,:root_v_dim].permute(1,0,2)
    quat_pred = output[:,test_idx,root_v_dim:].permute(1,0,2)

    global_pos_preds = []
    global_pos_lerps = []
    
    for t in range(horizon):
        root_fk = root_pred[:,t,:]
        quat_fk = quat_pred.reshape(len(test_idx), horizon, lafan_dataset.num_joints, 4)[:,t,:,:]
        pos_pred, _ = skeleton.forward_kinematics(root_fk, quat_fk, rot_repr='quaternion')
        global_pos_preds.append(pos_pred)

    for t in range(from_idx, target_idx):
        pos_lerped, _ = skeleton.forward_kinematics(torch.Tensor(root_lerped[test_idx,t,:]), 
                                                torch.Tensor(local_q_lerped[test_idx, t]), rot_repr='quaternion')
        global_pos_lerps.append(pos_lerped)

    # Compare Lerp, Prediction, GT
    for i in range(len(test_idx)):
        save_path = os.path.join(opt.save_path, 'test_' + f'{test_idx[i]}')
        Path(save_path).mkdir(parents=True, exist_ok=True)

        start_pose =  lafan_dataset.data['global_pos'][test_idx[i], from_idx]
        target_pose = lafan_dataset.data['global_pos'][test_idx[i], target_idx-1]

        img_aggr_list = []
        for t in range(horizon):
            
            lerp_img_path = os.path.join(save_path, 'lerp_img')
            plot_pose(start_pose, global_pos_lerps[t][i].detach().numpy(), target_pose, t, skeleton, save_dir=lerp_img_path, prefix='lerp')
            pred_img_path = os.path.join(save_path, 'pred_img')
            plot_pose(start_pose, global_pos_preds[t][i].detach().numpy(), target_pose, t, skeleton, save_dir=pred_img_path, prefix='pred')
            gt_img_path = os.path.join(save_path, 'gt_img')
            plot_pose(start_pose, lafan_dataset.data['global_pos'][test_idx[i], t+from_idx], target_pose, t, skeleton, save_dir=gt_img_path, prefix='gt')

            lerp_img = Image.open(os.path.join(lerp_img_path, 'lerp'+str(t)+'.png'), 'r')
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
    parser.add_argument('--ckpt_path', type=str, default='train-1500.pt', help='weights path')
    parser.add_argument('--data_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH', help='BVH dataset path')
    parser.add_argument('--skeleton_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh', help='path to reference skeleton')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data/', help='path to save pickled processed data')
    parser.add_argument('--save_path', type=str, default='runs/test', help='path to save model')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device("cpu")
    test(opt, device)
