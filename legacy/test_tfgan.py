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
from rmi.model.network import TransformerGenerator, TransformerModel
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


    # Extract dimension from processed data
    root_v_dim = lafan_dataset.root_v_dim
    local_q_dim = lafan_dataset.local_q_dim
    contact_dim = lafan_dataset.contact_dim
    repr_dim = root_v_dim + local_q_dim + contact_dim

    src_mask = torch.zeros((horizon, horizon), device=device).type(torch.bool)
    src_mask = src_mask.to(device)

    # fixed_code = 3
    # pose_vectorized_lerp[:,:,repr_dim + fixed_code] = 1

    model = TransformerGenerator(latent_dim=1024, seq_len=horizon, d_model=96, nhead=12, d_hid=1024, nlayers=6, dropout=0.1, out_dim=repr_dim, device=device)
    model.load_state_dict(ckpt['tf_generator_state_dict'])
    model.eval()

    

    input_noise = torch.randn(1, 1024, device=device)
    output = model(input_noise, src_mask)

    root_pred = output[:,:,:root_v_dim].permute(1,0,2)
    quat_pred = output[:,:,root_v_dim:root_v_dim + local_q_dim].permute(1,0,2)
    quat_pred_ = quat_pred.view(quat_pred.shape[0], quat_pred.shape[1], lafan_dataset.num_joints, 4)
    quat_pred_ = quat_pred_ / torch.norm(quat_pred_, dim = -1, keepdim = True)
    global_pos_pred = skeleton_mocap.forward_kinematics(quat_pred_, root_pred)


    # Compare Lerp, Prediction, GT
    save_path = os.path.join(opt.save_path, 'test_' + f'0')
    Path(save_path).mkdir(parents=True, exist_ok=True)

    start_pose =  lafan_dataset.data['global_pos'][0, from_idx]
    target_pose = lafan_dataset.data['global_pos'][0, target_idx+1]

    img_aggr_list = []
    for t in range(horizon):
        
        pred_img_path = os.path.join(save_path, 'pred_img')
        plot_pose(start_pose, global_pos_pred[0,t].detach().numpy(), target_pose, t, skeleton_mocap, save_dir=pred_img_path, prefix='pred')
        pred_img = Image.open(os.path.join(pred_img_path, 'pred'+str(t)+'.png'), 'r')
        img_aggr_list.append(pred_img)

    # Save images
    gif_path = os.path.join(save_path, f'img.gif')
    imageio.mimsave(gif_path, img_aggr_list, duration=0.1)
    print(f"test completed.")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--ckpt_path', type=str, default='train-400.pt', help='weights path')
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
