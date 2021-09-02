import argparse
import os
from pathlib import Path
import numpy as np
import torch
from rmi.model.skeleton import Skeleton, sk_joints_to_remove, sk_offsets, sk_parents
from PIL import Image
import imageio
from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import flip_bvh
from rmi.model.network import TransformerModel
from rmi.model.preprocess import vectorize_pose, replace_noise
from rmi.vis.pose import plot_pose
from sklearn.preprocessing import LabelEncoder


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
    
    # Flip, Load and preprocess data. It utilizes LAFAN1 utilities
    flip_bvh(opt.data_path)

    # Load LAFAN Dataset
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    lafan_dataset = LAFAN1Dataset(lafan_path=opt.data_path, processed_data_dir=opt.processed_data_dir, train=False, target_action=[''], device=device, start_seq_length=30, cur_seq_length=30, max_transition_length=30)
    
    # Replace with noise to In-betweening Frames
    from_idx, target_idx = ckpt['from_idx'], ckpt['target_idx'] # default: 9-40, max: 48
    horizon = ckpt['horizon']
    print(f"HORIZON: {horizon}")

    root_noised, local_q_noised = replace_noise(lafan_dataset.data, from_idx=from_idx, target_idx=target_idx)
    contact_init = torch.ones(lafan_dataset.data['contact'].shape) * 0.5

    pose_vec_gt, padding_dim = vectorize_pose(lafan_dataset.data['root_p'], lafan_dataset.data['local_q'], lafan_dataset.data['contact'], 96, device)
    pose_vectorized_gt = pose_vec_gt[:,from_idx:target_idx+1,:]

    pose_vec_noised, _ = vectorize_pose(root_noised, local_q_noised, contact_init, 96, device)
    pose_vectorized_noised = pose_vec_noised[:,from_idx:target_idx+1,:]

    # Extract dimension from processed data
    root_v_dim = lafan_dataset.root_v_dim
    local_q_dim = lafan_dataset.local_q_dim
    contact_dim = lafan_dataset.contact_dim
    repr_dim = root_v_dim + local_q_dim + contact_dim

    pose_vectorized_gt = pose_vectorized_gt.permute(1,0,2)
    pose_vectorized_noised = pose_vectorized_noised.permute(1,0,2)

    src_mask = torch.zeros((horizon, horizon), device=device).type(torch.bool)
    src_mask = src_mask.to(device)

    seq_categories = [x[:-1] for x in lafan_dataset.data['seq_names']]

    le = LabelEncoder()
    le.classes_ = np.load(os.path.join(save_dir, 'le_classes_.npy'))

    target_seq = opt.motion_type
    seq_id = np.where(le.classes_==target_seq)[0]
    conditioning_labels = np.expand_dims((np.repeat(seq_id[0], repeats=len(seq_categories))), axis=1)
    conditioning_labels = torch.Tensor(conditioning_labels).type(torch.int64).to(device)

    test_idx = [150,300,500,700,800,1000,1200,1600,1610,1620]

    model = TransformerModel(seq_len=ckpt['horizon'], d_model=ckpt['d_model'], nhead=ckpt['nhead'], d_hid=ckpt['d_hid'], nlayers=ckpt['nlayers'], dropout=0.05, out_dim=repr_dim)
    model.load_state_dict(ckpt['transformer_encoder_state_dict'])
    model.eval()

    output = model(pose_vectorized_noised, src_mask, conditioning_labels)

    root_pred = output[:,test_idx,:root_v_dim].permute(1,0,2)
    quat_pred = output[:,test_idx,root_v_dim:root_v_dim+local_q_dim].permute(1,0,2)

    root_fk = root_pred
    quat_fk_ = quat_pred.view(len(test_idx), horizon, lafan_dataset.num_joints, 4)
    quat_fk_ = quat_fk_ / torch.norm(quat_fk_, dim = -1, keepdim = True)
    pos_pred = skeleton_mocap.forward_kinematics(quat_fk_, root_fk)


    quat_noised_ = torch.Tensor(local_q_noised[test_idx, from_idx:target_idx+1])
    quat_noised_ = quat_noised_ / torch.norm(quat_noised_, dim = -1, keepdim = True)
    pos_noised = skeleton_mocap.forward_kinematics( 
                                            quat_noised_,
                                            torch.Tensor(root_noised[test_idx,from_idx:target_idx+1,:]) 
                                            )

    # Compare Input data, Prediction, GT
    for i in range(len(test_idx)):
        save_path = os.path.join(opt.save_path, 'test_' + f'{test_idx[i]}')
        Path(save_path).mkdir(parents=True, exist_ok=True)

        start_pose =  lafan_dataset.data['global_pos'][test_idx[i], from_idx]
        target_pose = lafan_dataset.data['global_pos'][test_idx[i], target_idx]

        img_aggr_list = []
        for t in range(horizon):
            
            # TODO: final frame does not match
            input_img_path = os.path.join(save_path, 'input')
            plot_pose(start_pose, pos_noised[i,t].detach().numpy(), target_pose, t, skeleton_mocap, save_dir=input_img_path, prefix='input')
            pred_img_path = os.path.join(save_path, 'pred_img')
            plot_pose(start_pose, pos_pred[i,t].detach().numpy(), target_pose, t, skeleton_mocap, save_dir=pred_img_path, prefix='pred')
            gt_img_path = os.path.join(save_path, 'gt_img')
            plot_pose(start_pose, lafan_dataset.data['global_pos'][test_idx[i], t+from_idx], target_pose, t, skeleton_mocap, save_dir=gt_img_path, prefix='gt')

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
    parser.add_argument('--exp_name', default='COND_BERT(64 d_hid 2048)', help='experiment name')
    parser.add_argument('--data_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH', help='BVH dataset path')
    parser.add_argument('--skeleton_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh', help='path to reference skeleton')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_all/', help='path to save pickled processed data')
    parser.add_argument('--save_path', type=str, default='runs/test', help='path to save model')
    parser.add_argument('--motion_type', type=str, default='walk', help='motion type')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device("cpu")
    test(opt, device)
