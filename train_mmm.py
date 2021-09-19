import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

import wandb
from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import flip_bvh
from rmi.model.network import TransformerModel
from rmi.model.preprocess import (lerp_input_repr, lerp_reshaped, replace_noise,
                                  vectorize_representation)
from rmi.model.skeleton import (Skeleton, sk_joints_to_remove, sk_offsets,
                                sk_parents)
from utils.general import increment_path


def train(opt, device):

    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)
    save_interval = opt.save_interval

    # Set device to use
    epochs = opt.epochs
                          
    # Loggers
    summary_writer = SummaryWriter(str(save_dir))
    wandb.init(config=opt, project="RMIB-InfoGAN", entity=opt.entity, name=opt.exp_name, dir=opt.save_dir)

    # Load Skeleton
    skeleton_mocap = Skeleton(offsets=sk_offsets, parents=sk_parents, device=device)
    skeleton_mocap.remove_joints(sk_joints_to_remove)

    # Flip, Load and preprocess data. It utilizes LAFAN1 utilities
    flip_bvh(opt.data_path)

    # Load LAFAN Dataset
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    lafan_dataset = LAFAN1Dataset(lafan_path=opt.data_path, processed_data_dir=opt.processed_data_dir, train=True, target_action=[''], device=device)
    
    # Replace to noise for inbetweening frames
    from_idx, target_idx = opt.from_idx, opt.target_idx  # default: 9-40, max: 48
    horizon = target_idx - from_idx + 1
    print(f"Horizon: {horizon}")
    print(f"Horizon with Conditioning: {horizon + 1}")

    root_pos = torch.Tensor(lafan_dataset.data['root_p'][:, from_idx:target_idx+1]).to(device)
    local_q = torch.Tensor(lafan_dataset.data['local_q'][:, from_idx:target_idx+1]).to(device)
    local_q_normalized = nn.functional.normalize(local_q, p=2.0, dim=-1)

    global_pos, global_q = skeleton_mocap.forward_kinematics_with_rotation(local_q_normalized, root_pos)

    if opt.preserve_link_train:
        print("Link Preserving: Training")
        print("USE BONE LENGTH NORMALIZATION...")
        global_pos = skeleton_mocap.convert_to_unit_offset_mat(global_pos)
        global_pose_vec_pos = global_pos.reshape(global_pos.size(0), global_pos.size(1), -1).contiguous()
        global_pose_vec_rot = global_q.reshape(global_q.size(0), global_q.size(1), -1).contiguous()
        global_pose_vec_gt = torch.cat([global_pose_vec_pos, global_pose_vec_rot], dim=2)
    else:
        print("Link Prserving: Post Processing")
        global_pose_vec_gt = vectorize_representation(global_pos, global_q)
    global_pose_vec_input = global_pose_vec_gt.clone().detach()

    seq_categories = [x[:-1] for x in lafan_dataset.data['seq_names']]

    le = LabelEncoder()
    le_np = le.fit_transform(seq_categories)
    seq_labels = torch.Tensor(le_np).type(torch.int64).unsqueeze(1).to(device)
    np.save(f'{save_dir}/le_classes_.npy', le.classes_)

    tensor_dataset = TensorDataset(global_pose_vec_input, global_pose_vec_gt, seq_labels)
    lafan_data_loader = DataLoader(tensor_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    pos_dim = lafan_dataset.num_joints * 3
    rot_dim = lafan_dataset.num_joints * 4
    repr_dim = pos_dim + rot_dim
    nhead = 7 # repr_dim = 154

    transformer_encoder = TransformerModel(seq_len=horizon+1, d_model=repr_dim, nhead=nhead, d_hid=1024, nlayers=8, dropout=0.05, out_dim=repr_dim)
    transformer_encoder.to(device)

    l1_loss = nn.L1Loss()
    optim = Adam(params=transformer_encoder.parameters(), lr=opt.learning_rate, betas=(opt.optim_beta1, opt.optim_beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=400, gamma=0.8)

    for epoch in range(1, epochs + 1):

        pbar = tqdm(lafan_data_loader, position=1, desc="Batch")

        recon_pos_loss = []
        recon_rot_loss = []
        total_loss_list = []

        for minibatch_pose_input, minibatch_pose_gt, seq_label in pbar:

            cur_batch_size = minibatch_pose_input.size(0)
            seq_len = minibatch_pose_input.size(1)
            feature_dims = minibatch_pose_input.size(2)

            num_clue = 1

            # for _ in range(5):
            # mask_start_frame = np.random.randint(0, horizon)
            mask_start_frame = 0

            if opt.preserve_link_train:
                root_pos = minibatch_pose_input[:,:,:3]
                link_vec = minibatch_pose_input[:,:,3:pos_dim]
                rot_vec = minibatch_pose_input[:,:,pos_dim:]
                root_lerped = lerp_input_repr(root_pos, mask_start_frame)
                link_lerped = lerp_reshaped(link_vec, mask_start_frame, 21)
                rot_lerped = lerp_reshaped(rot_vec, mask_start_frame, 22)
                pose_interpolated_input = torch.cat([root_lerped, link_lerped, rot_lerped], dim=2)
            else:
                pose_interpolated_input = replace_noise(minibatch_pose_input, mask_start_frame)

            pose_interpolated_input = pose_interpolated_input.permute(1,0,2)

            src_mask = torch.zeros((seq_len, seq_len), device=device).type(torch.bool)
            src_mask = src_mask.to(device)
            
            output = transformer_encoder(pose_interpolated_input, src_mask, seq_label)

            pos_pred = output[:,:,:pos_dim].permute(1,0,2)
            pos_gt = minibatch_pose_gt[:,:,:pos_dim]
            pos_loss = l1_loss(pos_pred, pos_gt)
            recon_pos_loss.append(opt.loss_pos_weight * pos_loss)

            rot_pred = output[:,:,pos_dim:].permute(1,0,2)
            rot_gt = minibatch_pose_gt[:,:,pos_dim:]
            rot_loss = l1_loss(rot_pred, rot_gt)
            recon_rot_loss.append(opt.loss_rot_weight * rot_loss)

            total_g_loss = opt.loss_pos_weight * pos_loss + \
                            opt.loss_rot_weight * rot_loss

            total_loss_list.append(total_g_loss)
        
            optim.zero_grad()
            total_g_loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer_encoder.parameters(), 1.0, error_if_nonfinite=False)
            optim.step()

        scheduler.step()

        # Log
        log_dict = {
            "Train/Loss/Position Loss": torch.stack(recon_pos_loss).mean().item(), 
            "Train/Loss/Rotatation Loss": torch.stack(recon_rot_loss).mean().item(),
            "Train/Loss/Total Loss": torch.stack(total_loss_list).mean().item(),
        }

        for k, v in log_dict.items():
            summary_writer.add_scalar(k, v, epoch)
        wandb.log(log_dict)        

        # Save model
        if (epoch % save_interval) == 0:
            ckpt = {'epoch': epoch,
                    'transformer_encoder_state_dict': transformer_encoder.state_dict(),
                    'horizon': transformer_encoder.seq_len,
                    'from_idx': opt.from_idx,
                    'target_idx': opt.target_idx,
                    'd_model': transformer_encoder.d_model,
                    'nhead': transformer_encoder.nhead,
                    'd_hid': transformer_encoder.d_hid,
                    'nlayers': transformer_encoder.nlayers,
                    'optimizer_state_dict': optim.state_dict(),
                    'preserve_link_train': opt.preserve_link_train,
                    'loss': total_g_loss}
            torch.save(ckpt, os.path.join(wdir, f'train-{epoch}.pt'))
            print(f"{epoch} Epoch: model weights saved")

    wandb.run.finish()
    torch.cuda.empty_cache()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--weights', type=str, default='', help='weights path')
    parser.add_argument('--data_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH', help='BVH dataset path')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_original/', help='path to save pickled processed data')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or -1 or cpu')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='exp', help='save to project/name')
    parser.add_argument('--save_interval', type=int, default=1, help='Log model after every "save_period" epoch')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='generator_learning_rate')
    parser.add_argument('--optim_beta1', type=float, default=0.9, help='optim_beta1')
    parser.add_argument('--optim_beta2', type=float, default=0.99, help='optim_beta2')
    parser.add_argument('--loss_pos_weight', type=float, default=1.0, help='loss_pos_weight')
    parser.add_argument('--loss_rot_weight', type=float, default=1.0, help='loss_rot_weight')
    parser.add_argument('--from_idx', type=int, default=9, help='from idx')
    parser.add_argument('--target_idx', type=int, default=40, help='target idx')
    parser.add_argument('--preserve_link_train', action='store_true', help='use bone length to maintain link length')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    train(opt, device)
