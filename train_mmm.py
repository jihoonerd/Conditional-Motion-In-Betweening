import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import LabelEncoder
from torch.cuda import amp
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

import wandb
from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import flip_bvh
from rmi.model.network import TransformerModel
from rmi.model.preprocess import replace_inpainting_range, vectorize_pose
from rmi.model.skeleton import (Skeleton, sk_joints_to_remove, sk_offsets,
                                sk_parents)
from utils.general import increment_path

LOGGER = logging.getLogger(__name__)

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
    cuda = device.type != 'cpu'
    epochs = opt.epochs
                          
    # Loggers
    LOGGER.info(f"Start with 'tensorboard --logdir {opt.project}")
    summary_writer = SummaryWriter(str(save_dir))
    wandb.init(config=opt, project="RMIB-InfoGAN", entity=opt.entity, name=opt.exp_name, dir=opt.save_dir)

    # Load Skeleton
    skeleton_mocap = Skeleton(offsets=sk_offsets, parents=sk_parents, device=device)
    skeleton_mocap.remove_joints(sk_joints_to_remove)

    # Flip, Load and preprocess data. It utilizes LAFAN1 utilities
    flip_bvh(opt.data_path)

    # Load LAFAN Dataset
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    lafan_dataset = LAFAN1Dataset(lafan_path=opt.data_path, processed_data_dir=opt.processed_data_dir, train=True, target_action=[''], device=device, start_seq_length=30, cur_seq_length=30, max_transition_length=30)
    
    # Replace to noise for inbetweening frames
    from_idx, target_idx = opt.from_idx, opt.target_idx  # default: 9-40, max: 48
    horizon = target_idx - from_idx + 1
    print(f"Horizon: {horizon}")

    pose_vec_gt, _ = vectorize_pose(lafan_dataset.data['root_p'], lafan_dataset.data['local_q'], lafan_dataset.data['contact'], 96, device)
    pose_vectorized_gt = pose_vec_gt[:,from_idx:target_idx+1,:]
    pose_vectorized_input = pose_vectorized_gt.clone().detach()
    
    global_pos = torch.Tensor(lafan_dataset.data['global_pos']).to(device)
    global_pos_gt = global_pos[:,from_idx:target_idx+1,:]

    seq_categories = [x[:-1] for x in lafan_dataset.data['seq_names']]

    le = LabelEncoder()
    le_np = le.fit_transform(seq_categories)
    seq_labels = torch.Tensor(le_np).type(torch.int64).unsqueeze(1).to(device)
    np.save(f'{save_dir}/le_classes_.npy', le.classes_)
    
    tensor_dataset = TensorDataset(pose_vectorized_input, pose_vectorized_gt, global_pos_gt, seq_labels)
    lafan_data_loader = DataLoader(tensor_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    # Extract dimension from processed data
    root_v_dim = lafan_dataset.root_v_dim
    local_q_dim = lafan_dataset.local_q_dim
    contact_dim = lafan_dataset.contact_dim
    repr_dim = root_v_dim + local_q_dim + contact_dim

    transformer_encoder = TransformerModel(seq_len=horizon, d_model=96, nhead=8, d_hid=2048, nlayers=12, dropout=0.05, out_dim=repr_dim)
    transformer_encoder.to(device)

    l1_loss = nn.L1Loss()
    optim = Adam(params=transformer_encoder.parameters(), lr=opt.learning_rate, betas=(opt.optim_beta1, opt.optim_beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=400, gamma=0.8)

    scaler = amp.GradScaler(enabled=cuda)

    LOGGER.info(f'Starting training for {epochs} epochs...')
    for epoch in range(1, epochs + 1):

        pbar = tqdm(lafan_data_loader, position=1, desc="Batch")

        root_loss_list = []
        quat_loss_list = []
        contact_loss_list = []
        global_pos_loss_list = []
        lock_in_loss_list = []
        total_loss_list = []

        for pose_vectorized_input, pose_vectorized_gt, global_pos_gt, seq_label in pbar:
            # N, L, D
            # L = starting frame + inbetweening frame + target frame = horizon
            cur_batch_size = pose_vectorized_gt.size(0)
            seq_len = pose_vectorized_gt.size(1)
            feature_dims = pose_vectorized_input.size(2)

            ## REPLACE INPUT INBETWEEN Frames
            num_clue = np.random.choice([1])

            valid_upper = horizon - num_clue
            valid_lower = 1
            mask_start_frame = np.random.randint(valid_lower, valid_upper)

            pose_vectorized_inpainting = replace_inpainting_range(pose_vectorized_input, mask_start_frame, num_clue, cur_batch_size, feature_dims, infill_value=0.1)
   
            ## MAKE INFILLING CODE HERE ##
            infilling_code = np.zeros((1, horizon))
            infilling_code[0, 1:mask_start_frame] = 1
            infilling_code[0, mask_start_frame+num_clue:-1] = 1
            infilling_code = torch.tensor(infilling_code, dtype=torch.int, device=device)
            ##############################

            pose_vectorized_gt = pose_vectorized_gt.permute(1,0,2)
            pose_vectorized_inpainting = pose_vectorized_inpainting.permute(1,0,2)

            src_mask = torch.zeros((seq_len, seq_len), device=device).type(torch.bool)
            src_mask = src_mask.to(device)
            
            with amp.autocast(enabled=cuda):
                output = transformer_encoder(pose_vectorized_inpainting, src_mask, seq_label, infilling_code)

                root_pred = output[:,:,:root_v_dim].permute(1,0,2)
                quat_pred = output[:,:,root_v_dim:root_v_dim + local_q_dim].permute(1,0,2)
                quat_pred_ = quat_pred.view(quat_pred.shape[0], quat_pred.shape[1], lafan_dataset.num_joints, 4)
                quat_pred_ = quat_pred_ / torch.norm(quat_pred_, dim = -1, keepdim = True)
                global_pos_pred = skeleton_mocap.forward_kinematics(quat_pred_, root_pred)
                contact_pred = torch.sigmoid(output[:,:,root_v_dim + local_q_dim:root_v_dim+local_q_dim+contact_dim]).permute(1,0,2)
                start_lock_in_pred = global_pos_pred[:, :5]
                end_lock_in_pred = global_pos_pred[:, -5:]

                root_gt = pose_vectorized_gt[:,:,:root_v_dim].permute(1,0,2)
                quat_gt = pose_vectorized_gt[:,:,root_v_dim: root_v_dim + local_q_dim].permute(1,0,2)
                contact_gt = pose_vectorized_gt[:,:,root_v_dim + local_q_dim: root_v_dim + local_q_dim + contact_dim].permute(1,0,2)
                start_lock_in_gt = global_pos_gt[:, :5]
                end_lock_in_gt = global_pos_gt[:, -5:]

                root_loss = l1_loss(root_pred, root_gt)
                quat_loss = l1_loss(quat_pred, quat_gt)
                contact_loss = l1_loss(contact_pred, contact_gt)
                global_pos_loss = l1_loss(global_pos_pred, global_pos_gt)
                lock_in_loss = l1_loss(start_lock_in_pred, start_lock_in_gt) + l1_loss(end_lock_in_pred, end_lock_in_gt)

                total_g_loss = opt.loss_root_weight * root_loss + \
                               opt.loss_quat_weight * quat_loss + \
                               opt.loss_contact_weight * contact_loss + \
                               opt.loss_global_pos_weight * global_pos_loss + \
                               opt.loss_lock_in_weight * lock_in_loss


                root_loss_list.append(opt.loss_root_weight * root_loss)
                quat_loss_list.append(opt.loss_quat_weight * quat_loss)
                contact_loss_list.append(opt.loss_contact_weight * contact_loss)
                global_pos_loss_list.append(opt.loss_global_pos_weight * global_pos_loss)
                lock_in_loss_list.append(opt.loss_lock_in_weight * lock_in_loss)
                total_loss_list.append(total_g_loss)
            

            optim.zero_grad()
            scaler.scale(total_g_loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(transformer_encoder.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

        scheduler.step()

        # Log
        log_dict = {
            "Train/Loss/Root Loss": torch.stack(root_loss_list).mean().item(), 
            "Train/Loss/Quaternion Loss": torch.stack(quat_loss_list).mean().item(),
            "Train/Loss/Contact Loss": torch.stack(contact_loss_list).mean().item(),
            "Train/Loss/Global Position Loss": torch.stack(global_pos_loss_list).mean().item(),
            "Train/Loss/Lock-in Loss": torch.stack(lock_in_loss_list).mean().item(),
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
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_all/', help='path to save pickled processed data')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or -1 or cpu')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='exp', help='save to project/name')
    parser.add_argument('--save_interval', type=int, default=1, help='Log model after every "save_period" epoch')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='generator_learning_rate')
    parser.add_argument('--optim_beta1', type=float, default=0.5, help='optim_beta1')
    parser.add_argument('--optim_beta2', type=float, default=0.99, help='optim_beta2')
    parser.add_argument('--loss_root_weight', type=float, default=0.01, help='loss_pos_weight')
    parser.add_argument('--loss_quat_weight', type=float, default=1.0, help='loss_quat_weight')
    parser.add_argument('--loss_contact_weight', type=float, default=0.2, help='loss_contact_weight')
    parser.add_argument('--loss_global_pos_weight', type=float, default=0.01, help='loss_global_pos_weight')
    parser.add_argument('--loss_lock_in_weight', type=float, default=0.02, help='loss_lock_in_weight')
    parser.add_argument('--from_idx', type=int, default=9, help='from idx')
    parser.add_argument('--target_idx', type=int, default=40, help='target idx')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    train(opt, device)
