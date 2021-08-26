import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from kpt.model.skeleton import TorchSkeleton
from pymo.parsers import BVHParser
from torch.cuda import amp
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from rmi.model.skeleton import Skeleton, sk_joints_to_remove, sk_offsets, sk_parents

import wandb
from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import flip_bvh
from rmi.model.network import TransformerModel
from rmi.model.preprocess import lerp_pose, vectorize_pose, replace_noise
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
    
    # LERP In-betweening Frames
    from_idx, target_idx = 9, 39
    horizon = target_idx - from_idx
    root_lerped, local_q_lerped = replace_noise(lafan_dataset.data, from_idx=from_idx, target_idx=target_idx)
    contact_init = torch.ones(lafan_dataset.data['contact'].shape) * 0.5

    # FK To get global pos, and global rotation

    # TODO: Add contact
    pose_vectorized_gt = vectorize_pose(lafan_dataset.data['root_p'], lafan_dataset.data['local_q'], lafan_dataset.data['contact'], 96, device)[:,from_idx:target_idx,:]
    pose_vectorized_lerp = vectorize_pose(root_lerped, local_q_lerped, contact_init, 96, device)[:,from_idx:target_idx,:]
    global_pos = torch.Tensor(lafan_dataset.data['global_pos']).to(device)
    global_pos_gt = global_pos[:,from_idx:target_idx,:]

    tensor_dataset = TensorDataset(pose_vectorized_lerp, pose_vectorized_gt, global_pos_gt)
    lafan_data_loader = DataLoader(tensor_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    # Extract dimension from processed data
    root_v_dim = lafan_dataset.root_v_dim
    local_q_dim = lafan_dataset.local_q_dim
    contact_dim = lafan_dataset.contact_dim
    repr_dim = root_v_dim + local_q_dim + contact_dim

    transformer_encoder = TransformerModel(seq_len=horizon, d_model=96, nhead=8, d_hid=1024, nlayers=8, dropout=0.05, out_dim=repr_dim, device=device)
    transformer_encoder.to(device)

    l1_loss = nn.L1Loss()
    optim = Adam(params=transformer_encoder.parameters(), lr=opt.generator_learning_rate, betas=(opt.optim_beta1, opt.optim_beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=200, gamma=0.75)

    scaler = amp.GradScaler(enabled=cuda)

    LOGGER.info(f'Starting training for {epochs} epochs...')
    for epoch in range(1, epochs + 1):

        pbar = tqdm(lafan_data_loader, position=1, desc="Batch")

        for pose_vectorized_lerp, pose_vectorized_gt, global_pos_gt in pbar:

            pose_vectorized_gt = pose_vectorized_gt.permute(1,0,2)
            pose_vectorized_lerp = pose_vectorized_lerp.permute(1,0,2)

            seq_len = pose_vectorized_lerp.shape[0]
            src_mask = torch.zeros((seq_len, seq_len), device=device).type(torch.bool)
            src_mask = src_mask.to(device)

            # TODO: FK takes too much time. Need to make batch-FK
            
            with amp.autocast(enabled=cuda):
                output = transformer_encoder(pose_vectorized_lerp, src_mask)

                root_pred = output[:,:,:root_v_dim].permute(1,0,2)
                quat_pred = output[:,:,root_v_dim:root_v_dim + local_q_dim].permute(1,0,2)
                quat_pred_ = quat_pred.view(quat_pred.shape[0], quat_pred.shape[1], lafan_dataset.num_joints, 4)
                quat_pred_ = quat_pred_ / torch.norm(quat_pred_, dim = -1, keepdim = True)
                global_pos_pred = skeleton_mocap.forward_kinematics(quat_pred_, root_pred)
                contact_pred = torch.sigmoid(output[:,:,root_v_dim + local_q_dim:root_v_dim+local_q_dim+contact_dim]).permute(1,0,2)

                root_gt = pose_vectorized_gt[:,:,:root_v_dim].permute(1,0,2)
                quat_gt = pose_vectorized_gt[:,:,root_v_dim: root_v_dim + local_q_dim].permute(1,0,2)
                contact_gt = pose_vectorized_gt[:,:,root_v_dim + local_q_dim: root_v_dim + local_q_dim + contact_dim].permute(1,0,2)

                root_loss = l1_loss(root_pred, root_gt)
                quat_loss = l1_loss(quat_pred, quat_gt)
                contact_loss = l1_loss(contact_pred, contact_gt)
                global_pos_loss = l1_loss(global_pos_pred, global_pos_gt)

                total_g_loss = opt.loss_root_weight * root_loss + \
                               opt.loss_quat_weight * quat_loss + \
                               opt.loss_contact_weight * contact_loss + \
                               opt.loss_global_pos_weight * global_pos_loss

            optim.zero_grad()
            scaler.scale(total_g_loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(transformer_encoder.parameters(), 0.5)
            scaler.step(optim)
            scaler.update()

        scheduler.step()
                                
        # Log
        log_dict = {
            "Train/Loss/Root Loss": opt.loss_root_weight * root_loss, 
            "Train/Loss/Quaternion Loss": opt.loss_quat_weight * quat_loss,
            "Train/Loss/Contact Loss": opt.loss_contact_weight * contact_loss,
            "Train/Loss/Global Position Loss": opt.loss_global_pos_weight * global_pos_loss,
            "Train/Loss/Total Loss": total_g_loss
        }

        for k, v in log_dict.items():
            summary_writer.add_scalar(k, v, epoch)
        wandb.log(log_dict)        

        # Save model
        if (epoch % save_interval) == 0:
            ckpt = {'epoch': epoch,
                    'transformer_encoder_state_dict': transformer_encoder.state_dict(),
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
    parser.add_argument('--skeleton_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh', help='path to reference skeleton')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_walk_dance/', help='path to save pickled processed data')
    parser.add_argument('--batch_size', type=int, default=8 , help='batch size')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or -1 or cpu')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='exp', help='save to project/name')
    parser.add_argument('--save_interval', type=int, default=50, help='Log model after every "save_period" epoch')
    parser.add_argument('--generator_learning_rate', type=float, default=0.0001, help='generator_learning_rate')
    parser.add_argument('--discriminator_learning_rate', type=float, default=0.0001, help='discriminator_learning_rate')
    parser.add_argument('--cr_learning_rate', type=float, default=0.001, help='crh_infogan learning rate')
    parser.add_argument('--optim_beta1', type=float, default=0.9, help='optim_beta1')
    parser.add_argument('--optim_beta2', type=float, default=0.99, help='optim_beta2')
    parser.add_argument('--loss_root_weight', type=float, default=0.01, help='loss_pos_weight')
    parser.add_argument('--loss_quat_weight', type=float, default=1.0, help='loss_quat_weight')
    parser.add_argument('--loss_contact_weight', type=float, default=0.2, help='loss_contact_weight')
    parser.add_argument('--loss_global_pos_weight', type=float, default=0.01, help='loss_global_pos_weight')
    parser.add_argument('--loss_discriminator_weight', type=float, default=1.0, help='loss_gan_discriminator_weight')
    parser.add_argument('--loss_generator_weight', type=float, default=1.0, help='loss_gan_weight')
    parser.add_argument('--loss_code_weight', type=float, default=1.0, help='loss_code_weight')
    parser.add_argument('--infogan_disc_code', type=int, default=3, help='number of discrete codes for InfoGAN')
    parser.add_argument('--infogan_cont_code', type=int, default=0, help='number of continuous codes for InfoGAN')
    parser.add_argument('--loss_crh_weight', type=float, default=0.2, help='weight of H in InfoGAN-CR')
    opt = parser.parse_args()
    return opt



if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    train(opt, device)