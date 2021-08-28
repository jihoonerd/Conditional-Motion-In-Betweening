import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
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
from rmi.model.network import TransformerDiscriminator, TransformerGenerator, compute_gradient_penalty
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
    lafan_dataset = LAFAN1Dataset(lafan_path=opt.data_path, processed_data_dir=opt.processed_data_dir, train=True, target_action=['walk'], device=device, start_seq_length=30, cur_seq_length=30, max_transition_length=30)
    
    # Replace to noise for inbetweening frames
    from_idx, target_idx = 9, 40 # Starting frame: 9, Endframe:40, Inbetween start: 10, Inbetween end: 39
    horizon = target_idx - from_idx + 1

    pose_vec_gt, padding_dim = vectorize_pose(lafan_dataset.data['root_p'], lafan_dataset.data['local_q'], lafan_dataset.data['contact'], 96, device)
    pose_vectorized_gt = pose_vec_gt[:,from_idx:target_idx+1,:]

    global_pos = torch.Tensor(lafan_dataset.data['global_pos']).to(device)
    global_pos_gt = global_pos[:,from_idx:target_idx+1,:]

    tensor_dataset = TensorDataset(pose_vectorized_gt, global_pos_gt)
    lafan_data_loader = DataLoader(tensor_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    # Extract dimension from processed data
    root_v_dim = lafan_dataset.root_v_dim
    local_q_dim = lafan_dataset.local_q_dim
    contact_dim = lafan_dataset.contact_dim
    repr_dim = root_v_dim + local_q_dim + contact_dim

    tf_generator = TransformerGenerator(latent_dim=1024, seq_len=horizon, d_model=96, nhead=12, d_hid=1024, nlayers=6, dropout=0.1, out_dim=repr_dim, device=device)
    tf_generator.to(device)

    tf_discriminator = TransformerDiscriminator(seq_len=horizon, d_model=162, nhead=9, d_hid=512, nlayers=4, dropout=0.0, out_dim=repr_dim, device=device)
    tf_discriminator.to(device)

    generator_optim = Adam(params=tf_generator.parameters(), lr=opt.generator_learning_rate, betas=(opt.optim_beta1, opt.optim_beta2))
    generator_scheduler = torch.optim.lr_scheduler.StepLR(generator_optim, step_size=600, gamma=0.8)

    discriminator_optim = Adam(params=tf_discriminator.parameters(), lr=opt.discriminator_learning_rate, betas=(opt.optim_beta1, opt.optim_beta2))
    discriminator_scheudler = torch.optim.lr_scheduler.StepLR(discriminator_optim, step_size=600, gamma=0.8)

    LOGGER.info(f'Starting training for {epochs} epochs...')
    for epoch in range(1, epochs + 1):

        pbar = tqdm(lafan_data_loader, position=1, desc="Batch")

        log_dict = {}

        for pose_vectorized_gt, global_pos_gt in pbar:

            pose_vectorized_gt = pose_vectorized_gt.permute(1,0,2)
            seq_len = pose_vectorized_gt.shape[0]
            current_batch_size = pose_vectorized_gt.shape[1]

            input_noise = torch.randn(current_batch_size, opt.latent_dim, device=device)

            src_mask = torch.zeros((seq_len, seq_len), device=device).type(torch.bool)
            src_mask = src_mask.to(device)
            
            with amp.autocast(enabled=cuda):
                output = tf_generator(input_noise, src_mask)

                root_pred = output[:,:,:root_v_dim].permute(1,0,2)
                quat_pred = output[:,:,root_v_dim:root_v_dim + local_q_dim].permute(1,0,2)
                quat_pred_ = quat_pred.view(quat_pred.shape[0], quat_pred.shape[1], lafan_dataset.num_joints, 4)
                quat_pred_ = quat_pred_ / torch.norm(quat_pred_, dim = -1, keepdim = True)
                global_pos_pred = skeleton_mocap.forward_kinematics(quat_pred_, root_pred)

                output_w_pos = torch.cat([output, global_pos_pred.reshape(current_batch_size, seq_len, -1).permute(1,0,2)], axis=2)

                output_gt = pose_vectorized_gt[:,:,:repr_dim]
                output_gt_w_pos = torch.cat([output_gt, global_pos_gt.reshape(current_batch_size, seq_len, -1).permute(1,0,2)], axis=2)

                exp_dim = 1
                fake_discriminator_input = torch.cat([output_w_pos, torch.zeros((seq_len,current_batch_size,exp_dim), device=device)], axis=2)
                real_discriminator_input = torch.cat([output_gt_w_pos, torch.zeros((seq_len,current_batch_size,exp_dim), device=device)], axis=2)

                fake_disc_d_out = tf_discriminator(fake_discriminator_input.detach(), src_mask)
                fake_disc_lsgan = torch.mean((fake_disc_d_out) ** 2)
                real_disc_d_out = tf_discriminator(real_discriminator_input, src_mask)
                real_disc_lsgan = torch.mean((real_disc_d_out -  1) ** 2)

                total_d_loss = opt.loss_discriminator_weight * (real_disc_lsgan + fake_disc_lsgan)
        
            if total_d_loss > 0.25:
                discriminator_optim.zero_grad()
                total_d_loss.backward()
                discriminator_optim.step()

            with amp.autocast(enabled=cuda):
                gen_fake_disc_d_out = tf_discriminator(fake_discriminator_input, src_mask)
                gen_fake_ls_gan = torch.mean((gen_fake_disc_d_out -  1) ** 2)
                total_g_loss = opt.loss_generator_weight * gen_fake_ls_gan

            generator_optim.zero_grad()
            total_g_loss.backward()
            generator_optim.step()

        log_dict.update({"Train/Loss/Discriminator Loss": total_d_loss})
        log_dict.update({"Train/Loss/Generator Loss": total_g_loss})       

        for k, v in log_dict.items():
            summary_writer.add_scalar(k, v, epoch)
        wandb.log(log_dict)        

        # Save model
        if (epoch % save_interval) == 0:
            ckpt = {'epoch': epoch,
                    'tf_generator_state_dict': tf_generator.state_dict(),
                    'generator_optim_state_dict': generator_optim.state_dict()}
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
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_walk/', help='path to save pickled processed data')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or -1 or cpu')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='exp', help='save to project/name')
    parser.add_argument('--save_interval', type=int, default=50, help='Log model after every "save_period" epoch')
    parser.add_argument('--generator_learning_rate', type=float, default=0.001, help='generator_learning_rate')
    parser.add_argument('--discriminator_learning_rate', type=float, default=0.0001, help='discriminator_learning_rate')
    parser.add_argument('--cr_learning_rate', type=float, default=0.001, help='crh_infogan learning rate')
    parser.add_argument('--optim_beta1', type=float, default=0.5, help='optim_beta1')
    parser.add_argument('--optim_beta2', type=float, default=0.99, help='optim_beta2')
    parser.add_argument('--loss_root_weight', type=float, default=0.01, help='loss_pos_weight')
    parser.add_argument('--loss_quat_weight', type=float, default=1.0, help='loss_quat_weight')
    parser.add_argument('--loss_contact_weight', type=float, default=0.2, help='loss_contact_weight')
    parser.add_argument('--loss_global_pos_weight', type=float, default=1.0, help='loss_global_pos_weight')
    parser.add_argument('--loss_discriminator_weight', type=float, default=1.0, help='loss_gan_discriminator_weight')
    parser.add_argument('--loss_generator_weight', type=float, default=1.0, help='loss_gan_weight')
    parser.add_argument('--loss_code_weight', type=float, default=1.0, help='loss_code_weight')
    parser.add_argument('--infogan_disc_code', type=int, default=3, help='number of discrete codes for InfoGAN')
    parser.add_argument('--infogan_cont_code', type=int, default=0, help='number of continuous codes for InfoGAN')
    parser.add_argument('--loss_crh_weight', type=float, default=0.2, help='weight of H in InfoGAN-CR')
    parser.add_argument('--latent_dim', type=int, default=1024, help='input noise dimension')
    parser.add_argument('--gp_lambda', type=float, default=10, help='lambda for WGANGP')
    parser.add_argument('--n_critic', type=int, default=5, help='number of discriminator updates per generator update')
    opt = parser.parse_args()
    return opt



if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    train(opt, device)