import argparse
import logging
import os
from pathlib import Path
import random
import torch
import torch.nn as nn
from torch.nn.modules import transformer
import yaml
from kpt.model.skeleton import TorchSkeleton
from pymo.parsers import BVHParser
from torch.cuda import amp
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

import wandb
from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import flip_bvh, generate_infogan_code
from rmi.model.network import TransformerModel, MotionDiscriminator, DInfoGAN, QInfoGAN, InfoGANCRH
from rmi.model.preprocess import lerp_pose, vectorize_pose
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
    parsed = BVHParser().parse(opt.skeleton_path) # Use first bvh info as a reference skeleton.
    skeleton = TorchSkeleton(skeleton=parsed.skeleton, root_name='Hips', device=device)

    # Flip, Load and preprocess data. It utilizes LAFAN1 utilities
    flip_bvh(opt.data_path)

    # Load LAFAN Dataset
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    lafan_dataset = LAFAN1Dataset(lafan_path=opt.data_path, processed_data_dir=opt.processed_data_dir, train=True, target_action=['walk', 'dance'], device=device, start_seq_length=30, cur_seq_length=30, max_transition_length=30)
    
    # LERP In-betweening Frames
    from_idx, target_idx = 9, 39
    horizon = target_idx - from_idx
    root_lerped, local_q_lerped = lerp_pose(lafan_dataset.data, from_idx=from_idx, target_idx=target_idx)
    contact_init = torch.ones(lafan_dataset.data['contact'].shape) * 0.5

    # FK To get global pos, and global rotation
    pose_vectorized_gt = vectorize_pose(lafan_dataset.data['root_p'], lafan_dataset.data['local_q'], lafan_dataset.data['contact'], 100, device)[:,from_idx:target_idx,:]
    pose_vectorized_lerp = vectorize_pose(root_lerped, local_q_lerped, contact_init, 100, device)[:,from_idx:target_idx,:]

    tensor_dataset = TensorDataset(pose_vectorized_lerp, pose_vectorized_gt)
    lafan_data_loader = DataLoader(tensor_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    # Extract dimension from processed data
    root_v_dim = lafan_dataset.root_v_dim
    local_q_dim = lafan_dataset.local_q_dim
    contact_dim = lafan_dataset.contact_dim
    repr_dim = root_v_dim + local_q_dim + contact_dim

    transformer_encoder = TransformerModel(seq_len=horizon, d_model=100, nhead=10, d_hid=1024, nlayers=8, dropout=0.03, out_dim=repr_dim, device=device)
    transformer_encoder.to(device)

    # Discriminator
    infogan_discriminator = MotionDiscriminator(input_dim=95, out_dim=128)
    infogan_discriminator.to(device)
    
    # DInfoGAN
    d_infogan = DInfoGAN(input_dim=128)
    d_infogan.to(device)

    # QInfoGAN
    q_infogan = QInfoGAN(input_dim=128, discrete_code_dim=opt.infogan_disc_code, continuous_code_dim=opt.infogan_cont_code)
    q_infogan.to(device)

    # InfoGAN CRH
    infogan_crh = InfoGANCRH(input_dim=190, out_dim=opt.infogan_disc_code)
    infogan_crh.to(device)

    categorical_loss = nn.CrossEntropyLoss()

    generator_optim = Adam(params=list(transformer_encoder.parameters()) +
                                  list(q_infogan.parameters()),
                                  lr=opt.generator_learning_rate,
                                  betas=(opt.optim_beta1, opt.optim_beta2))
    discriminator_optim = Adam(params=list(infogan_discriminator.parameters()) +
                                      list(d_infogan.parameters()), 
                                      lr=opt.discriminator_learning_rate, 
                                      betas=(opt.optim_beta1, opt.optim_beta2))

    cr_optim = Adam(params=list(transformer_encoder.parameters()) + 
                           list(infogan_crh.parameters()), 
                    lr=opt.cr_learning_rate, 
                    betas=(opt.optim_beta1, opt.optim_beta2))

    scaler = amp.GradScaler(enabled=cuda)

    LOGGER.info(f'Starting training for {epochs} epochs...')
    for epoch in range(1, epochs + 1):

        pbar = tqdm(lafan_data_loader, position=1, desc="Batch")

        for pose_vectorized_lerp, pose_vectorized_gt in pbar:

            current_batch_size = pose_vectorized_gt.shape[0]
            pose_vectorized_gt = pose_vectorized_gt.permute(1,0,2)
            pose_vectorized_lerp = pose_vectorized_lerp.permute(1,0,2)

            # InfoGAN code (per motion)
            infogan_code_gen, fake_indices = generate_infogan_code(batch_size=current_batch_size, discrete_code_dim=opt.infogan_disc_code, continuous_code_dim=opt.infogan_cont_code, device=device)
            infogan_code_gen1, fake_indices1 = generate_infogan_code(batch_size=current_batch_size, discrete_code_dim=opt.infogan_disc_code, continuous_code_dim=opt.infogan_cont_code, device=device)
            infogan_code_gen2, fake_indices2 = generate_infogan_code(batch_size=current_batch_size, discrete_code_dim=opt.infogan_disc_code, continuous_code_dim=opt.infogan_cont_code, device=device)
            
            seq_len = pose_vectorized_lerp.shape[0]
            src_mask = torch.zeros((seq_len, seq_len), device=device).type(torch.bool)
            src_mask = src_mask.to(device)
    
            with amp.autocast(enabled=cuda):

                # Infogan CR
                fixed_code = random.randint(0,opt.infogan_disc_code - 1)

                # x in Figure 3 (InfoGAN-CR)
                pose_vectorized_lerp[:,:,repr_dim: repr_dim + opt.infogan_disc_code + opt.infogan_cont_code] = infogan_code_gen
                output = transformer_encoder(pose_vectorized_lerp, src_mask)

                root_pred = output[:,:,:root_v_dim].permute(1,0,2)
                quat_pred = output[:,:,root_v_dim:root_v_dim + local_q_dim].permute(1,0,2)
                contact_pred = torch.sigmoid(output[:,:,root_v_dim + local_q_dim:root_v_dim+local_q_dim+contact_dim]).permute(1,0,2)

                root_gt = pose_vectorized_gt[:,:,:root_v_dim].permute(1,0,2)
                quat_gt = pose_vectorized_gt[:,:,root_v_dim: root_v_dim + local_q_dim].permute(1,0,2)
                contact_gt = pose_vectorized_gt[:,:,root_v_dim + local_q_dim: root_v_dim + local_q_dim + contact_dim].permute(1,0,2)

                # INFOGAN PART
                ## Discriminator Part
                fake_input = torch.cat([root_pred, quat_pred, contact_pred], dim=2).permute(0,2,1)
                real_input = torch.cat([root_gt, quat_gt, contact_gt], dim=2).permute(0,2,1)

                infogan_disc_fake_gan_out = infogan_discriminator(fake_input.detach())
                infogan_disc_fake_d_out = d_infogan(infogan_disc_fake_gan_out)
                info_disc_fake_loss = torch.mean((infogan_disc_fake_d_out) ** 2)

                infogan_disc_real_gan_out = infogan_discriminator(real_input)
                infogan_disc_real_d_out = d_infogan(infogan_disc_real_gan_out)
                info_disc_real_loss = torch.mean((infogan_disc_real_d_out -  1) ** 2)


            # Optimize
            # TODO: Update discriminator only when: 
            discriminator_optim.zero_grad()
            scaler.scale(opt.loss_discriminator_weight * info_disc_fake_loss).backward()
            scaler.scale(opt.loss_discriminator_weight * info_disc_real_loss).backward()
            total_d_loss = opt.loss_discriminator_weight * (info_disc_fake_loss+info_disc_real_loss)
            scaler.unscale_(discriminator_optim)
            torch.nn.utils.clip_grad_norm_(infogan_discriminator.parameters(), 0.05)
            torch.nn.utils.clip_grad_norm_(d_infogan.parameters(), 0.05)
            scaler.step(discriminator_optim)

            with amp.autocast(enabled=cuda):
                ## Generator Part (Consider training twice)
                info_gen_fake_gan_out = infogan_discriminator(fake_input)
                info_gen_fake_d_out = d_infogan(info_gen_fake_gan_out)
                info_gen_fake_loss = torch.mean((info_gen_fake_d_out - 1) ** 2)

                info_gen_fake_q_out, _, _ = q_infogan(info_gen_fake_gan_out)
                info_gen_code_loss_d = categorical_loss(info_gen_fake_q_out, fake_indices)

                total_g_loss = opt.loss_gan_weight * info_gen_fake_loss + opt.loss_code_weight * info_gen_code_loss_d

            generator_optim.zero_grad()
            scaler.scale(total_g_loss).backward()
            scaler.unscale_(generator_optim)
            torch.nn.utils.clip_grad_norm_(transformer_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(q_infogan.parameters(), 1.0)
            scaler.step(generator_optim)
            scaler.update()

            with amp.autocast(enabled=cuda):
                # x' in Figure 3 (InfoGAN-CR)
                pose_vectorized_lerp1 = pose_vectorized_lerp.clone()
                pose_vectorized_lerp1[:,:,repr_dim: repr_dim + opt.infogan_disc_code + opt.infogan_cont_code] = infogan_code_gen1
                pose_vectorized_lerp1[:,:,repr_dim + fixed_code] = 1
                output_1 = transformer_encoder(pose_vectorized_lerp1, src_mask)

                root_pred1 = output_1[:,:,:root_v_dim].permute(1,0,2)
                quat_pred1 = output_1[:,:,root_v_dim:root_v_dim + local_q_dim].permute(1,0,2)
                contact_pred1 = torch.sigmoid(output_1[:,:,root_v_dim + local_q_dim:root_v_dim+local_q_dim+contact_dim]).permute(1,0,2)
                fake_input1 = torch.cat([root_pred1, quat_pred1, contact_pred1], dim=2).permute(0,2,1)

                # x'' in Figure 3 (InfoGAN-CR)
                pose_vectorized_lerp2 = pose_vectorized_lerp.clone()
                pose_vectorized_lerp2[:,:,repr_dim: repr_dim + opt.infogan_disc_code + opt.infogan_cont_code] = infogan_code_gen2
                pose_vectorized_lerp2[:,:,repr_dim + fixed_code] = 1
                output_2 = transformer_encoder(pose_vectorized_lerp2, src_mask)
                
                root_pred2 = output_2[:,:,:root_v_dim].permute(1,0,2)
                quat_pred2 = output_2[:,:,root_v_dim:root_v_dim + local_q_dim].permute(1,0,2)
                contact_pred2 = torch.sigmoid(output_2[:,:  ,root_v_dim + local_q_dim:root_v_dim+local_q_dim+contact_dim]).permute(1,0,2)
                fake_input2 = torch.cat([root_pred2, quat_pred2, contact_pred2], dim=2).permute(0,2,1)

                fixed_out = torch.ones(current_batch_size, dtype=torch.long, device=device) * fixed_code

                crh_input = torch.cat([fake_input1, fake_input2], dim=1)
                crh_out = infogan_crh(crh_input)
                crh_out_loss = categorical_loss(crh_out, fixed_out)

            cr_optim.zero_grad()
            scaler.scale(opt.loss_crh_weight * crh_out_loss).backward()
            scaler.unscale_(cr_optim)
            torch.nn.utils.clip_grad_norm_(transformer_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(infogan_crh.parameters(), 1.0)
            scaler.step(cr_optim)
            scaler.update()
                                
        # Log
        log_dict = {
            "Train/Loss/Generator GAN Loss": opt.loss_gan_weight * info_gen_fake_loss,
            "Train/Loss/Generator Code Loss": opt.loss_code_weight * info_gen_code_loss_d,
            "Train/Loss/Contrastive Regularization Loss": opt.loss_crh_weight * crh_out_loss,
            "Train/Loss/Total Discriminator Loss": total_d_loss,
            "Train/Loss/Total Generator Loss": total_g_loss
        }

        for k, v in log_dict.items():
            summary_writer.add_scalar(k, v, epoch)
        wandb.log(log_dict)        

        # Save model
        if (epoch % save_interval) == 0:
            ckpt = {'epoch': epoch,
                    'transformer_encoder_state_dict': transformer_encoder.state_dict(),
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
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_walk_dance/', help='path to save pickled processed data')
    parser.add_argument('--batch_size', type=int, default=8 , help='batch size')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or -1 or cpu')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='exp', help='save to project/name')
    parser.add_argument('--save_interval', type=int, default=50, help='Log model after every "save_period" epoch')
    parser.add_argument('--generator_learning_rate', type=float, default=0.001, help='generator_learning_rate')
    parser.add_argument('--discriminator_learning_rate', type=float, default=0.00001, help='discriminator_learning_rate')
    parser.add_argument('--cr_learning_rate', type=float, default=0.001, help='crh_infogan learning rate')
    parser.add_argument('--optim_beta1', type=float, default=0.5, help='optim_beta1')
    parser.add_argument('--optim_beta2', type=float, default=0.9, help='optim_beta2')
    parser.add_argument('--loss_discriminator_weight', type=float, default=1.0, help='loss_gan_discriminator_weight')
    parser.add_argument('--loss_gan_weight', type=float, default=1.0, help='loss_gan_weight')
    parser.add_argument('--loss_code_weight', type=float, default=1.0, help='loss_code_weight')
    parser.add_argument('--infogan_disc_code', type=int, default=5, help='number of discrete codes for InfoGAN')
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
