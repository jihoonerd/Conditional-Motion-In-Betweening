import os
import pathlib
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from kpt.model.skeleton import TorchSkeleton
from pymo.parsers import BVHParser
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import flip_bvh, generate_infogan_code
from rmi.model.network import (Decoder, InfoGANDiscriminator, InputEncoder,
                               LSTMNetwork)
from rmi.model.noise_injector import noise_injector
from rmi.model.positional_encoding import PositionalEncoding


def train():
    # Load configuration from yaml
    config = yaml.safe_load(open('./config/config_base.yaml', 'r').read())

    # Set device to use
    gpu_id = config['device']['gpu_id']
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    infogan_code = config['model']['infogan_code']

    # Prepare Directory
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_path = os.path.join('model_weights', time_stamp)
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    
    # Load Skeleton
    parsed = BVHParser().parse(config['data']['skeleton_path']) # Use first bvh info as a reference skeleton.
    skeleton = TorchSkeleton(skeleton=parsed.skeleton, root_name='Hips', device=device)

    # Flip, Load and preprocess data. It utilizes LAFAN1 utilities
    flip_bvh(config['data']['data_dir'])
    lafan_dataset = LAFAN1Dataset(lafan_path=config['data']['data_dir'], train=True, device=device)
    lafan_data_loader = DataLoader(lafan_dataset, batch_size=config['model']['batch_size'], shuffle=True, num_workers=config['data']['data_loader_workers'])

    # Extract dimension from processed data
    root_v_dim = lafan_dataset.root_v_dim
    local_q_dim = lafan_dataset.local_q_dim
    contact_dim = lafan_dataset.contact_dim
    ig_d_code_dim = infogan_code

    # Initializing networks
    state_in = root_v_dim + local_q_dim + contact_dim
    infogan_in = state_in + ig_d_code_dim
    state_encoder = InputEncoder(input_dim=infogan_in)
    state_encoder.to(device)

    offset_in = root_v_dim + local_q_dim
    offset_encoder = InputEncoder(input_dim=offset_in)
    offset_encoder.to(device)

    target_in = local_q_dim
    target_encoder = InputEncoder(input_dim=target_in)
    target_encoder.to(device)

    # LSTM
    lstm_in = state_encoder.out_dim * 3
    lstm = LSTMNetwork(input_dim=lstm_in, hidden_dim=lstm_in*2, device=device)
    lstm.to(device)

    # Decoder
    decoder = Decoder(input_dim=lstm_in*2, out_dim=state_in)
    decoder.to(device)

    discriminator_in = lafan_dataset.num_joints * 3 * 2 # See Appendix
    short_discriminator = InfoGANDiscriminator(input_dim=discriminator_in, discrete_code_dim=ig_d_code_dim, length=3)
    short_discriminator.to(device)
    long_discriminator = InfoGANDiscriminator(input_dim=discriminator_in, discrete_code_dim=ig_d_code_dim, length=10)
    long_discriminator.to(device)

    infogan_disc_loss = nn.NLLLoss()

    pe = PositionalEncoding(dimension=256, max_len=50, device=device)

    generator_optimizer = Adam(params=list(state_encoder.parameters()) + 
                                      list(offset_encoder.parameters()) + 
                                      list(target_encoder.parameters()) +
                                      list(lstm.parameters()) +
                                      list(decoder.parameters()),
                                lr=config['model']['learning_rate'],
                                betas=(config['model']['optim_beta1'], config['model']['optim_beta2']),
                                amsgrad=True)

    discriminator_optimizer = Adam(params=list(short_discriminator.parameters()) +
                                          list(long_discriminator.parameters()),
                                    lr=config['model']['learning_rate'],
                                    betas=(config['model']['optim_beta1'], config['model']['optim_beta2']),
                                    amsgrad=True)

    for epoch in tqdm(range(config['model']['epochs']), position=0, desc="Epoch"):
        state_encoder.train()
        offset_encoder.train()
        target_encoder.train()
        lstm.train()
        decoder.train()

        batch_pbar = tqdm(lafan_data_loader, position=1, desc="Batch")
        for sampled_batch in batch_pbar:
            loss_pos = 0
            loss_quat = 0
            loss_contact = 0
            loss_root = 0

            current_batch_size = len(sampled_batch['global_pos'])

            # state input
            local_q = sampled_batch['local_q'].to(device)
            root_v = sampled_batch['root_v'].to(device)
            contact = sampled_batch['contact'].to(device)
            # offset input
            root_p_offset = sampled_batch['root_p_offset'].to(device)
            local_q_offset = sampled_batch['local_q_offset'].to(device)
            local_q_offset = local_q_offset.view(current_batch_size, -1)
            # target input
            target = sampled_batch['q_target'].to(device)
            target = target.view(current_batch_size, -1)
            # root pos
            root_p = sampled_batch['root_p'].to(device)
            # global pos
            global_pos = sampled_batch['global_pos'].to(device)

            lstm.init_hidden(current_batch_size)
            pred_list = []
            pred_list.append(global_pos[:,0])

            # 3.4: target noise is sampled once per sequence
            target_noise = torch.normal(mean=0, std=config['model']['target_noise'], size=(current_batch_size, 256 * 2), device=device)

            # InfoGAN code (per motion)
            infogan_code_gen, fake_indices = generate_infogan_code(batch_size=current_batch_size, discrete_code_dim=ig_d_code_dim, device=device)

            # Generating Frames. It uses fixed 50 frames of generation for now.
            for t in range(lafan_dataset.cur_seq_length - 1):
                if t  == 0: # if initial frame
                    root_p_t = root_p[:,t]
                    root_v_t = root_v[:,t]
                    local_q_t = local_q[:,t]
                    local_q_t = local_q_t.view(local_q_t.size(0), -1)
                    contact_t = contact[:,t]
                else:
                    root_p_t = root_pred  # Be careful about dimension
                    root_v_t = root_v_pred[0]
                    local_q_t = local_q_pred[0]
                    contact_t = contact_pred[0]

                assert root_p_offset.shape == root_p_t.shape

                # state input
                vanilla_state_input = torch.cat([local_q_t, root_v_t, contact_t], -1)
                state_input = torch.cat([vanilla_state_input, infogan_code_gen], dim=1)
                # offset input
                root_p_offset_t = root_p_offset - root_p_t
                local_q_offset_t = local_q_offset - local_q_t
                offset_input = torch.cat([root_p_offset_t, local_q_offset_t], -1)
                # target input
                target_input = target

                h_state = state_encoder(state_input)
                h_offset = offset_encoder(offset_input)
                h_target = target_encoder(target_input)
                
                # Use positional encoding
                h_state = pe(h_state, t)
                h_offset = pe(h_offset, t)  # (batch size, 256)
                h_target = pe(h_target, t)  # (batch size, 256)

                offset_target = torch.cat([h_offset, h_target], dim=1)
                # Inject noise by scheduling
                noise_multiplier = noise_injector(t, length=lafan_dataset.cur_seq_length)  # Noise injection
                prtbd_offset_target = offset_target + noise_multiplier * target_noise

                # lstm
                h_in = torch.cat([h_state, prtbd_offset_target], dim=1).unsqueeze(0)
                h_out = lstm(h_in)

                # decoder
                h_pred, contact_pred = decoder(h_out)
                local_q_v_pred = h_pred[:,:,:target_in]
                local_q_pred = local_q_v_pred + local_q_t

                local_q_pred_ = local_q_pred.view(local_q_pred.size(0), local_q_pred.size(1), -1, 4)
                local_q_pred_ = local_q_pred_ / torch.norm(local_q_pred_, dim = -1, keepdim = True)

                root_v_pred = h_pred[:,:,target_in:]
                root_pred = root_v_pred + root_p_t

                # FK
                root_pred = root_pred.squeeze()
                local_q_pred_ = local_q_pred_.squeeze()
                pos_pred, _ = skeleton.forward_kinematics(root_pred, local_q_pred_, rot_repr='quaternion')
                pred_list.append(pos_pred)

                # Loss
                pos_next = global_pos[:,t+1]
                local_q_next = local_q[:,t+1]
                local_q_next = local_q_next.view(local_q_next.size(0), -1)
                root_p_next = root_p[:,t+1]
                contact_next = contact[:,t+1]

                # Calculate L1 Norm
                # 3.7.3: We scale all of our losses to be approximately equal on the LaFAN1 dataset 
                # for an untrained network before tuning them with custom weights.
                loss_pos += torch.mean(torch.abs(pos_pred - pos_next) / lafan_dataset.global_pos_std) / lafan_dataset.cur_seq_length
                loss_root += torch.mean(torch.abs(root_pred - root_p_next) / lafan_dataset.global_pos_std[0]) / lafan_dataset.cur_seq_length
                loss_quat += torch.mean(torch.abs(local_q_pred[0] - local_q_next)) / lafan_dataset.cur_seq_length
                loss_contact += torch.mean(torch.abs(contact_pred[0] - contact_next)) / lafan_dataset.cur_seq_length

            # Adversarial
            fake_pos_input = torch.cat([x.reshape(current_batch_size, -1).unsqueeze(-1) for x in pred_list], -1)
            fake_v_input = torch.cat([fake_pos_input[:,:,1:] - fake_pos_input[:,:,:-1], torch.zeros_like(fake_pos_input[:,:,0:1], device=device)], -1)
            fake_input = torch.cat([fake_pos_input, fake_v_input], 1)

            real_pos_input = torch.cat([global_pos[:, i].reshape(current_batch_size, -1).unsqueeze(-1) for i in range(lafan_dataset.cur_seq_length)], -1)
            real_v_input = torch.cat([real_pos_input[:,:,1:] - real_pos_input[:,:,:-1], torch.zeros_like(real_pos_input[:,:,0:1], device=device)], -1)
            real_input = torch.cat([real_pos_input, real_v_input], 1)

            ## Discriminator
            discriminator_optimizer.zero_grad()

            # InfoGAN Loss (maintain LSGAN for original gal V(D,G))
            short_fake_gan_out, _ = short_discriminator(fake_input.detach())
            short_fake_gan_score = torch.mean(short_fake_gan_out[:,0], dim=1)

            short_real_gan_out, _ = short_discriminator(real_input)
            short_real_gan_score = torch.mean(short_real_gan_out[:,0], dim=1)

            short_d_fake_loss = torch.mean((short_fake_gan_score) ** 2)  
            short_d_real_loss = torch.mean((short_real_gan_score -  1) ** 2)

            short_d_loss = (short_d_fake_loss + short_d_real_loss) / 2.0


            long_fake_gan_out, _ = long_discriminator(fake_input.detach())
            long_fake_gan_score = torch.mean(long_fake_gan_out[:,0], dim=1)

            long_real_gan_out, _ = long_discriminator(real_input)
            long_real_gan_score = torch.mean(long_real_gan_out[:,0], dim=1)

            long_d_fake_loss = torch.mean((long_fake_gan_score) ** 2)
            long_d_real_loss = torch.mean((long_real_gan_score -  1) ** 2)

            long_d_loss = (long_d_fake_loss + long_d_real_loss) / 2.0

            total_d_loss = config['model']['loss_discriminator_weight'] * (long_d_loss + short_d_loss)
            total_d_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad()

            loss_total = config['model']['loss_pos_weight'] * loss_pos + \
                         config['model']['loss_quat_weight'] * loss_quat + \
                         config['model']['loss_root_weight'] * loss_root + \
                         config['model']['loss_contact_weight'] * loss_contact
            
            # Adversarial
            short_fake_gan_out, short_fake_q_discrete = short_discriminator(fake_input)
            short_g_score = torch.mean(short_fake_gan_out[:,0], dim=1)
            short_g_loss = torch.mean((short_g_score -  1) ** 2)
            short_disc_code_loss = infogan_disc_loss(short_fake_q_discrete, fake_indices)

            long_fake_gan_out, long_fake_q_discrete = long_discriminator(fake_input)
            long_g_score = torch.mean(long_fake_gan_out[:,0], dim=1)
            long_g_loss = torch.mean((long_g_score -  1) ** 2)
            long_disc_code_loss = infogan_disc_loss(long_fake_q_discrete, fake_indices)

            total_g_loss = config['model']['loss_generator_weight'] * (long_g_loss + long_disc_code_loss + short_g_loss + short_disc_code_loss)
            loss_total += total_g_loss

            # TOTAL LOSS
            loss_total.backward()

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(state_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(offset_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(target_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            generator_optimizer.step()
            batch_pbar.set_postfix({'LOSS': np.round(loss_total.item(), decimals=3)})

        if (epoch + 1) % config['log']['weight_save_interval'] == 0:
            weight_epoch = 'trained_weight_' + str(epoch + 1)
            weight_path = os.path.join(model_path, weight_epoch)
            pathlib.Path(weight_path).mkdir(parents=True, exist_ok=True)
            torch.save(state_encoder.state_dict(), weight_path + '/state_encoder.pkl')
            torch.save(target_encoder.state_dict(), weight_path + '/target_encoder.pkl')
            torch.save(offset_encoder.state_dict(), weight_path + '/offset_encoder.pkl')
            torch.save(lstm.state_dict(), weight_path + '/lstm.pkl')
            torch.save(decoder.state_dict(), weight_path + '/decoder.pkl')
            torch.save(short_discriminator.state_dict(), weight_path + '/short_discriminator.pkl')
            torch.save(long_discriminator.state_dict(), weight_path + '/long_discriminator.pkl')
            if config['model']['save_optimizer']:
                torch.save(generator_optimizer.state_dict(), weight_path + '/generator_optimizer.pkl')
                torch.save(discriminator_optimizer.state_dict(), weight_path + '/discriminator_optimizer.pkl')


if __name__ == '__main__':
    train()
