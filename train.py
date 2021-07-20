import os
import pathlib
import pickle
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from kpt.model.skeleton import TorchSkeleton
from pymo.parsers import BVHParser
from torch.distributions.normal import Normal
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import flip_bvh, generate_infogan_code
from rmi.model.network import (Decoder, Discriminator, InputEncoder,
                               LSTMNetwork, NDiscriminator, QDiscriminator)
from rmi.model.noise_injector import noise_injector
from rmi.model.positional_encoding import PositionalEncoding


def train():
    # Load configuration from yaml
    config = yaml.safe_load(open('./config/config_base.yaml', 'r').read())

    # Set device to use
    # TODO: Support Multi GPU
    gpu_id = config['device']['gpu_id']
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Set number of InfoGAN Code
    infogan_code = config['model']['infogan_code']

    # Prepare Directory
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_path = os.path.join('model_weights', time_stamp)
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    shutil.copyfile('./config/config_base.yaml', os.path.join(model_path, 'exp_config.yaml'))

    # Prepare Tensorboard
    tb_path = os.path.join('tensorboard', time_stamp)
    pathlib.Path(tb_path).mkdir(parents=True, exist_ok=True)
    summarywriter = SummaryWriter(log_dir=tb_path)
    
    # Load Skeleton
    parsed = BVHParser().parse(config['data']['skeleton_path']) # Use first bvh info as a reference skeleton.
    skeleton = TorchSkeleton(skeleton=parsed.skeleton, root_name='Hips', device=device)

    # Flip, Load and preprocess data. It utilizes LAFAN1 utilities
    flip_bvh(config['data']['data_dir'])

    # Load LAFAN Dataset
    saved_train = config['data']['saved_train']
    if os.path.exists(saved_train):
        print("Saved Pickle File Found.")
        with open(saved_train, 'rb') as f:
            lafan_dataset = pickle.load(f)
    else:
        lafan_dataset = LAFAN1Dataset(lafan_path=config['data']['data_dir'], train=True, device=device, start_seq_length=30, cur_seq_length=30, max_transition_length=30)
        with open(saved_train, 'wb') as f:
            pickle.dump(lafan_dataset, f)
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
    lstm_hidden = config['model']['lstm_hidden']
    lstm = LSTMNetwork(input_dim=lstm_in, hidden_dim=lstm_hidden, device=device)
    lstm.to(device)

    # Decoder
    decoder = Decoder(input_dim=lstm_hidden, out_dim=state_in)
    decoder.to(device)

    lstm_discriminator_in = 277

    lstm_discriminator = LSTMNetwork(input_dim=lstm_discriminator_in, hidden_dim=512, device=device)
    lstm_discriminator.to(device)
    lstm_extractor = Discriminator(input_dim=512, out_dim=256)
    lstm_extractor.to(device)
    n_discriminator = NDiscriminator(input_dim=256)
    n_discriminator.to(device)
    q_discriminator = QDiscriminator(input_dim=256, discrete_code_dim=infogan_code)
    q_discriminator.to(device)

    infogan_disc_loss = nn.CrossEntropyLoss()

    pe = PositionalEncoding(dimension=256, max_len=lafan_dataset.max_transition_length, device=device)

    generator_optimizer = Adam(params=list(state_encoder.parameters()) + 
                                      list(offset_encoder.parameters()) + 
                                      list(target_encoder.parameters()) +
                                      list(lstm.parameters()) +
                                      list(decoder.parameters()) + 
                                      list(q_discriminator.parameters()),
                                lr=config['model']['generator_learning_rate'],
                                betas=(config['model']['optim_beta1'], config['model']['optim_beta2']),
                                amsgrad=True)

    discriminator_optimizer = Adam(params=list(lstm_discriminator.parameters()) +
                                          list(lstm_extractor.parameters()) + 
                                          list(n_discriminator.parameters()),
                                    lr=config['model']['discriminator_learning_rate'],
                                    betas=(config['model']['optim_beta1'], config['model']['optim_beta2']),
                                    amsgrad=True)

    pdist = nn.PairwiseDistance(p=2)

    teacher_forcing = config['model']['teacher_forcing']

    for epoch in tqdm(range(config['model']['epochs']), position=0, desc="Epoch"):

        # Control transition length
        if lafan_dataset.cur_seq_length < lafan_dataset.max_transition_length:
            lafan_dataset.cur_seq_length =  np.int32(1/lafan_dataset.increase_rate * epoch + lafan_dataset.start_seq_length)

        teacher_forcing *= config['model']['teacher_forcing_decay']
        teacher_forcing_prob = teacher_forcing

        state_encoder.train()
        offset_encoder.train()
        target_encoder.train()
        lstm.train()
        decoder.train()

        batch_pbar = tqdm(lafan_data_loader, position=1, desc="Batch")
        for sampled_batch in batch_pbar:
            
            # sample from bernoulli by using teacher_forcing_prob
            teacher_forcing_bool = np.random.random() < teacher_forcing_prob

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

            # InfoGAN code (per motion)
            infogan_code_gen, fake_indices = generate_infogan_code(batch_size=current_batch_size, discrete_code_dim=ig_d_code_dim, device=device)
            
            # Generating Frames
            training_frames = torch.randint(low=lafan_dataset.start_seq_length, high=lafan_dataset.cur_seq_length + 1, size=(1,))[0]

            ## EXP
            diverging_code_0 = torch.zeros_like(infogan_code_gen, device=device)
            diverging_code_0[:, 0] = 1
            diverging_code_1 = torch.zeros_like(infogan_code_gen, device=device)
            diverging_code_1[:, 1] = 1

            local_q_pred_list = []
            local_q_cur_list = []
            root_p_pred_list = []
            root_p_cur_list = []
            contact_cur_list = []

            real_root_cur_list = []
            real_q_cur_list = []

            real_contact_cur_list = []
            
            real_root_noise_dist = Normal(loc=torch.zeros(3, device=device), scale=0.1)
            real_quaternion_noise_dist = Normal(loc=torch.zeros(88, device=device), scale=0.03)

            for t in range(training_frames):
                if (t  == 0) or teacher_forcing_bool: # if initial frame
                    root_p_t = root_p[:,t]
                    root_v_t = root_v[:,t]

                    local_q_t = local_q[:,t].view(local_q[:,t].size(0), -1)
                    contact_t = contact[:,t]
                else:
                    root_p_t = root_pred  # Be careful about dimension
                    root_v_t = root_v_pred[0]

                    local_q_t = local_q_pred[0]
                    contact_t = contact_pred[0]

                assert root_p_offset.shape == root_p_t.shape

                # state input
                vanilla_state_input = torch.cat([local_q_t, root_v_t, contact_t], -1)

                # concatenate InfoGAN code
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
                tta = training_frames - t
                h_state = pe(h_state, tta)
                h_offset = pe(h_offset, tta)  # (batch size, 256)
                h_target = pe(h_target, tta)  # (batch size, 256)

                offset_target = torch.cat([h_offset, h_target], dim=1)

                # lstm
                h_in = torch.cat([h_state, offset_target], dim=1).unsqueeze(0)
                h_out = lstm(h_in)

                # decoder
                h_pred, contact_pred = decoder(h_out)

                local_q_v_pred = h_pred[:,:,:target_in]
                
                local_q_pred = local_q_v_pred + local_q_t

                local_q_pred_list.append(local_q_pred[0])
                local_q_cur_list.append(local_q_t)
                
                local_q_pred_ = local_q_pred.view(local_q_pred.size(0), local_q_pred.size(1), -1, 4)
                local_q_pred_ = local_q_pred_ / torch.norm(local_q_pred_, dim = -1, keepdim = True)

                root_v_pred = h_pred[:,:,target_in:]
                root_pred = root_v_pred + root_p_t

                root_p_pred_list.append(root_pred[0])
                root_p_cur_list.append(root_p_t)

                contact_cur_list.append(contact_t)

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

                real_root_cur_list.append(root_p[:,t])
                real_q_cur_list.append(local_q[:,t].view(local_q_next.size(0), -1))
                real_contact_cur_list.append(contact[:,t])

                
            # Adversarial
            fake_pos_input = torch.cat([x.reshape(current_batch_size, -1).unsqueeze(-1) for x in pred_list[:-1]], -1)
            fake_v_input = torch.cat([fake_pos_input[:,:,1:] - fake_pos_input[:,:,:-1], torch.zeros_like(fake_pos_input[:,:,0:1], device=device)], -1)
            fake_input = torch.cat([fake_pos_input, fake_v_input], 1)

            real_pos_input = torch.cat([global_pos[:, i].reshape(current_batch_size, -1).unsqueeze(-1) for i in range(lafan_dataset.cur_seq_length)], -1)
            real_v_input = torch.cat([real_pos_input[:,:,1:] - real_pos_input[:,:,:-1], torch.zeros_like(real_pos_input[:,:,0:1], device=device)], -1)
            real_input = torch.cat([real_pos_input, real_v_input], 1)

            assert fake_input.shape == real_input.shape

            root_pred = torch.stack(root_p_pred_list, -1)
            single_pose_pred_quaternion = torch.stack(local_q_pred_list, -1)
            single_pose_real_quaternion = local_q[:,:lafan_dataset.cur_seq_length].reshape(current_batch_size, lafan_dataset.cur_seq_length, -1).permute(0,2,1)

            assert single_pose_pred_quaternion.shape == single_pose_real_quaternion.shape

            start_root = torch.stack([root_p[:, 0] for _ in range(lafan_dataset.max_transition_length)], dim=2)
            start_quaternion = torch.stack([local_q[:,0,:,:].reshape(current_batch_size, -1) for _ in range(lafan_dataset.max_transition_length)], dim=2)

            target_root = torch.stack([root_p[:, training_frames] for _ in range(lafan_dataset.max_transition_length)], dim=2)
            target_quaternion = torch.stack([local_q[:,training_frames,:,:].reshape(current_batch_size, -1) for _ in range(lafan_dataset.max_transition_length)], dim=2)

            current_root = torch.stack(root_p_cur_list, -1)
            current_real_root = torch.stack(real_root_cur_list, -1)
            current_real_root_noise = real_root_noise_dist.sample((current_real_root.shape[0], 30)).permute(0,2,1)
            current_real_root += current_real_root_noise

            current_quaternion = torch.stack(local_q_cur_list, -1)
            current_real_quaternion = torch.stack(real_q_cur_list, -1)
            current_real_quaternion_noise = torch.clamp(real_quaternion_noise_dist.sample((current_real_root.shape[0], 30)).permute(0,2,1), min=-1, max=1)
            current_real_quaternion += current_real_quaternion_noise

            current_contact = torch.stack(contact_cur_list, -1)
            current_real_contact = torch.stack(real_contact_cur_list, -1)

            single_pose_fake_input = torch.cat([start_root, start_quaternion, target_root, target_quaternion, current_root, current_quaternion, current_contact], dim=1)
            single_pose_real_input = torch.cat([start_root, start_quaternion, target_root, target_quaternion, current_real_root, current_real_quaternion, current_real_contact], dim=1)

            ## Adversarial Discriminator
            discriminator_optimizer.zero_grad()

            ## LSTM Discriminator
            ### Score fake data (->0)
            lstm_discriminator.init_hidden(current_batch_size)
            fake_lstm_disc_out = lstm_discriminator(single_pose_fake_input.permute(2,0,1).detach())[-1]
            d_fake_lstm_out = lstm_extractor(fake_lstm_disc_out)
            d_fake_gan_out = n_discriminator(d_fake_lstm_out)
            d_fake_gan_score = d_fake_gan_out[:, 0]
            lstm_d_fake_loss = 0.5 * torch.mean((d_fake_gan_score) ** 2) * config['model']['loss_discriminator_weight']
            lstm_d_fake_loss.backward()

            ### Score real data (->1)
            lstm_discriminator.init_hidden(current_batch_size)
            real_lstm_disc_out = lstm_discriminator(single_pose_real_input.permute(2,0,1))[-1]
            d_real_lstm_out = lstm_extractor(real_lstm_disc_out)
            d_real_gan_out = n_discriminator(d_real_lstm_out)
            d_real_gan_score = d_real_gan_out[:, 0]
            lstm_d_real_loss = 0.5 * torch.mean((d_real_gan_score - 1) ** 2) * config['model']['loss_discriminator_weight']
            lstm_d_real_loss.backward()

            lstm_d_loss = (lstm_d_fake_loss + lstm_d_real_loss)
            discriminator_optimizer.step()

            generator_optimizer.zero_grad()
            
            # Adversarial Geneartor
            ### Score fake data treated as real (->1)
            lstm_discriminator.init_hidden(current_batch_size)
            fake_lstm_gen_out = lstm_discriminator(single_pose_fake_input.permute(2,0,1))[-1]
            g_fake_lstm_out = lstm_extractor(fake_lstm_gen_out)
            g_fake_gan_out = n_discriminator(g_fake_lstm_out)
            g_fake_gan_score = g_fake_gan_out[:, 0]
            g_fake_loss = torch.mean((g_fake_gan_score - 1) **2)

            q_logit = q_discriminator(g_fake_lstm_out)
            disc_code_loss = infogan_disc_loss(q_logit, fake_indices)

            total_g_loss =  config['model']['loss_pos_weight'] * loss_pos + \
                            config['model']['loss_quat_weight'] * loss_quat + \
                            config['model']['loss_root_weight'] * loss_root + \
                            config['model']['loss_contact_weight'] * loss_contact + \
                            config['model']['loss_generator_weight'] * g_fake_loss + \
                            config['model']['loss_mi_weight'] * disc_code_loss
        
            loss_total = total_g_loss

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

        summarywriter.add_scalar("LOSS/Positional Loss", config['model']['loss_pos_weight'] * loss_pos, epoch + 1)
        summarywriter.add_scalar("LOSS/Quaternion Loss", config['model']['loss_quat_weight'] * loss_quat, epoch + 1)
        summarywriter.add_scalar("LOSS/Root Loss", config['model']['loss_root_weight'] * loss_root, epoch + 1)
        summarywriter.add_scalar("LOSS/Contact Loss", config['model']['loss_contact_weight'] * loss_contact, epoch + 1)

        summarywriter.add_scalar("LOSS/LSTM Discriminator", lstm_d_loss, epoch + 1)
        summarywriter.add_scalar("LOSS/LSTM Generator", config['model']['loss_generator_weight'] * g_fake_loss, epoch + 1)
        summarywriter.add_scalar("LOSS/Discrete Code", config['model']['loss_mi_weight'] * disc_code_loss, epoch + 1)
        summarywriter.add_scalar("LOSS/Total Generator", loss_total, epoch + 1)

        if (epoch + 1) % config['log']['weight_save_interval'] == 0:
            weight_epoch = 'trained_weight_' + str(epoch + 1)
            weight_path = os.path.join(model_path, weight_epoch)
            pathlib.Path(weight_path).mkdir(parents=True, exist_ok=True)
            torch.save(state_encoder.state_dict(), weight_path + '/state_encoder.pkl')
            torch.save(target_encoder.state_dict(), weight_path + '/target_encoder.pkl')
            torch.save(offset_encoder.state_dict(), weight_path + '/offset_encoder.pkl')
            torch.save(lstm.state_dict(), weight_path + '/lstm.pkl')
            torch.save(decoder.state_dict(), weight_path + '/decoder.pkl')
            if config['model']['save_optimizer']:
                torch.save(generator_optimizer.state_dict(), weight_path + '/generator_optimizer.pkl')
                torch.save(discriminator_optimizer.state_dict(), weight_path + '/discriminator_optimizer.pkl')


if __name__ == '__main__':
    train()
