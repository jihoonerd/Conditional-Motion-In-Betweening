import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from kpt.model.skeleton import TorchSkeleton
from pymo.parsers import BVHParser
from torch.cuda import amp
from torch.distributions.normal import Normal
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import flip_bvh, generate_infogan_code
from rmi.model.network import (Decoder, InfoganCodeEncoder, DInfoGAN, QInfoGAN,
                               InfoGANDiscriminator, InputEncoder, LSTMNetwork)
from rmi.model.positional_encoding import PositionalEncoding
from utils.general import colorstr, get_latest_run, increment_path
from utils.torch_utils import de_parallel, intersect_dicts, select_device
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

LOGGER = logging.getLogger(__name__)

def train(opt,
          device,
          ):

    save_dir, epochs, batch_size, weights, data_path, resume, noval, nosave, = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.weights, opt.data_path, \
        opt.resume, opt.noval, opt.nosave
    # Directories
    save_dir = Path(save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    project = opt.project
    save_interval = opt.save_interval

    # Set device to use
    cuda = device.type != 'cpu'
    epochs = opt.epochs

    # Set number of InfoGAN Code
    infogan_cont_code = opt.infogan_cont_code
    infogan_disc_code = opt.infogan_disc_code
    # Loggers
    loggers = {'wandb': None, 'tb': None}  # loggers dict

    prefix = colorstr('tensorboard: ')
    LOGGER.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
    loggers['tb'] = SummaryWriter(str(save_dir))

    run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
    run_id = run_id if resume else None  # start fresh run if transfer learning
    wandb_logger = WandbLogger(opt, save_dir.stem, run_id)
    loggers['wandb'] = wandb_logger.wandb
    if loggers['wandb']:
        weights, epochs = opt.weights, opt.epochs  # may update weights, epochs if resuming


    # Load Skeleton
    parsed = BVHParser().parse(opt.skeleton_path) # Use first bvh info as a reference skeleton.
    skeleton = TorchSkeleton(skeleton=parsed.skeleton, root_name='Hips', device=device)

    # Flip, Load and preprocess data. It utilizes LAFAN1 utilities
    flip_bvh(data_path)

    # Load LAFAN Dataset
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    lafan_dataset = LAFAN1Dataset(lafan_path=data_path, processed_data_dir=opt.processed_data_dir, train=True, target_action=['jump', 'run'], device=device, start_seq_length=30, cur_seq_length=30, max_transition_length=30)
    lafan_dataset.global_pos_std = lafan_dataset.data['global_pos_std']
    lafan_data_loader = DataLoader(lafan_dataset, batch_size=batch_size, shuffle=True, num_workers=opt.data_loader_workers)

    # Extract dimension from processed data
    root_v_dim = lafan_dataset.root_v_dim
    local_q_dim = lafan_dataset.local_q_dim
    contact_dim = lafan_dataset.contact_dim


    pretrained = weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weights)
        # Initializing networks
        state_in = root_v_dim + local_q_dim + contact_dim
        offset_in = root_v_dim + local_q_dim
        target_in = local_q_dim
        state_encoder = InputEncoder(input_dim=state_in)
        state_encoder.to(device)

        offset_encoder = InputEncoder(input_dim=offset_in)
        offset_encoder.to(device)

        target_encoder = InputEncoder(input_dim=target_in)
        target_encoder.to(device)

        lstm_hidden = int(opt.lstm_hidden)
        infogan_code_encoder = InfoganCodeEncoder(input_dim=infogan_cont_code + infogan_disc_code, out_dim=lstm_hidden)
        infogan_code_encoder.to(device)

        # LSTM
        lstm_in = state_encoder.out_dim * 3
        lstm = LSTMNetwork(input_dim=lstm_in, hidden_dim=lstm_hidden, device=device)
        lstm.to(device)

        # Decoder
        decoder = Decoder(input_dim=lstm_hidden, out_dim=state_in)
        decoder.to(device)

        # LSTM Discriminator
        discriminator_in = 277
        infogan_discriminator = InfoGANDiscriminator(input_dim=discriminator_in, hidden_dim=256)
        infogan_discriminator.to(device)

        # DInfoGAN
        d_infogan = DInfoGAN(input_dim=30)
        d_infogan.to(device)
        # QInfoGAN
        q_infogan = QInfoGAN(input_dim=30, discrete_code_dim=infogan_disc_code, continuous_code_dim=infogan_cont_code)
        q_infogan.to(device)

        
        #exclude 
        exclude = []
        
        #Load to FP32
        state_dict_state_encoder = ckpt['state_encoder']
        state_dict_state_encoder = intersect_dicts(state_dict_state_encoder, state_encoder.state_dict(), exclude=exclude)  
        state_encoder.load_state_dict(state_dict_state_encoder, strict=False)  

        state_dict_target_encoder = ckpt['target_encoder']
        state_dict_target_encoder = intersect_dicts(state_dict_target_encoder, target_encoder.state_dict(), exclude=exclude)  
        target_encoder.load_state_dict(state_dict_target_encoder, strict=False)  

        state_dict_offset_encoder = ckpt['offset_encoder']
        state_dict_offset_encoder = intersect_dicts(state_dict_offset_encoder, offset_encoder.state_dict(), exclude=exclude)  
        offset_encoder.load_state_dict(state_dict_offset_encoder, strict=False)  

        state_infogan_code_encoder = ckpt['infogan_code_encoder']
        state_infogan_code_encoder = intersect_dicts(state_infogan_code_encoder, infogan_code_encoder.state_dict(), exclude=exclude)
        infogan_code_encoder.load_state_dict(state_infogan_code_encoder, strict=False)
        
        state_dict_lstm = ckpt['lstm']
        state_dict_lstm = intersect_dicts(state_dict_lstm, lstm.state_dict(), exclude=exclude)  
        lstm.load_state_dict(state_dict_lstm, strict=False)  

        state_dict_decoder = ckpt['decoder']
        state_dict_decoder = intersect_dicts(state_dict_decoder, decoder.state_dict(), exclude=exclude)  
        decoder.load_state_dict(state_dict_decoder, strict=False)  
        
        state_infogan_discriminator = ckpt['infogan_discriminator']
        state_infogan_discriminator = intersect_dicts(state_infogan_discriminator, infogan_discriminator.state_dict(), exclude=exclude)
        infogan_discriminator.load_state_dict(state_infogan_discriminator, strict=False)

        state_d_infogan = ckpt['d_infogan']
        state_d_infogan = intersect_dicts(state_d_infogan, d_infogan.state_dict(), exclude=exclude)
        d_infogan.load_state_dict(state_d_infogan, strict=False)

        state_q_infogan = ckpt['q_infogan']
        state_q_infogan = intersect_dicts(state_q_infogan, q_infogan.state_dict(), exclude=exclude)
        q_infogan.load_state_dict(state_q_infogan, strict=False)
        infogan_cont_code = ckpt['cont_code']
        infogan_disc_code = ckpt['disc_code']
        epoch = ckpt['epoch']
        # LOGGER.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else : 
        # Initializing networks
        state_in = root_v_dim + local_q_dim + contact_dim
        offset_in = root_v_dim + local_q_dim
        target_in = local_q_dim
                
        state_encoder = InputEncoder(input_dim=state_in)
        state_encoder.to(device)

        offset_encoder = InputEncoder(input_dim=offset_in)
        offset_encoder.to(device)

        target_encoder = InputEncoder(input_dim=target_in)
        target_encoder.to(device)

        lstm_hidden = int(opt.lstm_hidden)
        infogan_code_encoder = InfoganCodeEncoder(input_dim=infogan_cont_code+infogan_disc_code, out_dim=lstm_hidden)
        infogan_code_encoder.to(device)

        # LSTM
        lstm_in = state_encoder.out_dim * 3
        lstm = LSTMNetwork(input_dim=lstm_in, hidden_dim=lstm_hidden, device=device)
        lstm.to(device)

        # Decoder
        decoder = Decoder(input_dim=lstm_hidden, out_dim=state_in)
        decoder.to(device)

        # Discriminator
        discriminator_in = 277
        infogan_discriminator = InfoGANDiscriminator(input_dim=discriminator_in, hidden_dim=256)
        infogan_discriminator.to(device)
        
        # DInfoGAN
        d_infogan = DInfoGAN(input_dim=30)
        d_infogan.to(device)
        # QInfoGAN
        q_infogan = QInfoGAN(input_dim=30, discrete_code_dim=infogan_disc_code, continuous_code_dim=infogan_cont_code)
        q_infogan.to(device)

    infogan_disc_code_loss = nn.CrossEntropyLoss()
    infogan_cont_code_loss = nn.GaussianNLLLoss(full=True)

    pe = PositionalEncoding(dimension=256, max_len=lafan_dataset.max_transition_length, device=device)

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)

    for k, v in state_encoder.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    for k, v in target_encoder.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    for k, v in offset_encoder.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    for k, v in infogan_code_encoder.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False            
    for k, v in lstm.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    for k, v in decoder.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    for k, v in infogan_discriminator.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    for k, v in q_infogan.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    for k, v in d_infogan.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
 



    # Try amsgrad True / False 
    # https://tgd.kr/c/deeplearning/19860071
    generator_optimizer = Adam(params=list(state_encoder.parameters()) + 
                                      list(offset_encoder.parameters()) + 
                                      list(target_encoder.parameters()) +
                                      list(infogan_code_encoder.parameters()) +
                                      list(lstm.parameters()) +
                                      list(decoder.parameters()) + 
                                      list(q_infogan.parameters()),
                                lr=opt.generator_learning_rate,
                                betas=(opt.optim_beta1, opt.optim_beta2),
                                amsgrad=True)

    discriminator_optimizer = Adam(params=list(infogan_discriminator.parameters()) +
                                          list(d_infogan.parameters()),
                                    lr=opt.discriminator_learning_rate,
                                    betas=(opt.optim_beta1, opt.optim_beta2),
                                    amsgrad=True)


    teacher_forcing = opt.teacher_forcing

    start_epoch = 0
    if pretrained:
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            LOGGER.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

    t0 = time.time()
    scaler = amp.GradScaler(enabled=cuda)

    LOGGER.info(f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):

        # Control transition length
        if lafan_dataset.cur_seq_length < lafan_dataset.max_transition_length:
            lafan_dataset.cur_seq_length =  np.int32(1/lafan_dataset.increase_rate * epoch + lafan_dataset.start_seq_length)

        teacher_forcing *= opt.teacher_forcing_decay
        teacher_forcing_prob = teacher_forcing

        #pbar = enumerate(lafan_data_loader)
        #pbar = tqdm(pbar, total=len(lafan_data_loader))
        
        pbar = tqdm(lafan_data_loader, position=1, desc="Batch")

        for sampled_batch in pbar: #batch
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
            infogan_code_gen, fake_indices = generate_infogan_code(batch_size=current_batch_size, discrete_code_dim=infogan_disc_code, continuous_code_dim=infogan_cont_code, device=device)
            
            lstm.h[0] = infogan_code_encoder(infogan_code_gen.to(torch.float))
            assert lstm.h[0].shape == (current_batch_size, lstm_hidden)

            # Generating Frames
            training_frames = torch.randint(low=lafan_dataset.start_seq_length, high=lafan_dataset.cur_seq_length + 1, size=(1,))[0]

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

                
            with amp.autocast(enabled=cuda):
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
                    state_input = torch.cat([local_q_t, root_v_t, contact_t], -1)

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

                    local_q_next = local_q[:,t+1]
                    local_q_next = local_q_next.view(local_q_next.size(0), -1)
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
                # current_real_root_noise = real_root_noise_dist.sample((current_real_root.shape[0], 30)).permute(0,2,1)
                # current_real_root += current_real_root_noise

                current_quaternion = torch.stack(local_q_cur_list, -1)
                current_real_quaternion = torch.stack(real_q_cur_list, -1)
                # current_real_quaternion_noise = torch.clamp(real_quaternion_noise_dist.sample((current_real_root.shape[0], 30)).permute(0,2,1), min=-1, max=1)
                # current_real_quaternion += current_real_quaternion_noise

                current_contact = torch.stack(contact_cur_list, -1)
                current_real_contact = torch.stack(real_contact_cur_list, -1)

                single_pose_fake_input = torch.cat([start_root, start_quaternion, target_root, target_quaternion, current_root, current_quaternion, current_contact], dim=1)
                single_pose_real_input = torch.cat([start_root, start_quaternion, target_root, target_quaternion, current_real_root, current_real_quaternion, current_real_contact], dim=1)

                if epoch >= opt.gan_start_epoch:
                    ## Adversarial Discriminator
                    discriminator_optimizer.zero_grad()
                    infogan_disc_fake_gan_out = infogan_discriminator(single_pose_fake_input.detach()).squeeze()
                    infogan_disc_fake_d_out = d_infogan(infogan_disc_fake_gan_out)
                    info_disc_fake_loss = torch.mean((infogan_disc_fake_d_out) ** 2)

                    infogan_disc_real_gan_out = infogan_discriminator(single_pose_real_input).squeeze()
                    infogan_disc_real_d_out = d_infogan(infogan_disc_real_gan_out)
                    info_disc_real_loss = torch.mean((infogan_disc_real_d_out -  1) ** 2)

                    info_d_loss = (info_disc_fake_loss + info_disc_real_loss) / 2.0
                else:
                    info_d_loss = 0


                # Adversarial Geneartor
                generator_optimizer.zero_grad()

                ### Score fake data treated as real (->1)
                if epoch >= opt.gan_start_epoch:
                    info_gen_fake_gan_out = infogan_discriminator(single_pose_fake_input).squeeze()
                    info_gen_fake_d_out = d_infogan(info_gen_fake_gan_out)
                    info_gen_fake_loss = torch.mean((info_gen_fake_d_out - 1) ** 2)

                    info_gen_fake_q_out, info_gen_fake_q_mu, info_gen_fake_q_var = q_infogan(info_gen_fake_gan_out)
                    if infogan_disc_code != 0 :
                        info_gen_code_loss_d = infogan_disc_code_loss(info_gen_fake_q_out, fake_indices)
                    else :
                        info_gen_code_loss_d = 0 
                    if infogan_cont_code != 0 :
                        info_gen_code_loss_c = infogan_cont_code_loss(infogan_code_gen[:, infogan_disc_code:], info_gen_fake_q_mu, info_gen_fake_q_var)
                    else : 
                        info_gen_code_loss_c = 0
                else:
                    info_gen_fake_loss = 0
                    info_gen_code_loss_d = 0
                    info_gen_code_loss_c = 0

                total_g_loss =  opt.loss_pos_weight * loss_pos + \
                                opt.loss_quat_weight * loss_quat + \
                                opt.loss_root_weight * loss_root + \
                                opt.loss_contact_weight * loss_contact + \
                                opt.loss_generator_weight * info_gen_fake_loss + \
                                opt.loss_mi_weight * (info_gen_code_loss_d + info_gen_code_loss_c)
            

            # TOTAL LOSS
            if epoch >= opt.gan_start_epoch:
                discriminator_optimizer.zero_grad()
                total_d_loss = opt.loss_discriminator_weight * info_d_loss
                scaler.scale(total_d_loss).backward()
                scaler.step(discriminator_optimizer)

            generator_optimizer.zero_grad()
            scaler.scale(total_g_loss).backward()
            # Gradient clipping for training stability
            scaler.unscale_(generator_optimizer)
            torch.nn.utils.clip_grad_norm_(state_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(offset_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(target_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(infogan_code_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(q_infogan.parameters(), 1.0)
            scaler.step(generator_optimizer)
            scaler.update()
        final_epoch = epoch + 1 == epochs

        # Log
        log_dict = {
            "Train/LOSS/Positional Loss": opt.loss_pos_weight * loss_pos, 
            "Train/LOSS/Quaternion Loss": opt.loss_quat_weight * loss_quat, 
            "Train/LOSS/Root Loss": opt.loss_root_weight * loss_root, 
            "Train/LOSS/Contact Loss": opt.loss_contact_weight * loss_contact, 
            "Train/LOSS/InfoGAN Discriminator": opt.loss_discriminator_weight * info_d_loss, 
            "Train/LOSS/InfoGAN Generator": opt.loss_generator_weight * info_gen_fake_loss,
            "Train/LOSS/Discrete Code": opt.loss_mi_weight * info_gen_code_loss_d,
            "Train/LOSS/Continuous Code": opt.loss_mi_weight * info_gen_code_loss_c,
            "Train/LOSS/Total Generator": total_g_loss,
        }

        for k, v in log_dict.items():
            if loggers['tb']:
                loggers['tb'].add_scalar(k, v, epoch)
            if loggers['wandb']:
                wandb_logger.log({k: v})        
        wandb_logger.end_epoch()

        # Save model
        if (not nosave) or (final_epoch):  # if save
            ckpt = {'epoch': epoch,
                    'state_encoder': state_encoder.state_dict(),
                    'target_encoder': target_encoder.state_dict(),
                    'offset_encoder': offset_encoder.state_dict(),
                    'infogan_code_encoder': infogan_code_encoder.state_dict(),
                    'lstm': lstm.state_dict(),
                    'decoder': decoder.state_dict(),
                    'infogan_discriminator': infogan_discriminator.state_dict(),
                    'q_infogan': q_infogan.state_dict(),
                    'd_infogan': d_infogan.state_dict(),
                    'disc_code': infogan_disc_code,
                    'cont_code': infogan_cont_code,
                    'wandb_id': wandb_logger.wandb_run.id if loggers['wandb'] else None}
            if (epoch % save_interval) == 0:
                torch.save(ckpt, os.path.join(wdir, f'train-{epoch}.pt'))

            if loggers['wandb']:
                if ((epoch + 1) % opt.save_interval == 0 and not epochs) and opt.save_interval != -1:
                    wandb_logger.log_model(last.parent, opt, epoch)
            del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------            
    LOGGER.info(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')
    wandb_logger.finish_run()
    torch.cuda.empty_cache()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--data_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH', help='dataset path')
    parser.add_argument('--skeleton_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh', help='dataset path')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data/', help='dataset path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--data_loader_workers', type=int, default=4, help='data_loader_workers')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save_interval', type=int, default=1, help='Log model after every "save_period" epoch')
    ##Model hyper parameters ##
    parser.add_argument('--lstm_hidden', type=int, default=1024, help='lstm_hidden layers')
    parser.add_argument('--loss_mi_weight', type=float, default=0.1, help='loss_mi_weight')
    parser.add_argument('--loss_discriminator_weight', type=float, default=1.0, help='loss_discriminator_weight')
    parser.add_argument('--loss_generator_weight', type=float, default=1.0, help='loss_generator_weight')
    parser.add_argument('--save_optimizer', type=bool, default=False, help='bool save_optimizer')
    parser.add_argument('--discriminator_learning_rate', type=float, default=0.0001, help='discriminator_learning_rate')
    parser.add_argument('--generator_learning_rate', type=float, default=0.001, help='generator_learning_rate')
    parser.add_argument('--optim_beta1', type=float, default=0.5, help='optim_beta1')
    parser.add_argument('--optim_beta2', type=float, default=0.9, help='optim_beta2')
    parser.add_argument('--gan_start_epoch', type=int, default=0, help='gan_start_epoch')
    parser.add_argument('--infogan_cont_code', type=int, default=5, help='# of infogan_cont_code')
    parser.add_argument('--infogan_disc_code', type=int, default=10, help='# of infogan_disc_code')
    parser.add_argument('--loss_pos_weight', type=float, default=0.5, help='loss_pos_weight')
    parser.add_argument('--loss_quat_weight', type=float, default=1.0, help='loss_quat_weight')
    parser.add_argument('--loss_root_weight', type=float, default=1.0, help='loss_root_weight')
    parser.add_argument('--loss_contact_weight', type=float, default=0.1, help='loss_contact_weight')
    parser.add_argument('--teacher_forcing', type=float, default=0, help='teacher_forcing')    
    parser.add_argument('--teacher_forcing_decay', type=float, default=0.9, help='teacher_forcing_decay')
    ########################
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):
    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.weights, opt.resume = ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.exp_name = opt.exp_name
        # time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # model_path = os.path.join(opt.exp_name, time_stamp)
        # opt.save_dir = Path(model_path).mkdir(parents=True, exist_ok=True)
        opt.save_dir = str(increment_path(Path(opt.project) / opt.exp_name, exist_ok=opt.exist_ok))

    device = select_device(opt.device, batch_size=opt.batch_size)

    train(opt, device)

def run(**kwargs):
    # Usage: import train; train.run(weights='RMIB_InfoGAN.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
