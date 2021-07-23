import os
import pickle
from pathlib import Path
import shutil
from datetime import datetime
import time
import logging 
from copy import deepcopy
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import yaml
from kpt.model.skeleton import TorchSkeleton
from pymo.parsers import BVHParser
from torch.distributions.normal import Normal
from torch.optim import Adam
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import flip_bvh, generate_infogan_code
from rmi.model.network import (Decoder, Discriminator, InfoganCodeEncoder,
                               InputEncoder, LSTMNetwork, NDiscriminator,
                               QDiscriminator)
from rmi.model.positional_encoding import PositionalEncoding

from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from utils.torch_utils import select_device, intersect_dicts, de_parallel
from utils.general import colorstr, get_latest_run, increment_path, check_file


FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

LOGGER = logging.getLogger(__name__)

def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
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
    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)# load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)


    project = opt.project
    save_interval = opt.save_interval

    # Set device to use
    # TODO: Support Multi GPU
    cuda = device.type != 'cpu'
    epochs = opt.epochs

    # Set number of InfoGAN Code

    infogan_code = hyp['infogan_code']
    # Loggers
    loggers = {'wandb': None, 'tb': None}  # loggers dict

    prefix = colorstr('tensorboard: ')
    LOGGER.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
    loggers['tb'] = SummaryWriter(str(save_dir))

    # # Prepare Tensorboard
    # tb_path = os.path.join('tensorboard', time_stamp)
    # pathlib.Path(tb_path).mkdir(parents=True, exist_ok=True)
    # summarywriter = SummaryWriter(log_dir=tb_path)
    # W&B

    run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
    run_id = run_id if resume else None  # start fresh run if transfer learning
    wandb_logger = WandbLogger(opt, save_dir.stem, run_id)
    loggers['wandb'] = wandb_logger.wandb
    if loggers['wandb']:
        weights, epochs, hyp = opt.weights, opt.epochs, hyp  # may update weights, epochs if resuming


    # Load Skeleton
    parsed = BVHParser().parse(opt.skeleton_path) # Use first bvh info as a reference skeleton.
    skeleton = TorchSkeleton(skeleton=parsed.skeleton, root_name='Hips', device=device)

    # Flip, Load and preprocess data. It utilizes LAFAN1 utilities
    flip_bvh(data_path)

    # Load LAFAN Dataset
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    lafan_dataset = LAFAN1Dataset(lafan_path=data_path, processed_data_dir=opt.processed_data_dir, train=True, device=device, start_seq_length=30, cur_seq_length=30, max_transition_length=30)
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

        lstm_hidden = int(hyp['lstm_hidden'])
        infogan_code_encoder = InfoganCodeEncoder(input_dim=infogan_code, out_dim=lstm_hidden)
        infogan_code_encoder.to(device)

        # LSTM
        lstm_in = state_encoder.out_dim * 3
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

        
        #exclude 
        exclude = []
        
        #Load to FP32
        state_dict_state_encoder = ckpt['state_encoder'].float().state_dict()  
        state_dict_state_encoder = intersect_dicts(state_dict_state_encoder, state_encoder.state_dict(), exclude=exclude)  
        state_encoder.load_state_dict(state_dict_state_encoder, strict=False)  

        state_dict_target_encoder = ckpt['target_encoder'].float().state_dict()  
        state_dict_target_encoder = intersect_dicts(state_dict_target_encoder, target_encoder.state_dict(), exclude=exclude)  
        target_encoder.load_state_dict(state_dict_target_encoder, strict=False)  
        
        state_dict_offset_encoder = ckpt['offset_encoder'].float().state_dict() 
        state_dict_offset_encoder = intersect_dicts(state_dict_offset_encoder, offset_encoder.state_dict(), exclude=exclude)  
        offset_encoder.load_state_dict(state_dict_offset_encoder, strict=False)  

        # TODO: Load InfoganCodeEncoder
        
        state_dict_lstm = ckpt['lstm'].float().state_dict()  
        state_dict_lstm = intersect_dicts(state_dict_lstm, lstm.state_dict(), exclude=exclude)  
        lstm.load_state_dict(state_dict_lstm, strict=False)  

        state_dict_decoder = ckpt['decoder'].float().state_dict()  
        state_dict_decoder = intersect_dicts(state_dict_decoder, decoder.state_dict(), exclude=exclude)  
        decoder.load_state_dict(state_dict_decoder, strict=False)  
        
        # TODO: Load newly added network.

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

        lstm_hidden = int(hyp['lstm_hidden'])
        infogan_code_encoder = InfoganCodeEncoder(input_dim=infogan_code, out_dim=lstm_hidden)
        infogan_code_encoder.to(device)

        # LSTM
        lstm_in = state_encoder.out_dim * 3
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
    # TODO: For LSTM Discriminator + newly added network

    # Try amsgrad True / False 
    # https://tgd.kr/c/deeplearning/19860071
    generator_optimizer = Adam(params=list(state_encoder.parameters()) + 
                                      list(offset_encoder.parameters()) + 
                                      list(target_encoder.parameters()) +
                                      list(lstm.parameters()) +
                                      list(decoder.parameters()) + 
                                      list(q_discriminator.parameters()) +
                                      list(infogan_code_encoder.parameters()),
                                lr=hyp['generator_learning_rate'],
                                betas=(hyp['optim_beta1'], hyp['optim_beta2']),
                                amsgrad=True)

    discriminator_optimizer = Adam(params=list(lstm_discriminator.parameters()) +
                                          list(lstm_extractor.parameters()) + 
                                          list(n_discriminator.parameters()),
                                    lr=hyp['discriminator_learning_rate'],
                                    betas=(hyp['optim_beta1'], hyp['optim_beta2']),
                                    amsgrad=True)


    teacher_forcing = hyp['teacher_forcing']

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

        teacher_forcing *= hyp['teacher_forcing_decay']
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
            infogan_code_gen, fake_indices = generate_infogan_code(batch_size=current_batch_size, discrete_code_dim=infogan_code, device=device)
            
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

                if epoch >= hyp['gan_start_epoch']:
                    ## Adversarial Discriminator

                    ## LSTM Discriminator
                    ### Score fake data (->0)
                    lstm_discriminator.init_hidden(current_batch_size)
                    fake_lstm_disc_out = lstm_discriminator(single_pose_fake_input.permute(2,0,1).detach())[-1]
                    d_fake_lstm_out = lstm_extractor(fake_lstm_disc_out)
                    d_fake_gan_out = n_discriminator(d_fake_lstm_out)
                    d_fake_gan_score = d_fake_gan_out[:, 0]
                    lstm_d_fake_loss = 0.5 * torch.mean((d_fake_gan_score) ** 2) * hyp['loss_discriminator_weight']

                    ### Score real data (->1)
                    lstm_discriminator.init_hidden(current_batch_size)
                    real_lstm_disc_out = lstm_discriminator(single_pose_real_input.permute(2,0,1))[-1]
                    d_real_lstm_out = lstm_extractor(real_lstm_disc_out)
                    d_real_gan_out = n_discriminator(d_real_lstm_out)
                    d_real_gan_score = d_real_gan_out[:, 0]
                    lstm_d_real_loss = 0.5 * torch.mean((d_real_gan_score - 1) ** 2) * hyp['loss_discriminator_weight']

                    lstm_d_loss = (lstm_d_fake_loss + lstm_d_real_loss)

                else:
                    lstm_d_loss = 0


                # Adversarial Geneartor
                generator_optimizer.zero_grad()

                ### Score fake data treated as real (->1)
                if epoch >= hyp['gan_start_epoch']:
                    lstm_discriminator.init_hidden(current_batch_size)
                    fake_lstm_gen_out = lstm_discriminator(single_pose_fake_input.permute(2,0,1))[-1]
                    g_fake_lstm_out = lstm_extractor(fake_lstm_gen_out)
                    g_fake_gan_out = n_discriminator(g_fake_lstm_out)
                    g_fake_gan_score = g_fake_gan_out[:, 0]
                    g_fake_loss = torch.mean((g_fake_gan_score - 1) **2)

                    q_logit = q_discriminator(g_fake_lstm_out)
                    disc_code_loss = infogan_disc_loss(q_logit, fake_indices)
                
                else:
                    g_fake_loss = 0
                    disc_code_loss = 0

                total_g_loss =  hyp['loss_pos_weight'] * loss_pos + \
                                hyp['loss_quat_weight'] * loss_quat + \
                                hyp['loss_root_weight'] * loss_root + \
                                hyp['loss_contact_weight'] * loss_contact + \
                                hyp['loss_generator_weight'] * g_fake_loss + \
                                hyp['loss_mi_weight'] * disc_code_loss
            
                loss_total = total_g_loss

            # TOTAL LOSS
            if epoch >= hyp['gan_start_epoch']:
                discriminator_optimizer.zero_grad()
                scaler.scale(lstm_d_loss).backward()
                scaler.step(discriminator_optimizer)

            generator_optimizer.zero_grad()
            scaler.scale(loss_total).backward()
            # Gradient clipping for training stability
            scaler.unscale_(generator_optimizer)
            torch.nn.utils.clip_grad_norm_(state_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(offset_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(target_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            scaler.step(generator_optimizer)
            scaler.update()
        final_epoch = epoch + 1 == epochs
        # Log


        tags = ["Train/LOSS/Positional Loss", 
                "Train/LOSS/Quaternion Loss", 
                "Train/LOSS/Root Loss", 
                "Train/LOSS/Contact Loss", 
                "Train/LOSS/LSTM Discriminator", 
                "Train/LOSS/LSTM Generator", 
                "Train/LOSS/Discrete Code", 
                "Train/LOSS/Total Generator"]  
        loss_list = [
            hyp['loss_pos_weight'] * loss_pos,
            hyp['loss_quat_weight'] * loss_quat,
            hyp['loss_root_weight'] * loss_root,
            hyp['loss_contact_weight'] * loss_contact,
            lstm_d_loss,
            hyp['loss_generator_weight'] * g_fake_loss,
            hyp['loss_mi_weight'] * disc_code_loss,
            loss_total]

        
        for x, tag in zip(loss_list, tags):
            if loggers['tb']:
                loggers['tb'].add_scalar(tag, x, epoch)
            if loggers['wandb']:
                wandb_logger.log({tag: x}) 
        wandb_logger.end_epoch()

        # Save model
        if (not nosave) or (final_epoch):  # if save
            ckpt = {'epoch': epoch,
                    'infogan_code_encoder': infogan_code_encoder.state_dict(),
                    'state_encoder': state_encoder.state_dict(),
                    'target_encoder': target_encoder.state_dict(),
                    'offset_encoder': offset_encoder.state_dict(),
                    'lstm': lstm.state_dict(),
                    'decoder': decoder.state_dict(),
                    'lstm_discriminator': lstm_discriminator.state_dict(),
                    'wandb_id': wandb_logger.wandb_run.id if loggers['wandb'] else None}

            # Save last, best and delete
            # TODO: FIx saving dir
            torch.save(ckpt, 'train-'+str(epoch)+'.pt')
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
    parser.add_argument('--hyp', type=str, default='config/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
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
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.hyp = check_file(opt.hyp)  # check files
        opt.exp_name = opt.exp_name
        # time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # model_path = os.path.join(opt.exp_name, time_stamp)
        # opt.save_dir = Path(model_path).mkdir(parents=True, exist_ok=True)
        opt.save_dir = str(increment_path(Path(opt.project) / opt.exp_name, exist_ok=opt.exist_ok))

    device = select_device(opt.device, batch_size=opt.batch_size)

    train(opt.hyp, opt, device)


def run(**kwargs):
    # Usage: import train; train.run(weights='RMIB_InfoGAN.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
