import os
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
from torch.optim import Adam
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from torch.distributions.normal import Normal
from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import flip_bvh, generate_infogan_code
from rmi.model.network import Decoder, InfoGANDiscriminator, InputEncoder, LSTMNetwork, SinglePoseDiscriminator
from rmi.model.noise_injector import noise_injector
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
    ig_d_code_dim = infogan_code

    pretrained = weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weights)
        # Initializing networks
        state_in = root_v_dim + local_q_dim + contact_dim
        infogan_in = state_in + ig_d_code_dim
        offset_in = root_v_dim + local_q_dim
        target_in = local_q_dim
        state_encoder = InputEncoder(input_dim=infogan_in)
        state_encoder.to(device)

        offset_encoder = InputEncoder(input_dim=offset_in)
        offset_encoder.to(device)


        target_encoder = InputEncoder(input_dim=target_in)
        target_encoder.to(device)

        # LSTM
        lstm_in = state_encoder.out_dim * 3
        print('lstm_hidden', hyp['lstm_hidden'], type(hyp['lstm_hidden']))
        lstm_hidden = int(hyp['lstm_hidden'])
        lstm = LSTMNetwork(input_dim=lstm_in, hidden_dim=lstm_hidden, device=device)
        lstm.to(device)

        # Decoder
        decoder = Decoder(input_dim=lstm_hidden, out_dim=state_in)
        decoder.to(device)

        # Discriminator
        discriminator_in = lafan_dataset.num_joints * 3 * 2 # See Appendix
        sp_discriminator_in = 372
        single_pose_discriminator = SinglePoseDiscriminator(input_dim=sp_discriminator_in, discrete_code_dim=infogan_code)
        single_pose_discriminator.to(device)
        short_discriminator = InfoGANDiscriminator(input_dim=discriminator_in, discrete_code_dim=infogan_code, length=2)
        short_discriminator.to(device)
        long_discriminator = InfoGANDiscriminator(input_dim=discriminator_in, discrete_code_dim=infogan_code, length=5)
        long_discriminator.to(device)
        
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
        
        state_dict_lstm = ckpt['lstm'].float().state_dict()  
        state_dict_lstm = intersect_dicts(state_dict_lstm, lstm.state_dict(), exclude=exclude)  
        lstm.load_state_dict(state_dict_lstm, strict=False)  

        state_dict_decoder = ckpt['decoder'].float().state_dict()  
        state_dict_decoder = intersect_dicts(state_dict_decoder, decoder.state_dict(), exclude=exclude)  
        decoder.load_state_dict(state_dict_decoder, strict=False)  
        
        state_dict_single_pose_discriminator = ckpt['single_pose_dicriminator'].float().state_dict()
        state_dict_single_pose_discriminator = intersect_dicts(state_dict_single_pose_discriminator, single_pose_discriminator.state_dict(), exclude=exclude)
        single_pose_discriminator.load_state_dict(state_dict_single_pose_discriminator, strict=False)

        state_dict_short_discriminator = ckpt['short_discriminator'].float().state_dict()  
        state_dict_short_discriminator = intersect_dicts(state_dict_short_discriminator, short_discriminator.state_dict(), exclude=exclude)  
        short_discriminator.load_state_dict(state_dict_short_discriminator, strict=False)  

        state_dict_long_discriminator = ckpt['long_discriminator'].float().state_dict()  
        state_dict_long_discriminator = intersect_dicts(state_dict_long_discriminator, long_discriminator.state_dict(), exclude=exclude)  
        long_discriminator.load_state_dict(state_dict_long_discriminator, strict=False)  

        # LOGGER.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else : 
        # Initializing networks
        state_in = root_v_dim + local_q_dim + contact_dim
        infogan_in = state_in + ig_d_code_dim
        offset_in = root_v_dim + local_q_dim
        target_in = local_q_dim
                
        state_encoder = InputEncoder(input_dim=infogan_in)
        state_encoder.to(device)

        offset_encoder = InputEncoder(input_dim=offset_in)
        offset_encoder.to(device)


        target_encoder = InputEncoder(input_dim=target_in)
        target_encoder.to(device)

        # LSTM
        lstm_in = state_encoder.out_dim * 3
        print(hyp)
        print('lstm_hidden', hyp['lstm_hidden'], type(hyp['lstm_hidden']))
        lstm_hidden = int(hyp['lstm_hidden'])        
        lstm = LSTMNetwork(input_dim=lstm_in, hidden_dim=lstm_hidden, device=device)
        lstm.to(device)

        # Decoder
        decoder = Decoder(input_dim=lstm_hidden, out_dim=state_in)
        decoder.to(device)

        # Discriminator
        discriminator_in = lafan_dataset.num_joints * 3 * 2 # See Appendix
        sp_discriminator_in = 372
        single_pose_discriminator = SinglePoseDiscriminator(input_dim=sp_discriminator_in, discrete_code_dim=infogan_code)
        single_pose_discriminator.to(device)
        short_discriminator = InfoGANDiscriminator(input_dim=discriminator_in, discrete_code_dim=infogan_code, length=2)
        short_discriminator.to(device)
        long_discriminator = InfoGANDiscriminator(input_dim=discriminator_in, discrete_code_dim=infogan_code, length=5)
        long_discriminator.to(device)
    


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
    for k, v in short_discriminator.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    for k, v in long_discriminator.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    for k, v in single_pose_discriminator.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Try amsgrad True / False 
    # https://tgd.kr/c/deeplearning/19860071
    generator_optimizer = Adam(params=list(state_encoder.parameters()) + 
                                      list(offset_encoder.parameters()) + 
                                      list(target_encoder.parameters()) +
                                      list(lstm.parameters()) +
                                      list(decoder.parameters()),
                                lr=hyp['learning_rate'],
                                betas=(hyp['optim_beta1'], hyp['optim_beta2']),
                                amsgrad=True)

    discriminator_optimizer = Adam(params=list(single_pose_discriminator.parameters()) + 
                                          list(short_discriminator.parameters()) +
                                          list(long_discriminator.parameters()),
                                    lr=hyp['learning_rate'],
                                    betas=(hyp['optim_beta1'], hyp['optim_beta2']),
                                    amsgrad=True)

    pdist = nn.PairwiseDistance(p=2)

    start_epoch = 0
    if pretrained:
        start_epoch = ckpt_state_encoder['epoch'] + 1
        if resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            LOGGER.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt_state_encoder['epoch'], epochs))
            epochs += ckpt_state_encoder['epoch']  # finetune additional epochs

    t0 = time.time()
    scaler = amp.GradScaler(enabled=cuda)

    LOGGER.info(f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):

        # Control transition length
        if lafan_dataset.cur_seq_length < lafan_dataset.max_transition_length:
            lafan_dataset.cur_seq_length =  np.int32(1/lafan_dataset.increase_rate * epoch + lafan_dataset.start_seq_length)

        state_encoder.train()
        offset_encoder.train()
        target_encoder.train()
        lstm.train()
        decoder.train()

        #pbar = enumerate(lafan_data_loader)
        #pbar = tqdm(pbar, total=len(lafan_data_loader))
        
        pbar = tqdm(lafan_data_loader, position=1, desc="Batch")
        generator_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()
        for sampled_batch in pbar: #batch
            div_adv = 0

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
            contact_pred_list = []
            contact_cur_list = []

            real_root_next_list = []
            real_root_cur_list = []
            real_q_next_list = []
            real_q_cur_list = []

            real_contact_next_list = []
            real_contact_cur_list = []
            
            real_root_noise_dist = Normal(loc=torch.zeros(3, device=device), scale=0.1)
            real_quaternion_noise_dist = Normal(loc=torch.zeros(88, device=device), scale=0.03)

            with amp.autocast(enabled=cuda):
                for t in range(training_frames):
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

                    contact_pred_list.append(contact_pred[0])
                    contact_cur_list.append(contact_t)

                    # FK
                    root_pred = root_pred.squeeze()
                    local_q_pred_ = local_q_pred_.squeeze()
                    pos_pred, _ = skeleton.forward_kinematics(root_pred, local_q_pred_, rot_repr='quaternion')
                    pred_list.append(pos_pred)

                    local_q_next = local_q[:,t+1]
                    local_q_next = local_q_next.view(local_q_next.size(0), -1)

                    # Divergence
                    diverging_state_0 = torch.cat([vanilla_state_input, diverging_code_0], dim=1)
                    diverging_state_1 = torch.cat([vanilla_state_input, diverging_code_1], dim=1)

                    h_state_diverging_0 = state_encoder(diverging_state_0)
                    h_state_diverging_1 = state_encoder(diverging_state_1)
                    
                    h_state_diverging_0 = pe(h_state_diverging_0, tta)
                    h_state_diverging_1 = pe(h_state_diverging_1, tta)

                    h_div_0_in = torch.cat([h_state_diverging_0, offset_target], dim=1).unsqueeze(0)
                    h_div_1_in = torch.cat([h_state_diverging_1, offset_target], dim=1).unsqueeze(0)

                    h_div_0_out = lstm(h_div_0_in)
                    h_div_1_out = lstm(h_div_1_in)

                    div_0_h_pred, _ = decoder(h_div_0_out)
                    div_0_local_q_v_pred = div_0_h_pred[:,:,:target_in]
                    div_0_local_q_pred = div_0_local_q_v_pred + local_q_t
                    div_0_root_v_pred = div_0_h_pred[:,:,target_in:]
                    div_0_root_pred = div_0_root_v_pred + root_p_t
                    div_0_root_pred = div_0_root_pred.squeeze()
                    
                    div_1_h_pred, _ = decoder(h_div_1_out)
                    div_1_local_q_v_pred = div_1_h_pred[:,:,:target_in]
                    div_1_local_q_pred = div_1_local_q_v_pred + local_q_t
                    div_1_root_v_pred = div_1_h_pred[:,:,target_in:]
                    div_1_root_pred = div_1_root_v_pred + root_p_t
                    div_1_root_pred = div_1_root_pred.squeeze()

                    noise_multiplier = noise_injector(t, length=training_frames)  # Noise injection
                    div_0_pred = torch.cat([div_0_root_pred, div_0_local_q_pred[0]], dim=1)
                    div_1_pred = torch.cat([div_1_root_pred, div_1_local_q_pred[0]], dim=1)
                    div_adv += torch.mean(pdist(div_0_pred, div_1_pred) * noise_multiplier * hyp['pdist_scale'])

                    real_root_next_list.append(root_p[:,t+1])
                    real_root_cur_list.append(root_p[:,t])
                    real_q_next_list.append(local_q[:,t+1].view(local_q_next.size(0), -1))
                    real_q_cur_list.append(local_q[:,t].view(local_q_next.size(0), -1))
                    real_contact_next_list.append(contact[:,t+1])
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
                next_real_root = torch.stack(real_root_next_list, -1)

                current_quaternion = torch.stack(local_q_cur_list, -1)
                current_real_quaternion = torch.stack(real_q_cur_list, -1)
                current_real_quaternion_noise = torch.clamp(real_quaternion_noise_dist.sample((current_real_root.shape[0], 30)).permute(0,2,1), min=-1, max=1)
                current_real_quaternion += current_real_quaternion_noise
                next_real_quaternion = torch.stack(real_q_next_list, -1)

                current_contact = torch.stack(contact_cur_list, -1)
                pred_contact = torch.stack(contact_pred_list, -1)
                current_real_contact = torch.stack(real_contact_cur_list, -1)
                next_real_contact = torch.stack(real_contact_next_list, -1)


                single_pose_fake_input = torch.cat([start_root, start_quaternion, target_root, target_quaternion, current_root, current_quaternion, current_contact, root_pred, single_pose_pred_quaternion, pred_contact], dim=1)
                single_pose_real_input = torch.cat([start_root, start_quaternion, target_root, target_quaternion, current_real_root, current_real_quaternion, current_real_contact, next_real_root, next_real_quaternion, next_real_contact], dim=1)

                # InfoGAN Loss (maintain LSGAN for original gal V(D,G))
                
                ## Single pose discriminator
                sp_fake_input = single_pose_fake_input.permute(0,2,1).reshape(-1, sp_discriminator_in)
                sp_d_fake_gan_out, _ = single_pose_discriminator(sp_fake_input.detach())
                sp_d_fake_gan_score = sp_d_fake_gan_out[:, 0]

                sp_real_input = single_pose_real_input.permute(0,2,1).reshape(-1, sp_discriminator_in)
                sp_d_real_gan_out, _ = single_pose_discriminator(sp_real_input.detach())
                sp_d_real_gan_score = sp_d_real_gan_out[:, 0]

                sp_d_fake_loss = torch.mean((sp_d_fake_gan_score) ** 2)
                sp_d_real_loss = torch.mean((sp_d_real_gan_score - 1) ** 2)
                sp_d_loss = (sp_d_fake_loss + sp_d_real_loss) / 2.0

                ## Short discriminator
                short_d_fake_gan_out, _ = short_discriminator(fake_input.detach())
                short_d_fake_gan_score = torch.mean(short_d_fake_gan_out[:,0], dim=1)

                short_d_real_gan_out, _ = short_discriminator(real_input)
                short_d_real_gan_score = torch.mean(short_d_real_gan_out[:,0], dim=1)

                short_d_fake_loss = torch.mean((short_d_fake_gan_score) ** 2)  
                short_d_real_loss = torch.mean((short_d_real_gan_score -  1) ** 2)

                short_d_loss = (short_d_fake_loss + short_d_real_loss) / 2.0

                ## Long  discriminator
                long_d_fake_gan_out, _ = long_discriminator(fake_input.detach())
                long_d_fake_gan_score = torch.mean(long_d_fake_gan_out[:,0], dim=1)

                long_d_real_gan_out, _ = long_discriminator(real_input)
                long_d_real_gan_score = torch.mean(long_d_real_gan_out[:,0], dim=1)

                long_d_fake_loss = torch.mean((long_d_fake_gan_score) ** 2)
                long_d_real_loss = torch.mean((long_d_real_gan_score -  1) ** 2)

                long_d_loss = (long_d_fake_loss + long_d_real_loss) / 2.0

                total_d_loss = hyp['loss_sp_discriminator_weight'] * (sp_d_loss) + \
                                hyp['loss_discriminator_weight'] * (short_d_loss + long_d_loss)
             
                # Adversarial
                ## Single pose generator
                sp_g_fake_gan_out, sp_g_fake_q_discrete = single_pose_discriminator(sp_fake_input)
                sp_g_fake_gan_score = sp_g_fake_gan_out[:, 0]
                sp_g_fake_loss = torch.mean((sp_g_fake_gan_score - 1) ** 2)

                fake_indices_expanded = fake_indices.unsqueeze(1).expand(fake_indices.shape[0], training_frames).reshape(sp_g_fake_q_discrete.shape[0])
                sp_disc_code_loss = infogan_disc_loss(sp_g_fake_q_discrete, fake_indices_expanded)

                short_g_fake_gan_out, _ = short_discriminator(fake_input)
                short_g_score = torch.mean(short_g_fake_gan_out[:,0], dim=1)
                short_g_loss = torch.mean((short_g_score -  1) ** 2)

                long_g_fake_gan_out, _ = long_discriminator(fake_input)
                long_g_score = torch.mean(long_g_fake_gan_out[:,0], dim=1)
                long_g_loss = torch.mean((long_g_score -  1) ** 2)

                total_g_loss = hyp['loss_sp_generator_weight'] * sp_g_fake_loss + \
                            hyp['loss_mi_weight'] * sp_disc_code_loss + \
                            hyp['loss_generator_weight'] * (short_g_loss + long_g_loss)
                div_adv = torch.clamp(div_adv, max=0.3)
                loss_total = total_g_loss - div_adv     
            # TOTAL LOSS
            scaler.scale(loss_total).backward()
       
            scaler.scale(total_d_loss).backward()
            scaler.unscale_(discriminator_optimizer)

            scaler.step(discriminator_optimizer)

    


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
        tags = ['train/LOSS/SP Discriminator', 'train/LOSS/ST Discriminator', 'train/OSS/LT Discriminator', 'train/LOSS/Total Discriminator', \
                'train/LOSS/SP Generator', 'train/LOSS/SP Code', \
                    'train/LOSS/ST Generator', 'train/LOSS/LT Generator', 'train/LOSS/Total Generator', "train/Divergence Advantage"]  
        loss_list = [sp_d_loss, short_d_loss, long_d_loss, total_d_loss, \
                    sp_g_fake_loss, sp_disc_code_loss, \
                        short_g_loss, long_g_loss, loss_total, div_adv]
        
        for x, tag in zip(loss_list, tags):
            if loggers['tb']:
                loggers['tb'].add_scalar(tag, x, epoch)
            if loggers['wandb']:
                wandb_logger.log({tag: x}) 

        # Save model
        if (not nosave) or (final_epoch):  # if save

            ckpt = {'epoch': epoch,
                    'state_encoder': deepcopy(de_parallel(state_encoder)).half(),
                    'target_encoder': deepcopy(de_parallel(target_encoder)).half(),
                    'offset_encoder': deepcopy(de_parallel(offset_encoder)).half(),
                    'lstm': deepcopy(de_parallel(lstm)).half(),
                    'decoder': deepcopy(de_parallel(decoder)).half(),
                    'short_discriminator': deepcopy(de_parallel(short_discriminator)).half(),
                    'long_discriminator': deepcopy(de_parallel(long_discriminator)).half(),
                    'single_pose_discriminator': deepcopy(de_parallel(single_pose_discriminator)).half(),
                    'discriminator_optimizer': discriminator_optimizer.state_dict(),
                    'generator_optimizer': generator_optimizer.state_dict(),
                    'wandb_id': wandb_logger.wandb_run.id if loggers['wandb'] else None}

            # Save last, best and delete
            torch.save(ckpt, last)
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
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--data_loader_workers', type=int, default=4, help='data_loader_workers')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save_interval', type=int, default=-1, help='Log model after every "save_period" epoch')
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
