import os
import pathlib
import shutil
from datetime import datetime
import logging 
from copy import deepcopy


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
from utils.general import colorstr
LOGGER = logging.getLogger(__name__)

def train():
    # Load configuration and hyperparameter setting from yaml
    config = yaml.safe_load(open('./config/config_base.yaml', 'r').read())    
    hyp = config['model']
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    project = config['project']
    save_interval = config['log']['weight_save_interval']
    # Set device to use
    # TODO: Support Multi GPU
    gpu_id = config['device']['gpu_id']
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    cuda = device.type != 'cpu'
    total_epochs = config['model']['total_epochs']
    # Set number of InfoGAN Code
    infogan_code = config['model']['infogan_code']

    # Prepare Directory    
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_path = os.path.join('model_weights', time_stamp)
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    shutil.copyfile('./config/config_base.yaml', os.path.join(model_path, 'exp_config.yaml'))

    state_encoder_path = model_path + '/state_encoder.pt'
    target_encoder_path = model_path + '/target_encoder.pt'
    offset_encoder_path = model_path + '/offset_encoder.pt'
    lstm_path = model_path + '/lstm.pt'
    decoder_path = model_path + '/decoder.pt'
    short_discriminator_path = model_path + '/short_discriminator.pt'
    long_discriminator_path = model_path +  '/long_discriminator.pt'
    single_pose_discriminator_path = model_path + '/single_pose_discriminator.pt'
    generator_optimizer_path = model_path + '/generator_optimizer.pt'
    discriminator_optimizer_path = model_path + '/discriminator_optimizer.pt'

    # Loggers
    loggers = {'wandb': None, 'tb': None}  # loggers dict

    prefix = colorstr('tensorboard: ')
    LOGGER.info(f"{prefix}Start with 'tensorboard --logdir {project}', view at http://localhost:6006/")
    loggers['tb'] = SummaryWriter(log_dir=tb_path)

    # # Prepare Tensorboard
    # tb_path = os.path.join('tensorboard', time_stamp)
    # pathlib.Path(tb_path).mkdir(parents=True, exist_ok=True)
    # summarywriter = SummaryWriter(log_dir=tb_path)
    # W&B
    run_id = torch.load(model_path).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
    run_id = run_id if resume else None  # start fresh run if transfer learning
    wandb_logger = WandbLogger(opt, model_path.stem, run_id, data_dict)
    loggers['wandb'] = wandb_logger.wandb
    if loggers['wandb']:
        data_dict = wandb_logger.data_dict
        weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # may update weights, epochs if resuming


    # Load Skeleton
    parsed = BVHParser().parse(config['data']['skeleton_path']) # Use first bvh info as a reference skeleton.
    skeleton = TorchSkeleton(skeleton=parsed.skeleton, root_name='Hips', device=device)

    # Flip, Load and preprocess data. It utilizes LAFAN1 utilities
    flip_bvh(config['data']['data_dir'])

    # Load LAFAN Dataset
    lafan_dataset = LAFAN1Dataset(lafan_path=config['data']['data_dir'], train=True, device=device, start_seq_length=30, cur_seq_length=30, max_transition_length=30)
    lafan_data_loader = DataLoader(lafan_dataset, batch_size=config['model']['batch_size'], shuffle=True, num_workers=config['data']['data_loader_workers'])

    # Extract dimension from processed data
    root_v_dim = lafan_dataset.root_v_dim
    local_q_dim = lafan_dataset.local_q_dim
    contact_dim = lafan_dataset.contact_dim
    ig_d_code_dim = infogan_code

    # Initializing networks
    state_in = root_v_dim + local_q_dim + contact_dim
    infogan_in = state_in + ig_d_code_dim
    offset_in = root_v_dim + local_q_dim
    target_in = local_q_dim

    # LSTM
    lstm_in = state_encoder.out_dim * 3
    lstm_hidden = config['model']['lstm_hidden']


    discriminator_in = lafan_dataset.num_joints * 3 * 2 # See Appendix
    sp_discriminator_in = 372

    pretrained = weights.endswith('.pt')
    if pretrained:
        ckpt_state_encoder = torch.load(weights_state_encoder)
        ckpt_target_encoder = torch.load(weights_target_encoder)
        ckpt_offset_encoder = torch.load(weights_offset_encoder)
        ckpt_lstm = torch.load(weights_lstm)
        ckpt_decoder = torch.load(weights_decoder)
        ckpt_short_discriminator = torch.load(weights_short_discriminator)
        ckpt_long_discriminator = torch.load(weights_long_discriminator)
        ckpt_single_pose_discriminator = torch.load(weights_single_pose_discriminator)

        state_encoder = InputEncoder(input_dim=infogan_in)
        state_encoder.to(device)

        offset_encoder = InputEncoder(input_dim=offset_in)
        offset_encoder.to(device)


        target_encoder = InputEncoder(input_dim=target_in)
        target_encoder.to(device)

        # LSTM
        lstm = LSTMNetwork(input_dim=lstm_in, hidden_dim=lstm_hidden, device=device)
        lstm.to(device)

        # Decoder
        decoder = Decoder(input_dim=lstm_hidden, out_dim=state_in)
        decoder.to(device)

        # Discriminator
        single_pose_discriminator = SinglePoseDiscriminator(input_dim=sp_discriminator_in, discrete_code_dim=infogan_code)
        single_pose_discriminator.to(device)
        short_discriminator = InfoGANDiscriminator(input_dim=discriminator_in, discrete_code_dim=infogan_code, length=2)
        short_discriminator.to(device)
        long_discriminator = InfoGANDiscriminator(input_dim=discriminator_in, discrete_code_dim=infogan_code, length=5)
        long_discriminator.to(device)
        #Load to FP32
        state_dict_state_encoder = ckpt_state_encoder['model'].float().state_dict()  
        state_dict_state_encoder = intersect_dicts(state_dict_state_encoder, state_encoder.state_dict(), exclude=exclude)  
        state_encoder.load_state_dict(state_dict_state_encoder, strict=False)  

        state_dict_target_encoder = ckpt_target_encoder['model'].float().state_dict()  
        state_dict_target_encoder = intersect_dicts(state_dict_target_encoder, target_encoder.state_dict(), exclude=exclude)  
        target_encoder.load_state_dict(state_dict_target_encoder, strict=False)  
        
        state_dict_offset_encoder = ckpt_offset_encoder['model'].float().state_dict() 
        state_dict_offset_encoder = intersect_dicts(state_dict_offset_encoder, offset_encoder.state_dict(), exclude=exclude)  
        offset_encoder.load_state_dict(state_dict_offset_encoder, strict=False)  
        
        state_dict_lstm = ckpt_lstm['model'].float().state_dict()  
        state_dict_lstm = intersect_dicts(state_dict_lstm, lstm.state_dict(), exclude=exclude)  
        lstm.load_state_dict(state_dict_lstm, strict=False)  

        state_dict_decoder = ckpt_decoder['model'].float().state_dict()  
        state_dict_decoder = intersect_dicts(state_dict_decoder, decoder.state_dict(), exclude=exclude)  
        decoder.load_state_dict(state_dict_decoder, strict=False)  
        
        state_dict_single_pose_discriminator = ckpt_single_pose_discriminator['model'].float().state_dict()
        state_dict_single_pose_discriminator = intersect_dicts(state_dict_single_pose_discriminator, single_pose_discriminator.state_dict(), exclude=exclude)
        single_pose_discriminator.load_state_dict(state_dict_single_pose_discriminator, strict=False)

        state_dict_short_discriminator = ckpt_short_discriminator['model'].float().state_dict()  
        state_dict_short_discriminator = intersect_dicts(state_dict_short_discriminator, short_discriminator.state_dict(), exclude=exclude)  
        short_discriminator.load_state_dict(state_dict_short_discriminator, strict=False)  

        state_dict_decoder = ckpt_decoder['model'].float().state_dict()  
        state_dict_decoder = intersect_dicts(state_dict_decoder, decoder.state_dict(), exclude=exclude)  
        decoder.load_state_dict(state_dict_decoder, strict=False)  


        #exclude 
        exclude = []
        
        # LOGGER.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else : 
        state_encoder = InputEncoder(input_dim=infogan_in)
        state_encoder.to(device)

        offset_encoder = InputEncoder(input_dim=offset_in)
        offset_encoder.to(device)


        target_encoder = InputEncoder(input_dim=target_in)
        target_encoder.to(device)

        # LSTM
        lstm = LSTMNetwork(input_dim=lstm_in, hidden_dim=lstm_hidden, device=device)
        lstm.to(device)

        # Decoder
        decoder = Decoder(input_dim=lstm_hidden, out_dim=state_in)
        decoder.to(device)

        # Discriminator
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
                                lr=config['model']['learning_rate'],
                                betas=(config['model']['optim_beta1'], config['model']['optim_beta2']),
                                amsgrad=True)

    discriminator_optimizer = Adam(params=list(single_pose_discriminator.parameters()) + 
                                          list(short_discriminator.parameters()) +
                                          list(long_discriminator.parameters()),
                                    lr=config['model']['learning_rate'],
                                    betas=(config['model']['optim_beta1'], config['model']['optim_beta2']),
                                    amsgrad=True)

    pdist = nn.PairwiseDistance(p=2)

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

        state_encoder.train()
        offset_encoder.train()
        target_encoder.train()
        lstm.train()
        decoder.train()

        pbar = enumerate(lafan_data_loader)
        pbar = tqdm(pbar, total=len(lafan_data_loader))
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
                    div_adv += torch.mean(pdist(div_0_pred, div_1_pred) * noise_multiplier * config['model']['pdist_scale'])

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

                total_d_loss = config['model']['loss_sp_discriminator_weight'] * (sp_d_loss) + \
                                config['model']['loss_discriminator_weight'] * (short_d_loss + long_d_loss)
             
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

                total_g_loss = config['model']['loss_sp_generator_weight'] * sp_g_fake_loss + \
                            config['model']['loss_mi_weight'] * sp_disc_code_loss + \
                            config['model']['loss_generator_weight'] * (short_g_loss + long_g_loss)
                div_adv = torch.clamp(div_adv, max=0.3)
                loss_total = total_g_loss - div_adv            
            scaler.scale(total_d_loss).backward()
            scaler.unscale(discriminator_optimizer)

            scaler.step(discriminator_optimizer)

    
            # TOTAL LOSS
            scaler.scale(loss_total).backward()

            # Gradient clipping for training stability
            scaler.unscale(generator_optimizer)
            torch.nn.utils.clip_grad_norm_(state_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(offset_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(target_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            scaler.step(generator_optimizer)

            scaler.update()

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
        if (total_epochs and not evolve):  # if save

            ckpt = {'epoch': epoch,
                    'wandb_id': wandb_logger.wandb_run.id if loggers['wandb'] else None}

            # ckpt = {'epoch': epoch,
            #         'state_encoder': deepcopy(de_parallel(state_encoder)).half(),
            #         'target_encoder': deepcopy(de_parallel(target_encoder)).half(),
            #         'offset_encoder': deepcopy(de_parallel(offset_encoder)).half(),
            #         'lstm': deepcopy(de_parallel(lstm)).half(),
            #         'decoder': deepcopy(de_parallel(decoder)).half(),
            #         'short_discriminator': deepcopy(de_parallel(short_discriminator)).half(),
            #         'long_discriminator': deepcopy(de_parallel(long_discriminator)).half(),
            #         'single_pose_discriminator': deepcopy(de_parallel(single_pose_discriminator)).half(),
            #         'discriminator_optimizer': discriminator_optimizer.state_dict(),
            #         'generator_optimizer': generator_optimizer.state_dict(),
            #         'wandb_id': wandb_logger.wandb_run.id if loggers['wandb'] else None}

            # Save last, best and delete
            torch.save(ckpt, weight_path+'/lastest.pt')
            pathlib.Path(weight_path).mkdir(parents=True, exist_ok=True)
            torch.save(state_encoder.state_dict(), weight_path + '/state_encoder.pkl')
            torch.save(target_encoder.state_dict(), weight_path + '/target_encoder.pkl')
            torch.save(offset_encoder.state_dict(), weight_path + '/offset_encoder.pkl')
            torch.save(lstm.state_dict(), weight_path + '/lstm.pkl')
            torch.save(decoder.state_dict(), weight_path + '/decoder.pkl')
            torch.save(short_discriminator.state_dict(), weight_path + '/short_discriminator.pkl')
            torch.save(long_discriminator.state_dict(), weight_path + '/long_discriminator.pkl')
            if loggers['wandb']:
                if ((epoch + 1) % save_period == 0 and not final_epoch) and save_period != -1:
                    wandb_logger.log_model(model_path, save_period, project, epoch)
            del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------            
    LOGGER.info(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')
    wandb_logger.finish_run()
    torch.cuda.empty_cache()
if __name__ == '__main__':
    train()
