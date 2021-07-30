import argparse
import os
from pathlib import Path

import imageio
import numpy as np
import torch
from kpt.model.skeleton import TorchSkeleton
from PIL import Image
from pymo.parsers import BVHParser
from torch.utils.data import DataLoader
from mpl_toolkits import mplot3d

from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import generate_infogan_code, write_json
from rmi.model.network import (Decoder, InfoGANDiscriminator, InfoganCodeEncoder,
                               InputEncoder, LSTMNetwork, DInfoGAN,
                               QInfoGAN)
from rmi.model.positional_encoding import PositionalEncoding
from rmi.vis.pose import plot_pose, plot_pose_compare, plot_pose_compare3
from utils.general import increment_path
from utils.torch_utils import select_device



def test(opt, device):

    save_dir, pretrained_weights, data_path = opt.save_dir, opt.pretrained_weights, opt.data_path    
    device = torch.device("cpu")


    infogan_cont_code = opt.infogan_cont_code 
    infogan_disc_code = opt.infogan_disc_code
    conditioning_disc_code = opt.conditioning_disc_code
    conditioning_cont_code = opt.conditioning_cont_code
    save_dir = Path(save_dir)

    # Load Skeleton
    parsed = BVHParser().parse(opt.skeleton_path)
    skeleton = TorchSkeleton(skeleton=parsed.skeleton, root_name='Hips', device=device)

    # Load and preprocess data. It utilizes LAFAN1 utilities
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    lafan_dataset_test = LAFAN1Dataset(lafan_path=data_path, processed_data_dir=opt.processed_data_dir, train=False,  target_action='walk', start_seq_length=30, cur_seq_length=30, max_transition_length=30, device=device)
    lafan_data_loader_test = DataLoader(lafan_dataset_test, batch_size=opt.batch_size, shuffle=False, num_workers=opt.data_loader_workers)

    inference_batch_index = opt.inference_batch_index

    # Extract dimension from processed data
    root_v_dim = lafan_dataset_test.root_v_dim
    local_q_dim = lafan_dataset_test.local_q_dim
    contact_dim = lafan_dataset_test.contact_dim

    # Initializing networks
    ckpt = torch.load(pretrained_weights, map_location=torch.device(device))
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

    lstm_hidden = opt.lstm_hidden
    infogan_code_encoder = InfoganCodeEncoder(input_dim=infogan_cont_code + infogan_disc_code, out_dim=lstm_hidden)
    infogan_code_encoder.to(device)

    # LSTM
    lstm_in = state_encoder.out_dim * 3
    lstm = LSTMNetwork(input_dim=lstm_in, hidden_dim=lstm_hidden, device=device)
    lstm.to(device)

    # Decoder
    decoder = Decoder(input_dim=lstm_hidden, out_dim=state_in)
    decoder.to(device)

    #Load to FP32
    state_dict_state_encoder = ckpt['state_encoder']
    state_encoder.load_state_dict(state_dict_state_encoder)  

    state_dict_target_encoder = ckpt['target_encoder']
    target_encoder.load_state_dict(state_dict_target_encoder)  
    
    state_dict_offset_encoder = ckpt['offset_encoder']
    offset_encoder.load_state_dict(state_dict_offset_encoder)  

    state_dict_infogan_code_encoder = ckpt['infogan_code_encoder']
    infogan_code_encoder.load_state_dict(state_dict_infogan_code_encoder) 

    state_dict_lstm = ckpt['lstm']
    lstm.load_state_dict(state_dict_lstm)  

    state_dict_decoder = ckpt['decoder']
    decoder.load_state_dict(state_dict_decoder)

    pe = PositionalEncoding(dimension=256, max_len=lafan_dataset_test.max_transition_length)

    print("MODELS LOADED WITH SAVED WEIGHTS")

    state_encoder.eval()
    offset_encoder.eval()
    target_encoder.eval()
    infogan_code_encoder.eval()
    lstm.eval()
    decoder.eval()
    cond_disc_code = list(range(0,5))
    for i_batch, sampled_batch in enumerate(lafan_data_loader_test):
            # img_integrated = []
        for conditioning_disc_code in [0]:

            current_batch_size = len(sampled_batch['global_pos'])
            c = [0]
            pred_pose = []
            with torch.no_grad():
                pred_pose = []
                pred_true = []
                # for conditioning_disc_code in [0,7,14]:            
                for cont_val in np.linspace(-5, 5, 3):
                    conditioning_disc_code = 1
                    # state input
                    img_gt = []
                    img_pred = []
                    img_integrated = []


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
                    # InfoGAN code
                    if infogan_cont_code ==0 :
                        infogan_disc_code_gen = torch.zeros(current_batch_size, infogan_disc_code)
                        infogan_disc_code_gen[:,conditioning_disc_code] = 1
                        infogan_code_gen = infogan_disc_code_gen
                    elif infogan_disc_code == 0:
                        infogan_cont_code_gen = torch.zeros(current_batch_size, infogan_cont_code)
                        infogan_cont_code_gen[:,conditioning_cont_code] = 0
                        infogan_code_gen = infogan_cont_code_gen
                    else :
                        infogan_disc_code_gen = torch.zeros(current_batch_size, infogan_disc_code)
                        infogan_disc_code_gen[:,conditioning_disc_code] = 1
                        infogan_cont_code_gen = torch.zeros(current_batch_size, infogan_cont_code)
                        infogan_cont_code_gen[:,conditioning_cont_code] = cont_val
                        infogan_code_gen = torch.cat([infogan_disc_code_gen, infogan_cont_code_gen], dim=1)

                    lstm.h[0] = infogan_code_encoder(infogan_code_gen.to(torch.float))

                    training_frames = opt.training_frames
                    for t in range(training_frames):
                        # root pos
                        if t  == 0:
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
                        h_offset = pe(h_offset, tta)
                        h_target = pe(h_target, tta)

                        offset_target = torch.cat([h_offset, h_target], dim=1)

                        # lstm
                        h_in = torch.cat([h_state, offset_target], dim=1).unsqueeze(0)
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
                        local_q_pred_ = local_q_pred_.squeeze() # (seq, joint, 4)
                        pos_pred, rot_pred = skeleton.forward_kinematics(root_pred, local_q_pred_, rot_repr='quaternion')
                        
                        # Exporting
                        root_pred_t = root_pred[inference_batch_index].numpy()
                        local_q_pred_t = local_q_pred_[inference_batch_index].numpy()

                        start_pose = global_pos[inference_batch_index, 0].numpy()
                        in_between_pose = pos_pred[inference_batch_index].numpy()
                        in_between_true = global_pos[inference_batch_index, t].numpy()
                        target_pose = global_pos[inference_batch_index, training_frames-1].numpy()

                        pose_path = os.path.join(save_dir, f"{i_batch}")
                        Path(pose_path).mkdir(parents=True, exist_ok=True)

                        # if t == 0: # root_pose[0] only root check
                        #     write_json(filename=os.path.join(pose_path, f'start.json'), local_q=sampled_batch['local_q'][inference_batch_index][0].numpy(), root_pos=start_pose[0], joint_names=skeleton.joints)
                        #     write_json(filename=os.path.join(pose_path, f'target.json'), local_q=sampled_batch['local_q'][inference_batch_index][-1].numpy(), root_pos=target_pose[0], joint_names=skeleton.joints)

                        # write_json(filename=os.path.join(pose_path, f'{t:05}.json'), local_q=local_q_pred_t, root_pos=root_pred_t, joint_names=skeleton.joints)
                        pred_pose.append(in_between_pose)
                        pred_true.append(in_between_true)
                if opt.plot :
                    for t in range(training_frames):
                        plot_pose_compare3(start_pose, pred_pose[t], pred_pose[t+training_frames], pred_pose[t+2*training_frames], target_pose, t, skeleton, save_dir=save_dir, pred=True)
                        plot_pose(start_pose, pred_true[t], target_pose, t, skeleton, save_dir=save_dir, pred=False)
                        img_path = os.path.join(save_dir, 'results/tmp/')
                        Path(img_path).mkdir(parents=True, exist_ok=True)

                        pred_img = Image.open(img_path +'pred_'+str(t)+'.png', 'r')
                        gt_img = Image.open(img_path+ 'gt_'+str(t)+'.png', 'r')
                        img_pred.append(pred_img)
                        img_gt.append(gt_img)
                        img_integrated.append(np.concatenate([pred_img, gt_img.resize(pred_img.size)], 1))
                if opt.plot:
                    # if i_batch < 49:
                    gif_path = os.path.join(opt.save_dir, 'img'+str(conditioning_disc_code)+'__'+'cond'+str(conditioning_cont_code)+'_%02d.gif' % i_batch)
                    imageio.mimsave(gif_path, img_integrated, duration=0.1)
            print('conditioning_cont_code', conditioning_cont_code)    
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_weights', type=str, default=None, help='load weight .pt')
    parser.add_argument('--data_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH', help='dataset path')
    parser.add_argument('--skeleton_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh', help='dataset path')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data/', help='dataset path')
    parser.add_argument('--batch_size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--lstm_hidden', type=int, default=1024, help='total batch size for all GPUs')
    parser.add_argument('--num_gifs', type=int, default=30, help='total batch size for all GPUs')
    parser.add_argument('--training_frames', type=int, default=30, help='total batch size for all GPUs')
    parser.add_argument('--inference_batch_index', type=int, default=20, help='total batch size for all GPUs')
    parser.add_argument('--infogan_disc_code', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--infogan_cont_code', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--conditioning_disc_code', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--conditioning_cont_code', type=int, default=1, help='total batch size for all GPUs')    
    parser.add_argument('--plot', type=bool, default=True, help='plot motion images')
    parser.add_argument('--data_loader_workers', type=int, default=4, help='data_loader_workers')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):
    opt.exp_name = opt.exp_name
    opt.save_dir = str(increment_path(Path(opt.project) / opt.exp_name, exist_ok=opt.exist_ok))
    device = select_device(opt.device, batch_size=opt.batch_size)
    test(opt, device)


def run(**kwargs):
    # Usage: import train; train.run(weights='RMIB_InfoGAN.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
