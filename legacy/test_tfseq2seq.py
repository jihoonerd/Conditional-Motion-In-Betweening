import argparse
import logging
import os
from pathlib import Path

import torch
from torch.nn.functional import l1_loss
import yaml
from torch.cuda import amp
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from rmi.model.skeleton import Skeleton, sk_joints_to_remove, sk_offsets, sk_parents
import numpy as np
import wandb
from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import flip_bvh
from rmi.model.network import Seq2SeqTransformer
from rmi.model.preprocess import lerp_pose, vectorize_pose, replace_noise
from utils.general import increment_path
from sklearn import preprocessing

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

    # Load Skeleton
    skeleton_mocap = Skeleton(offsets=sk_offsets, parents=sk_parents, device=device)
    skeleton_mocap.remove_joints(sk_joints_to_remove)

    # Flip, Load and preprocess data. It utilizes LAFAN1 utilities
    flip_bvh(opt.data_path)

    # Load LAFAN Dataset
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    lafan_dataset = LAFAN1Dataset(lafan_path=opt.data_path, processed_data_dir=opt.processed_data_dir, train=True, target_action=[''], device=device, start_seq_length=30, cur_seq_length=30, max_transition_length=30)
    
    # Replace to noise for inbetweening frames
    from_idx, target_idx = 9, 40 # Starting frame: 9, Endframe:40, Inbetween start: 10, Inbetween end: 39
    horizon = target_idx - from_idx + 1

    pose_vec_gt, padding_dim = vectorize_pose(lafan_dataset.data['root_p'], lafan_dataset.data['local_q'], lafan_dataset.data['contact'], 96, device)
    pose_vectorized_gt = pose_vec_gt[:,from_idx:target_idx+1,:]

    global_pos = torch.Tensor(lafan_dataset.data['global_pos']).to(device)
    global_pos_gt = global_pos[:,from_idx:target_idx+1,:]


    le = preprocessing.LabelEncoder()
    seq_cat = [x[:-1] for x in lafan_dataset.data['seq_names']]
    le_np = le.fit_transform(seq_cat)
    cdict = {i: le.classes_[i] for i in range(len(le.classes_))}
    seq_labels = torch.Tensor(le_np).type(torch.int8).to(device)

    tensor_dataset = TensorDataset(pose_vectorized_gt, global_pos_gt, seq_labels)
    lafan_data_loader = DataLoader(tensor_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    # Extract dimension from processed data
    root_v_dim = lafan_dataset.root_v_dim
    local_q_dim = lafan_dataset.local_q_dim
    contact_dim = lafan_dataset.contact_dim
    repr_dim = root_v_dim + local_q_dim + contact_dim

    tf_seq2seq = Seq2SeqTransformer(num_encoder_layers=3, num_decoder_layers=3, emb_size=96, nhead=12, dim_feedforward=512, out_dim=repr_dim, bottleneck_dim=opt.bottleneck_dim)
    ckpt = torch.load(opt.prt_weight, map_location=device)
    tf_seq2seq.load_state_dict(ckpt['tf_generator_state_dict'])
    tf_seq2seq.to(device)
    tf_seq2seq.eval()


    pbar = tqdm(lafan_data_loader, position=1, desc="Batch")

    context_vector = []
    seq_label_list = []

    for pose_vectorized_gt, global_pos_gt, seq_label in pbar:

        seq_label_list.extend(seq_label.cpu().detach().tolist())
        pose_vectorized_gt = pose_vectorized_gt.permute(1,0,2)
        seq_len = pose_vectorized_gt.shape[0]
        current_batch_size = pose_vectorized_gt.shape[1]

        src_mask = torch.zeros((seq_len, seq_len), device=device).type(torch.bool)
        src_mask = src_mask.to(device)
        
        with amp.autocast(enabled=cuda):

            output = tf_seq2seq.encode(pose_vectorized_gt)
            context_vector.append(tf_seq2seq.encode(pose_vectorized_gt).cpu().detach().numpy())

    context_vector_set = np.concatenate(context_vector, axis=0)

    import umap
    reducer = umap.UMAP(n_neighbors=300, n_components=3, min_dist=0.3, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(context_vector_set[:2000])
    print(embedding.shape)

    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, ax = plt.subplots(figsize=(15,8))
    ax = fig.add_subplot(projection='3d')

    x = embedding[:,0]
    y = embedding[:,1]
    z = embedding[:,2]

    for k,v in cdict.items():
        if v == 'walk':
            walk_id = k
        if v == 'obstacles':
            obstacles_id = k

    for g in np.unique(seq_label_list[:2000]):
        if g in [walk_id, obstacles_id]:
            continue
        ix = np.where(seq_label_list[:2000] == g)
        ax.scatter(x[ix], y[ix], z[ix], label=cdict[g], alpha=0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 0.5))
    plt.title('UMAP projection', fontsize=12)
    plt.show()
    plt.savefig('umap.png')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--weights', type=str, default='', help='weights path')
    parser.add_argument('--data_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH', help='BVH dataset path')
    parser.add_argument('--skeleton_path', type=str, default='ubisoft-laforge-animation-dataset/output/BVH/walk1_subject1.bvh', help='path to reference skeleton')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_all/', help='path to save pickled processed data')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or -1 or cpu')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='exp', help='save to project/name')
    parser.add_argument('--save_interval', type=int, default=50, help='Log model after every "save_period" epoch')
    parser.add_argument('--generator_learning_rate', type=float, default=0.001, help='generator_learning_rate')
    parser.add_argument('--discriminator_learning_rate', type=float, default=0.0001, help='discriminator_learning_rate')
    parser.add_argument('--cr_learning_rate', type=float, default=0.001, help='crh_infogan learning rate')
    parser.add_argument('--optim_beta1', type=float, default=0.5, help='optim_beta1')
    parser.add_argument('--optim_beta2', type=float, default=0.99, help='optim_beta2')
    parser.add_argument('--loss_recon_weight', type=float, default=1.0, help='loss_recon_weight')
    parser.add_argument('--loss_root_weight', type=float, default=0.01, help='loss_pos_weight')
    parser.add_argument('--loss_quat_weight', type=float, default=1.0, help='loss_quat_weight')
    parser.add_argument('--loss_contact_weight', type=float, default=0.2, help='loss_contact_weight')
    parser.add_argument('--loss_global_pos_weight', type=float, default=1.0, help='loss_global_pos_weight')
    parser.add_argument('--loss_fk_weight', type=float, default=0.1, help='loss_fk_weight')
    parser.add_argument('--bottleneck_dim', type=int, default=256, help='bottleneck_dim')
    parser.add_argument('--prt_weight', default='AE-300.pt')
    opt = parser.parse_args()
    return opt



if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    train(opt, device)