import torch
import numpy as np

def lerp_pose(data, from_idx=10, target_idx=40):
    """
    Make linear interpolation in [from_idx, target_idx].
    """

    root_p = data['root_p'].copy()
    local_q = data['local_q'].copy()

    from_root_p = root_p[:, from_idx]
    target_root_p = root_p[:, target_idx]
    inter_root_p = np.linspace(from_root_p, target_root_p, num=target_idx-from_idx, endpoint=False).swapaxes(1,0)
    root_p[:,from_idx:target_idx,:] = inter_root_p

    from_local_q = local_q[:, from_idx]
    target_local_q = local_q[:, target_idx]
    inter_local_q = np.linspace(from_local_q, target_local_q, num=target_idx-from_idx, endpoint=False).swapaxes(1,0)
    local_q[:,from_idx:target_idx,:] = inter_local_q

    return root_p, local_q

def vectorize_pose(root_p, local_q, vector_dim, device):
    """Reshape root_p and local_q to match with transformer src dimension
    
    Returns: root_p, local_q
    """
    batch_size, seq_len = local_q.shape[0], local_q.shape[1]

    # Should have (Seq len, Batch size, Embedding dim)
    root_p = torch.Tensor(root_p).to(device)
    local_q = torch.Tensor(local_q.reshape(batch_size, seq_len, -1)).to(device)

    padding_dim = vector_dim - (root_p.shape[-1] + local_q.shape[-1])
    dummy = torch.zeros([batch_size, seq_len, padding_dim], device=device)
    out = torch.cat([root_p, local_q, dummy], dim=2)
    return out


def create_mask(pose_vector, device, mask_start=10, mask_end=49):
    """Generate masks for pose vector. Masking range is [mask_start, mask_end)

    Returns: src_mask, src_padding_mask
    """
    
    seq_len = pose_vector.shape[0]
    batch_size = pose_vector.shape[1]
    src_mask = torch.zeros((seq_len, seq_len), device=device).type(torch.bool)  # (seq_len, seq_len)
    src_mask[:, mask_start:mask_end] = True
    src_padding_mask = torch.zeros((batch_size, seq_len), device=device).type(torch.bool)
    return src_mask, src_padding_mask