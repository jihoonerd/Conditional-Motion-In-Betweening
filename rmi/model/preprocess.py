import torch
import numpy as np


def replace_noise(data, from_idx=9, target_idx=40, fixed=None):

    root_p = data['root_p'].copy()
    local_q = data['local_q'].copy()

    if not fixed:
        # Starting frame: 9, Endframe:40, Inbetween start: 10, Inbetween end: 39
        noise_root_p = np.random.normal(size=(root_p.shape[0], target_idx-from_idx-1, root_p.shape[2]))
        root_p[:,from_idx+1:target_idx,:] = noise_root_p # Replace with noise from [from_idx, target_idx)

        noise_local_q = np.random.normal(size=(local_q.shape[0], target_idx-from_idx-1, local_q.shape[2], local_q.shape[3]))
        local_q[:,from_idx+1:target_idx,:] = noise_local_q
    else:
        noise_root_p = np.random.normal(size=(root_p.shape[0], target_idx-from_idx-1, root_p.shape[2]))
        root_p[:,from_idx+1:from_idx+1+fixed,:] = noise_root_p[:,:fixed,:]
        root_p[:,from_idx+1+fixed+1:target_idx,:] = noise_root_p[:,fixed+1:,:]

        noise_local_q = np.random.normal(size=(local_q.shape[0], target_idx-from_idx-1, local_q.shape[2], local_q.shape[3]))
        local_q[:,from_idx+1:from_idx+1+fixed,:] = noise_local_q[:,:fixed,:,:]
        local_q[:,from_idx+1+fixed+1:target_idx,:] = noise_local_q[:,fixed+1:,:,:]

    return root_p, local_q


def replace_given_range(pose_vectorized_input, mask_start_frame, num_masks, batch_size, feature_dims, infill_value=0.1):
    pose_vectorized_input[:,mask_start_frame:mask_start_frame+num_masks,:] = torch.ones((batch_size, num_masks, feature_dims)) * infill_value
    return pose_vectorized_input



def replace_infill(data, from_idx=9, target_idx=40, fixed=None, infill_value=0.1):

    root_p = data['root_p'].copy()
    local_q = data['local_q'].copy()

    if not fixed:
        # Starting frame: 9, Endframe:40, Inbetween start: 10, Inbetween end: 39
        noise_root_p = np.ones((root_p.shape[0], target_idx-from_idx-1, root_p.shape[2])) * infill_value
        root_p[:,from_idx+1:target_idx,:] = noise_root_p # Replace with noise from [from_idx, target_idx)

        noise_local_q = np.ones((local_q.shape[0], target_idx-from_idx-1, local_q.shape[2], local_q.shape[3])) * infill_value
        local_q[:,from_idx+1:target_idx,:] = noise_local_q
    else:
        noise_root_p = np.ones((root_p.shape[0], target_idx-from_idx-1, root_p.shape[2])) * infill_value
        root_p[:,from_idx+1:from_idx+1+fixed,:] = noise_root_p[:,:fixed,:]
        root_p[:,from_idx+1+fixed+1:target_idx,:] = noise_root_p[:,fixed+1:,:]

        noise_local_q = np.ones((local_q.shape[0], target_idx-from_idx-1, local_q.shape[2], local_q.shape[3])) * infill_value
        local_q[:,from_idx+1:from_idx+1+fixed,:] = noise_local_q[:,:fixed,:,:]
        local_q[:,from_idx+1+fixed+1:target_idx,:] = noise_local_q[:,fixed+1:,:,:]

    return root_p, local_q

def lerp_pose(data, from_idx=9, target_idx=39):
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

def vectorize_pose(root_p, local_q, contact, vector_dim, device):
    """Reshape root_p and local_q to match with transformer src dimension
    
    Returns: root_p, local_q
    """
    batch_size, seq_len = local_q.shape[0], local_q.shape[1]

    # Should have (Seq len, Batch size, Embedding dim)
    root_p = torch.Tensor(root_p).to(device)
    local_q = torch.Tensor(local_q.reshape(batch_size, seq_len, -1)).to(device)
    contact = torch.Tensor(contact).to(device)

    padding_dim = vector_dim - (root_p.shape[-1] + local_q.shape[-1] + contact.shape[-1])
    if padding_dim != 0:
        dummy = torch.zeros([batch_size, seq_len, padding_dim], device=device)
        out = torch.cat([root_p, local_q, contact, dummy], dim=2)
    else:
        out = torch.cat([root_p, local_q, contact], dim=2)
    return out, padding_dim


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