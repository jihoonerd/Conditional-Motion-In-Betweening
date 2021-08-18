import torch

def vectorize_pose(root_p, local_q, vector_dim, device):
    """Reshape root_p and local_q to match with transformer src dimension
    
    Returns: root_p, local_q
    """
    batch_size, seq_len = local_q.shape[0], local_q.shape[1]

    # Should have (Seq len, Batch size, Embedding dim)
    root_p = root_p.permute(1,0,2) 
    local_q = local_q.reshape(batch_size, seq_len, -1).permute(1, 0, 2)

    padding_dim = vector_dim - (root_p.shape[-1] + local_q.shape[-1])
    dummy = torch.zeros([seq_len, batch_size, padding_dim], device=device)
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