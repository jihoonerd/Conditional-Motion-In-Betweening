import torch

def vectorize_pose(sampled_batch):
    """Reshape root_p and local_q to match with transformer src dimension
    
    Returns: root_p, local_q
    """
    batch_size, seq_len = sampled_batch['local_q'].shape[0], sampled_batch['local_q'].shape[1]

    # Should have (Seq len, Batch size, Embedding dim)
    root_p = sampled_batch['root_p'].permute(1,0,2) 
    local_q = sampled_batch['local_q'].reshape(batch_size, seq_len, -1).permute(1, 0, 2)

    out = torch.cat([root_p, local_q], dim=2)
    return out