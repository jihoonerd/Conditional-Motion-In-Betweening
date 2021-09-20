import torch
import numpy as np


def replace_noise(minibatch_pose_input, mask_start_frame):
    
    seq_len = minibatch_pose_input.size(1)
    interpolated = torch.ones_like(minibatch_pose_input, device=minibatch_pose_input.device) * 0.1

    if mask_start_frame == 0 or mask_start_frame == (seq_len -1):
        interpolate_start = minibatch_pose_input[:,0,:]
        interpolate_end = minibatch_pose_input[:,seq_len-1,:]

        interpolated[:,0,:] = interpolate_start
        interpolated[:,seq_len-1,:] = interpolate_end

        assert torch.allclose(interpolated[:,0,:], interpolate_start)
        assert torch.allclose(interpolated[:,seq_len-1,:], interpolate_end)

    else:
        interpolate_start1 = minibatch_pose_input[:,0,:]
        interpolate_end1 = minibatch_pose_input[:,mask_start_frame,:]

        interpolate_start2 = minibatch_pose_input[:, mask_start_frame, :]
        interpolate_end2 = minibatch_pose_input[:, seq_len-1,:]

        interpolated[:,0,:] = interpolate_start1
        interpolated[:,mask_start_frame,:] = interpolate_end1

        interpolated[:,mask_start_frame,:] = interpolate_start2
        interpolated[:,seq_len-1,:] = interpolate_end2

        
        assert torch.allclose(interpolated[:,0,:], interpolate_start1)
        assert torch.allclose(interpolated[:,mask_start_frame,:], interpolate_end1)
        
        assert torch.allclose(interpolated[:,mask_start_frame,:], interpolate_start2)
        assert torch.allclose(interpolated[:,seq_len-1,:], interpolate_end2)
    return interpolated


def replace_inpainting_range(pose_vectorized_input, mask_start_frame, num_masks, batch_size, feature_dims, infill_value=0.1):
    seq_len = pose_vectorized_input.shape[1]
    pose_vectorized_input[:,1:mask_start_frame,:] = torch.ones((batch_size, mask_start_frame-1, feature_dims)) * infill_value
    pose_vectorized_input[:,mask_start_frame+num_masks:-1, :] = torch.ones((batch_size, seq_len - (mask_start_frame + num_masks + 1), feature_dims)) * infill_value
    return pose_vectorized_input
    

def lerp_reshaped(input_vec, mask_start_frame, num_offsets):
    batch_size = input_vec.size(0)
    seq_len = input_vec.size(1)
    vec_reshaped = input_vec.reshape(batch_size, seq_len, num_offsets, -1)
    interpolated = torch.zeros_like(vec_reshaped, device=input_vec.device)

    if mask_start_frame == 0 or mask_start_frame == (seq_len -1):
        interpolate_start = vec_reshaped[:,0]
        interpolate_end = vec_reshaped[:,seq_len-1]

        for i in range(seq_len):
            dt = 1 / (seq_len-1)
            interpolated[:,i,:] = torch.lerp(interpolate_start, interpolate_end, dt * i)

        assert torch.allclose(interpolated[:,0], interpolate_start)
        assert torch.allclose(interpolated[:,seq_len-1], interpolate_end)
    else:
        interpolate_start1 = vec_reshaped[:,0]
        interpolate_end1 = vec_reshaped[:,mask_start_frame]

        interpolate_start2 = vec_reshaped[:, mask_start_frame]
        interpolate_end2 = vec_reshaped[:, -1]

        for i in range(mask_start_frame+1):
            dt = 1 / mask_start_frame
            interpolated[:,i] = torch.lerp(interpolate_start1, interpolate_end1, dt * i)

        assert torch.allclose(interpolated[:,0], interpolate_start1)
        assert torch.allclose(interpolated[:,mask_start_frame], interpolate_end1)
        
        for i in range(mask_start_frame, seq_len):
            dt = 1 / (seq_len - mask_start_frame - 1)
            interpolated[:,i,:] = torch.lerp(interpolate_start2, interpolate_end2, dt * (i - mask_start_frame))
        
        assert torch.allclose(interpolated[:,mask_start_frame], interpolate_start2)
        assert torch.allclose(interpolated[:,-1], interpolate_end2)

    interpolated = torch.nn.functional.normalize(interpolated, p=2.0, dim=3)
    return interpolated.reshape(batch_size, seq_len, -1)


def slerp(low, high, t):
    # low, high: [N, 22, 3]

    low_norm = torch.nn.functional.normalize(low, p=2.0, dim=2)
    high_norm = torch.nn.functional.normalize(high, p=2.0, dim=2)
    omega = torch.acos((low_norm * high_norm).sum(2)) # (Batch, 22)
    so = torch.sin(omega)
    res = (torch.sin((1.0-t) * omega)/so).unsqueeze(2) * low + (torch.sin(t * omega)/so).unsqueeze(2) * high
    return res

def lerp_input_repr(minibatch_pose_input, mask_start_frame):
    # Use LERP for positions and SLERP for rotations
    # minibatch_pose_input (N.L.D)

    seq_len = minibatch_pose_input.size(1)
    interpolated = torch.zeros_like(minibatch_pose_input, device=minibatch_pose_input.device)

    if mask_start_frame == 0 or mask_start_frame == (seq_len -1):
        interpolate_start = minibatch_pose_input[:,0,:]
        interpolate_end = minibatch_pose_input[:,seq_len-1,:]

        for i in range(seq_len):
            dt = 1 / (seq_len-1)
            interpolated[:,i,:] = torch.lerp(interpolate_start, interpolate_end, dt * i)

        assert torch.allclose(interpolated[:,0,:], interpolate_start)
        assert torch.allclose(interpolated[:,seq_len-1,:], interpolate_end)
    else:
        interpolate_start1 = minibatch_pose_input[:,0,:]
        interpolate_end1 = minibatch_pose_input[:,mask_start_frame,:]

        interpolate_start2 = minibatch_pose_input[:, mask_start_frame, :]
        interpolate_end2 = minibatch_pose_input[:, -1,:]

        
        for i in range(mask_start_frame+1):
            dt = 1 / mask_start_frame
            interpolated[:,i,:] = torch.lerp(interpolate_start1, interpolate_end1, dt * i)

        assert torch.allclose(interpolated[:,0,:], interpolate_start1)
        assert torch.allclose(interpolated[:,mask_start_frame,:], interpolate_end1)
        
        for i in range(mask_start_frame, seq_len):
            dt = 1 / (seq_len - mask_start_frame - 1)
            interpolated[:,i,:] = torch.lerp(interpolate_start2, interpolate_end2, dt * (i - mask_start_frame))
        
        assert torch.allclose(interpolated[:,mask_start_frame,:], interpolate_start2)
        assert torch.allclose(interpolated[:,-1,:], interpolate_end2)
    return interpolated

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

def vectorize_representation(global_position, global_rotation):

    batch_size = global_position.shape[0]
    seq_len = global_position.shape[1]

    global_pos_vec = global_position.reshape(batch_size, seq_len, -1).contiguous()
    global_rot_vec = global_rotation.reshape(batch_size, seq_len, -1).contiguous()

    global_pose_vec_gt = torch.cat([global_pos_vec, global_rot_vec], dim=2)
    return global_pose_vec_gt

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
