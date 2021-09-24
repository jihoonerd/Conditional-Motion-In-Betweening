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


def slerp_input_repr(low, high, t):
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

def vectorize_representation(global_position, global_rotation):

    batch_size = global_position.shape[0]
    seq_len = global_position.shape[1]

    global_pos_vec = global_position.reshape(batch_size, seq_len, -1).contiguous()
    global_rot_vec = global_rotation.reshape(batch_size, seq_len, -1).contiguous()

    global_pose_vec_gt = torch.cat([global_pos_vec, global_rot_vec], dim=2)
    return global_pose_vec_gt
