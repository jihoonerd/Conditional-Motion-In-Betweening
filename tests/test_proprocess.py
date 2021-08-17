from rmi.model.preprocess import vectorize_pose, create_mask
import numpy as np
import torch

device = torch.device("cpu")

def test_vectorize_pose():
    pose_local_q = torch.randn((64, 50, 22, 4))
    pose_root_p = torch.randn((64, 50, 3))
    sampled_batch = {
        'local_q': pose_local_q,
        'root_p': pose_root_p
    }
    
    pose_vec = vectorize_pose(sampled_batch)

    np.testing.assert_array_equal(pose_vec[:,:,:3].permute(1,0,2), pose_root_p)
    np.testing.assert_array_equal(pose_vec[:,:,3:].permute(1,0,2).reshape(64, 50, 22, 4), pose_local_q)
    np.testing.assert_equal(pose_vec.shape, (50, 64, 91))


def test_create_mask():

    pose_local_q = torch.randn((64, 50, 22, 4))
    pose_root_p = torch.randn((64, 50, 3))
    sampled_batch = {
        'local_q': pose_local_q,
        'root_p': pose_root_p
    }
    
    pose_vec = vectorize_pose(sampled_batch)
    src_mask1, src_padding_mask1 = create_mask(pose_vec, device=device, mask_start=10, mask_end=49)
    assert (src_mask1[:,10:49] == True).all()
    assert (src_padding_mask1 == False).all()
    
    src_mask2, src_padding_mask2 = create_mask(pose_vec, device=device, mask_start=20, mask_end=45)
    assert (src_mask2[:,20:45] == True).all()
    assert (src_padding_mask2 == False).all()
