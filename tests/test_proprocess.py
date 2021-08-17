from rmi.model.preprocess import vectorize_pose
import numpy as np
import torch

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
