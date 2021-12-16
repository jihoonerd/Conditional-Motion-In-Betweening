from torch.utils.data import Dataset
from rmi.lafan1 import extract, utils
import numpy as np
import torch
import pickle
import os 

class LAFAN1Dataset(Dataset):
    def __init__(self, lafan_path: str, processed_data_dir : str, train: bool, device: str, window: int=65):
        self.lafan_path = lafan_path

        self.train = train
        # 4.3: It contains actions performedby 5 subjects, with Subject 5 used as the test set.
        self.actors = (
            ["subject1", "subject2", "subject3", "subject4"] if train else ["subject5"]
        )

        # 4.3: ... The training statistics for normalization are computed on windows of 50 frames offset by 20 frames.
        self.window = window

        # 4.3: Given the larger size of ... we sample our test windows from Subject 5 at every 40 frames.
        # The training statistics for normalization are computed on windows of 50 frames offset by 20 frames.
        self.offset = 20 if self.train else 40

        self.device = device
        
        pickle_name = "processed_train_data.pkl" if train else "processed_test_data.pkl"

        if pickle_name in os.listdir(processed_data_dir):
            with open(os.path.join(processed_data_dir, pickle_name), 'rb') as f:
                self.data = pickle.load(f)
        else: 
            self.data = self.load_lafan()  # Call this last
            with open(os.path.join(processed_data_dir, pickle_name), 'wb') as f:
                pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)

    @property
    def root_v_dim(self):
        return self.data["root_v"].shape[2]

    @property
    def local_q_dim(self):
        return self.data["local_q"].shape[2] * self.data["local_q"].shape[3]

    @property
    def contact_dim(self):
        return self.data["contact"].shape[2]

    @property
    def num_joints(self):
        return self.data["global_pos"].shape[2]

    def load_lafan(self):
        # This uses method provided with LAFAN1.
        # X and Q are local position/quaternion. Motions are rotated to make 10th frame facing X+ position.
        # Refer to paper 3.1 Data formatting
        X, Q, parents, contacts_l, contacts_r, seq_names = extract.get_lafan1_set(
            self.lafan_path, self.actors, self.window, self.offset, self.train
        )

        # Retrieve global representations. (global quaternion, global positions)
        _, global_pos = utils.quat_fk(Q, X, parents)

        input_data = {}
        input_data["local_q"] = Q  # q_{t}
        input_data["local_q_offset"] = Q[:, -1, :, :]  # lasst frame's quaternions
        input_data["q_target"] = Q[:, -1, :, :]  # q_{T}

        input_data["root_v"] = (
            global_pos[:, 1:, 0, :] - global_pos[:, :-1, 0, :]
        )  # \dot{r}_{t}
        input_data["root_p_offset"] = global_pos[
            :, -1, 0, :
        ]  # last frame's root positions
        input_data["root_p"] = global_pos[:, :, 0, :]

        input_data["contact"] = np.concatenate(
            [contacts_l, contacts_r], -1
        )  # Foot contact
        input_data["global_pos"] = global_pos[
            :, :, :, :
        ]  # global position (N, 50, 22, 30) why not just global_pos
        input_data['seq_names'] = seq_names
        return input_data

    def __len__(self):
        return self.data["global_pos"].shape[0]

    def __getitem__(self, index):
        query = {}
        query["local_q"] = self.data["local_q"][index].astype(np.float32)
        query["local_q_offset"] = self.data["local_q_offset"][index].astype(np.float32)
        query["q_target"] = self.data["q_target"][index].astype(np.float32)
        query["root_v"] = self.data["root_v"][index].astype(np.float32)
        query["root_p_offset"] = self.data["root_p_offset"][index].astype(np.float32)
        query["root_p"] = self.data["root_p"][index].astype(np.float32)
        query["contact"] = self.data["contact"][index].astype(np.float32)
        query["global_pos"] = self.data["global_pos"][index].astype(np.float32)
        return query
