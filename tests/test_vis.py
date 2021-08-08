from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.vis.pose import project_root_position
import torch
from pathlib import Path

def test_project_root_position():
    data_path = 'ubisoft-laforge-animation-dataset/output/BVH'
    action = 'dance'
    processed_path = f'processed_data_{action}/'
    Path(processed_path).mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    lafan_dataset = LAFAN1Dataset(lafan_path=data_path, processed_data_dir=processed_path, train=True, target_action=[action], device=device, start_seq_length=30, cur_seq_length=30, max_transition_length=30)


    sample_data = lafan_dataset[0:200]['global_pos']
    project_root_position(sample_data, action)