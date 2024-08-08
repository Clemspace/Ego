# File: data_utils.py

import torch
import torch.nn.functional as F
import json
from torch.utils.data import Dataset

def arc_collate_fn(batch):
    # This function is no longer needed as we're processing one subtask at a time
    return batch[0]

class ARCDataset(Dataset):
    def __init__(self, tasks):
        self.subtasks = []
        for task in tasks:
            self.subtasks.append((task.train, task.test))

    def __len__(self):
        return len(self.subtasks)

    def __getitem__(self, idx):
        train_data, test_data = self.subtasks[idx]
        
        train_inputs = [pair[0] for pair in train_data]
        train_outputs = [pair[1] for pair in train_data]
        
        return train_inputs, train_outputs
    
class ARCTask:
    def __init__(self, task_data):
        self.train = [self.parse_grid(item) for item in task_data['train']]
        self.test = [self.parse_grid(item) for item in task_data['test']]

    @staticmethod
    def parse_grid(item):
        input_grid = torch.tensor(item['input'], dtype=torch.float32)
        output_grid = torch.tensor(item['output'], dtype=torch.float32) if 'output' in item else None
        return (input_grid, output_grid)

def load_arc_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {task_id: ARCTask(task_data) for task_id, task_data in data.items()}

def augment_grid(grid):
    augmented = []
    # Original
    augmented.append(grid)
    # Rotations
    for k in range(1, 4):
        augmented.append(torch.rot90(grid, k))
    # Flips
    augmented.append(torch.flip(grid, [0]))
    augmented.append(torch.flip(grid, [1]))
    # Scaling (example: 0.5x and 2x)
    augmented.append(F.interpolate(grid.unsqueeze(0).unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze())
    augmented.append(F.interpolate(grid.unsqueeze(0).unsqueeze(0), scale_factor=2, mode='nearest').squeeze())
    return augmented