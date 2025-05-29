import torch
from normalized_flow_model import NormalizedFlowModel
from utils import visualize_points_with_rings
import numpy as np

inside_points = [
        [0.1, 0.05], 
        [1.95, -0.1],
        [3.1, -0.95],
    ]
    
outside_points = [
    [3, 1],
    [5.5, 0],
]
all_points = np.array(inside_points + outside_points)
mean = np.array([2, -0.5])
std = np.array([1.5, 0.7])
normalized = (all_points - mean) / std

inside_points = torch.tensor(normalized[:3], dtype=torch.float32)
outside_points = torch.tensor(normalized[3:], dtype=torch.float32)

model = NormalizedFlowModel(input_dim=2)
model.load_state_dict(torch.load("normalized_flow_model.pth"))


visualize_points_with_rings(model, inside_points, outside_points)
