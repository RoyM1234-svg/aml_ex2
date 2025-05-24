import torch
from normalized_flow_model import NormalizedFlowModel
from src.create_data import create_unconditional_olympic_rings




data = create_unconditional_olympic_rings(10000, ring_thickness=0.25, verbose=True)
print(data.shape)






