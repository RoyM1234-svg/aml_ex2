import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int = 1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU()]
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU()])

        layers.append(nn.Linear(hidden_dim, input_dim))
        self.scale_network = nn.Sequential(*layers)
        

    def forward(self, zl: torch.Tensor) -> torch.Tensor:
        return self.scale_network(zl)

        

    