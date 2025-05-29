import torch
import torch.nn as nn
from scale_network import ScaleNetwork
from abc import ABC, abstractmethod

class ReversibleLayer(nn.Module, ABC):
    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        pass


class NormalizedFlowModel(ReversibleLayer):
    def __init__(self, input_dim: int, n_layers: int = 15):
        super().__init__()
        self.base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(input_dim), 
            torch.eye(input_dim)
        )
        layers: list[ReversibleLayer] = []
        layers.append(AffineCouplingLayer(input_dim))
        for _ in range(n_layers - 1):
            layers.extend([PermutationLayer(input_dim), AffineCouplingLayer(input_dim)])
        self.layers: nn.ModuleList = nn.ModuleList(layers) 

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z)
        return z

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.layers):
            y = layer.inverse(y) # type: ignore
        return y

    def log_inverse_jacobian_determinant(self, y: torch.Tensor) -> torch.Tensor:
        log_det = torch.zeros(y.shape[0])
        for layer in reversed(self.layers):
            if isinstance(layer, AffineCouplingLayer):
                log_det += layer.log_inverse_jacobian_determinant(y)
            y = layer.inverse(y) # type: ignore
        return log_det

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_dist.log_prob(self.inverse(y)) + self.log_inverse_jacobian_determinant(y)

class PermutationLayer(ReversibleLayer):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

        self.register_buffer("permutation", torch.randperm(input_dim))
        self.register_buffer("inverse_permutation", torch.argsort(self.permutation)) # type: ignore

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, self.permutation] # type: ignore
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y[:, self.inverse_permutation] # type: ignore
    

class AffineCouplingLayer(ReversibleLayer):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.log_s_network = ScaleNetwork(input_dim // 2, hidden_dim=8, num_hidden_layers=5)
        self.b_network = ScaleNetwork(input_dim // 2, hidden_dim=8, num_hidden_layers=5)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        left_half = z[:, :self.input_dim // 2]
        right_half = z[:, self.input_dim // 2:]

        log_s = self.log_s_network(left_half)
        b = self.b_network(left_half)
        right_half = right_half * torch.exp(log_s) + b

        return torch.cat([left_half, right_half], dim = 1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        left_half = y[:, :self.input_dim // 2]
        right_half = y[:, self.input_dim // 2:]

        log_s = self.log_s_network(left_half)
        b = self.b_network(left_half)
        s = torch.exp(log_s)
        right_half = (right_half - b) / s

        return torch.cat([left_half, right_half], dim = 1)

    def log_inverse_jacobian_determinant(self, y: torch.Tensor) -> torch.Tensor:
        left_half = y[:, :self.input_dim // 2]
        log_s = self.log_s_network(left_half)
        return -torch.sum(log_s, dim = 1)


    
    

        
