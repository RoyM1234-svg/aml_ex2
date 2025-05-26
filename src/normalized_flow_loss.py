import torch
from torch import nn
from normalized_flow_model import ReversibleLayer

class NormalizedFlowLoss(nn.Module):
    def __init__(self, input_dim: int, model: ReversibleLayer):
        super().__init__()
        self.input_dim = input_dim
        self.model = model

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(self.input_dim), 
            torch.eye(self.input_dim)
        )

        inverse_y = self.model.inverse(y)
        log_prob = base_dist.log_prob(inverse_y)
        log_det = self.model.log_inverse_jacobian_determinant(y) # type: ignore

        return (-log_prob - log_det).mean(), -log_prob.mean(), -log_det.mean()

