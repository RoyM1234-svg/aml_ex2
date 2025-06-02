import torch
from torch import nn
from flow_matching_models import ConditionalFlowMatchingModel

class ConditionalFlowMatchingLoss(nn.Module):
    def __init__(self, model: ConditionalFlowMatchingModel, input_dim: int):
        super().__init__()
        self.model = model
        self.input_dim = input_dim
        self.base_dist = torch.distributions.MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))

    def forward(self, y_1: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        batch_dim = y_1.shape[0]
        y_0 = self.base_dist.sample((batch_dim,))
        t = torch.rand(batch_dim, 1)
        y_t = t * y_1 + (1 - t) * y_0
        v_t = self.model(y_t, t, class_labels)
        return ((v_t - (y_1 - y_0)) ** 2).mean()
        