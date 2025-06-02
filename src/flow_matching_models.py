import torch
from torch import nn

class UnconditionalFlowMatchingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.flow_model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )


    def forward(self, y, t):
        return self.flow_model(torch.cat([y, t], dim=1))
    

class ConditionalFlowMatchingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=5, embedding_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.embedding_model = nn.Embedding(num_classes, embedding_dim)

        self.flow_model = nn.Sequential(
            nn.Linear(input_dim + 1 + embedding_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, y, t, class_label):
        embedding = self.embedding_model(class_label)
        return self.flow_model(torch.cat([y, t, embedding], dim=1))



