import torch.nn as nn
import torch

from settings import NUM_TAGS


class BaselineModel(nn.Module):
    def __init__(self, num_classes=NUM_TAGS, input_dim=768, hidden_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.bn = nn.LayerNorm(hidden_dim)
        self.projector = nn.Linear(input_dim, hidden_dim)
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeds):
        x = [self.projector(x) for x in embeds]
        x = [v.mean(0).unsqueeze(0) for v in x]
        x = self.bn(torch.cat(x, dim=0))
        x = self.lin(x)
        outs = self.fc(x)
        return outs
