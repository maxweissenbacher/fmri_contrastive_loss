import torch.nn as nn
import torch.nn.functional as F


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_dim, out_dim + 1)

    def forward(self, input):
        x = self.flatten(input)  # equivalently, x = x.view(x.size()[0], -1)
        x = self.linear(x)
        x = F.normalize(x, dim=-1)  # Normalise the output so it's contained on the sphere of dimension out_dim
        return x

    def __repr__(self):
        return "linear"
