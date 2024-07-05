import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, dim, nenc, width, depth):
        super().__init__()
        self.flatten = nn.Flatten()
        modules = [nn.BatchNorm1d(dim), nn.Linear(dim, width), nn.ReLU()]
        for dp in range(depth):
            modules.append(nn.BatchNorm1d(width))
            modules.append(nn.Linear(width, width))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(width, nenc))  # output is thus something of size nenc, the size of the encoding
        # modules.append(nn.Linear(width, nenc+1))  # output is thus something of size nenc+1, the size of the encoding
        self.sequential = nn.Sequential(*modules)

    def forward(self, input):
        x = self.flatten(input)  # equivalently, x = x.view(x.size()[0], -1)
        x = self.sequential.forward(x)
        # x = F.normalize(x, dim=-1)  # Normalise the output so it's contained on the sphere of dimension nenc
        return x

    def __repr__(self):
        return "neural_net"
