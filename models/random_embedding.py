import numpy as np
import matplotlib.pyplot as plt
import h5py  # https://pypi.org/project/h5py/
import spatiotemporal  # https://pypi.org/project/spatiotemporal/
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import scipy
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import gc
import math
from pathlib import Path
from data.dataloading import load_data, ourDataset


class RandomEmbedding(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, input):
        batch_size = input.size()[0]
        x = torch.normal(torch.zeros(batch_size, self.out_dim+1), 1.)
        x = F.normalize(x, dim=-1)
        return x

    def __repr__(self):
        return "random_embedding"