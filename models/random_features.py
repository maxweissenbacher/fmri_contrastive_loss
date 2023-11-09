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


class RandomFeatures(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.flatten = nn.Flatten()
        self.inds = np.random.permutation(360 * 2) < num_features

    def forward(self, input):
        x = self.flatten(input)  # equivalently, x = x.view(x.size()[0], -1)
        output = x[..., self.inds]
        return output
