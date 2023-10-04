import numpy as np
import matplotlib.pyplot as plt
import h5py # https://pypi.org/project/h5py/
import spatiotemporal # https://pypi.org/project/spatiotemporal/
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

'''
We follow here the vision transformer approach
Assume input is of size [number of channels, length of time series]
or [nr channels, time variables].
First step is to map this through a linear projection E to sth d-dimensional
Then add a positional encoding (just the channel nr?)
Then: 
z = [x_{class}^d, x^1E, ..., x^NE] + E_{positional}
Pass this z through:
z'_l = MSA(LN(z_{l-1})) + z_{l-1} [layernorm, multihead attntn, residual connection]
z_l = MLP(LN(z_l)) + z'_l [MLP, residual connection]
y = LN(z_L^0) [as output we take first encoded element of final layer, this is our encoding which I need to train with contrastive loss]
'''

class PositionalEncoding(nn.Module):
    """
    compute trivial positional encoding
    """
    def __init__(self, length, d_model, device):
        """
        constructor of positional encoding class
        :param n_clengthhans: first dimension of input
        :param d_model: dimension of input after mapping
        NOTE: probs this is overkill for our simple encoding but oh well
        """
        super(PositionalEncoding, self).__init__()
        #self.encoding = torch.zeros(n_chans, d_model)

        pos = torch.arange(0, length, device=device)
        pos = pos.float().unsqueeze(dim=1) # makes pos into [n_chans,1]

        #every element in d_model gets a vector [n_chans,1]; not sure this will work
        self.encoding = pos.tile((1,d_model)) # thus of size [n_chans, d_model]
        self.encoding.requires_grad = False # not sure 

    def forward(self, x): 
        return self.encoding 
    
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        # input is of size (batch_size, nr_heads, length, d_model/nr_heads)
        batch_size, head, length, d_tensor = k.size()
        # do dot-product between Query and Key^T
        k_t = k.transpose(2,3)
        # score should be [batch_size, nr_heads, length, length] - chekc
        score = (q @ k_t) / math.sqrt(d_tensor)
        score = self.softmax(score)
        # output is [batch_size, nr_heads, length, d_model/nr_heads] - check
        output = score @ v
        return output, score

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, n_head, device):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        # to get the q,v,k we do a linear multiplication
        self.w_q = nn.Linear(d_model, d_model, device=device)
        self.w_k = nn.Linear(d_model, d_model, device=device)
        self.w_v = nn.Linear(d_model, d_model, device=device)
        self.w_concat = nn.Linear(d_model, d_model, device=device)
    
    def forward(self, q, k, v):
        # take in Q,K,V and do dot product with big weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # split tensor by number of heads 
        # (i.e. split big vector into smaller chuns = equivalent to multiplying each q,v,k with many smaller weight matrices)
        q, k, v = self.split(q), self.split(k), self.split(v)

        # do attention on each head (?)
        out, attention_score = self.attention(q, k, v)

        # concatenate and pass through a linear layer; output has size [batch_size, 1, length, d_model]
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        if d_model % self.n_head != 0:
            raise AssertionError('Model dimension d_model must be dividable by number of attention heads n_head.')
        d_tensor = d_model // self.n_head
        # this splits a big vector of size (x,y,z) into (x,y,n_head,z/n_head)
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor)
        # transpose to (batch_size, nr_heads, length, d_model/nr_heads) 
        tensor = tensor.transpose(1, 2)
        return tensor

    def concat(self, tensor):
        # d_tensor is d_model / nr_heads
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        # continuous is used to avoid error from view() as transpose makes tensor non-continuous
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
    
class LayerNorm(nn.Module):
    def __init__(self, d_model, device, eps=1e-12):
        super(LayerNorm, self).__init__()
        # Parameters are Tensor subclasses, that have a very special property when used with Module s - when they’re assigned as 
        # Module attributes they are automatically added to the list of its parameters, and will appear e.g. in parameters() iterator. Also requires_grad is by default true
        self.gamma = nn.Parameter(torch.ones(d_model, device=device))
        self.beta = nn.Parameter(torch.zeros(d_model, device=device))
        self.eps = eps

    def forward(self, x):
        # mean is taken over last dimension; this is def of layernorm; so over d_model?
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # so output remains of size x
        out = (x-mean) / torch.sqrt(var + self.eps)
        return out

class OneLayer(nn.Module):

    def __init__(self, d_init, d_model, n_hidden, n_head, device):
        super(OneLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head, device=device)
        self.norm1 = LayerNorm(d_model=d_model, device=device)
        # this next piece is the positionwise feedforward - to maintain sizes
        self.linear2 = nn.Linear(d_model, n_hidden, device=device)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(n_hidden, d_model, device=device)
        self.norm2 = LayerNorm(d_model=d_model, device=device)

    def forward(self, x):
        # compute linear transformation
        x_ = self.norm1(x)
        x = self.attention(q = x_, k = x_, v = x_) + x
        x_ = self.norm2(x)
        x = self.linear3(self.relu(self.linear2(x_))) + x
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, length, d_init, d_model, n_hidden, n_head, n_layers, device):
        super().__init__()
        self.linear1 = nn.Linear(d_init, d_model, device=device)  #maps initial dimension to d_model
        self.encoding = PositionalEncoding(length=length, d_model=d_model, device=device)
        self.layers = nn.ModuleList([OneLayer(d_init=d_init, d_model=d_model, n_hidden=n_hidden, n_head=n_head, device=device)])
        self.linear2 = nn.Linear(d_model*length, 1, device=device)

    def forward(self, x):
        x = self.linear1(x)
        x = self.encoding(x) + x  # not sure if this goes right; encoding output is [N, d_model]
        for layer in self.layers:
            x = layer(x)
        # final output is of size [n_batches, length, d_model]
        # Output size is correct (as in the vision transformer paper)

        # Is this necessary? We want to just pick the first entry, like below!
        # One final linear layer over all dimensions
        x = torch.flatten(x, start_dim=1)
        x = self.linear2(x)  # shape [batch_size, 1] ... the 1 is the FINAL embedding dimension of the model
        return x

        # Previously, we had this:
        # return x[:,:,0] # not sure, but may make sense to take first element as our encoding
        # This returns something of shape [batch_size, length] (currently length = 360)
        # In the vision transformer paper this would be x[:,0,:] so that we have shape [batch_size, d_model]
        # --------------
        # In the vision transformer paper the point is that they use a class embedding token
        # at position zero, so it makes sense to keep looking at the first output. Can we do that?
        # Use whatever works at the end of the day... there is no real strong reason to prefer
        # any final layer architecture over another

if __name__=="__main__":
    # Load data
    cwd = Path.cwd()  # Current working directory
    rel_path = 'data/timeseries_max_all_subjects.hdf5'  # Relative path from project directory
    file_path = (cwd.parent / rel_path).resolve()
    data = load_data(file_path, number_patients=10)  # Load only 10 patients for testing

    device = 'cpu'

    # data is a dict, extract the relevant entries
    raw_features = data['raw']
    same_subject = data['same_subject']
    diff_subject = data['diff_subject']
    dataset = ourDataset(raw_features, device=device)
    dataloader = DataLoader(dataset, batch_size=256)
    print(raw_features.shape)
    n_chans = raw_features.shape[1]
    d_init = raw_features.shape[2] 
    d_model = 10
    n_hidden = 10
    n_head = 5
    n_layers = 1

    # Map through model to test...
    model = VisionTransformer(n_chans, d_init, d_model, n_hidden, n_head, n_layers, device)
    output = model(torch.tensor(raw_features))
    print(output.shape)