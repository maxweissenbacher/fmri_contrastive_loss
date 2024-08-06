import numpy as np
import torch
from pathlib import Path
from models.nn import Net
from data.dataloading import load_features
from models.linear import LinearLayer
from torch.nn import MSELoss
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Load structural data
    cwd = Path.cwd()  # Current working directory
    rel_path = 'data/structural_data/curvature.npy'
    file_path = (cwd.parent / rel_path).resolve()
    with open(file_path, "rb") as f:
        structural = np.loadtxt(f)  # shape 999 x something
        structural = torch.tensor(structural, dtype=torch.float32)

    # Load trained model
    name = 'mean'  # mean, std, var or ar1
    model_params = {
        'dim': 360,
        'width': 512,
        'depth': 1,
        'nenc': 1,
    }
    if name == 'std':
        model_params.update({'width': 256})

    # Load model (neural net)
    rel_path = f"learn_latent_from_structural/testing_models/model_neural_net_WIDTH-{model_params['width']}_DEPTH-1_FEATURES-{name}.pt"
    model_path = (cwd.parent / rel_path).resolve()
    with open(model_path, "rb") as f:
        net = Net(**model_params)
        net.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
        net.eval()

    # Load data
    data = load_features("../data", [name])

    # Compute latent space
    latent_dimension = net(data['features'])

    # Compute mean across scans for each subject
    latent_mean = latent_dimension.view(999, 4).mean(dim=-1, keepdim=True).detach()

    # Try to reproduce output from trained model from structural data
    if not len(structural.shape) == 2:
        raise ValueError(f"Structural data must be of shape num_samples x D where D is an integer. Got {structural.shape}")
    model = LinearLayer(in_dim=structural.shape[-1], out_dim=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = MSELoss()
    losses = []

    for _ in range(250):
        output = model(structural)
        loss = criterion(output, latent_mean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())

    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.savefig('losses.png')
