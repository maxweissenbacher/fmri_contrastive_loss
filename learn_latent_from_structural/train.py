import numpy as np
import torch
from pathlib import Path
from models.nn import Net

if __name__ == '__main__':
    # Load structural data
    cwd = Path.cwd()  # Current working directory
    rel_path = 'data/structural_data/curvature.npy'
    file_path = (cwd.parent / rel_path).resolve()
    with open(file_path, "rb") as f:
        structural = np.loadtxt(f)  # shape 999 x something

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



    # Try to reproduce output from trained model from structural data