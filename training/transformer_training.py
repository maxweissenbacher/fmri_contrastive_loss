from data.dataloading import load_data, ourDataset
from pathlib import Path
import torch
from models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader
from models.losses.contrastive_losses import contr_loss_simple
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = 'cpu'
    batch_size = 512

    # Load data
    cwd = Path.cwd()  # Current working directory
    rel_path = 'data/timeseries_max_all_subjects.hdf5'  # Relative path from project directory, depends on where you store the data
    file_path = (cwd.parent / rel_path).resolve()
    data = load_data(file_path, number_patients=10)  # Load only subset of patients for testing for now

    # data is a dict, extract the relevant entries
    raw_features = data['raw']
    same_subject = data['same_subject']
    diff_subject = data['diff_subject']
    dataset = ourDataset(raw_features, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    print(raw_features.shape)

    # Hyperparameters
    n_chans = raw_features.shape[1]
    d_init = raw_features.shape[2]
    d_model = 10
    n_hidden = 10
    n_head = 5
    n_layers = 1
    lr = 1e-5
    eps = 10
    num_epochs = 500

    # Instantiate model, optimiser and learning rate scheduler
    model = VisionTransformer(n_chans, d_init, d_model, n_hidden, n_head, n_layers, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 0.0)

    # Testing model output
    output = model(torch.tensor(raw_features))
    print(output.shape)

    # Set up for logging training metrics
    losses = []

    # Training loop
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        avg_loss = []  # average loss across batches
        avg_gn = []    # average gradient norm across batches
        # iterate over all data
        for (d, batch_idx) in dataloader:
            batch_idx = batch_idx.detach().numpy()
            # get submatrices of same and diff
            same = torch.tensor(same_subject[batch_idx[:, None], batch_idx[None, :]])
            diff = torch.tensor(diff_subject[batch_idx[:, None], batch_idx[None, :]])
            # pass through model
            output = model.forward(d)
            # Compute the loss value
            # Currently uses the 'simple' contrastive loss!
            loss = contr_loss_simple(output, same, diff, eps)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Compute the gradients
            loss.backward()
            # Gradient clipping
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            # Take the optimisation step
            optimizer.step()

            # Remember loss and gradient norm per batch
            avg_loss.append(loss.detach().item())
            avg_gn.append(gn.detach().item())
        # Update learning rate
        scheduler.step()

        # Update progress bar
        description = (
                        f'Loss {np.array(avg_loss).mean():.2f} | '  
                        f'grad norm {np.array(avg_gn).mean():.2f} | '
                        f'learning rate {optimizer.param_groups[0]["lr"]:.9f}'
        )
        pbar.set_description(description)

        # Logging
        losses.append(np.array(avg_loss).mean())
