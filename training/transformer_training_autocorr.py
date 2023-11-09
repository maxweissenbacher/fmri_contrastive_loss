from data.dataloading import load_data, ourDataset, train_test_split
from pathlib import Path
import torch
from models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader
from models.losses.contrastive_losses import contr_loss_simple
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import gc
from datetime import datetime
from metrics.evaluation import compute_eval_metrics


def transformer_train_autocorr(num_epochs, num_patients, batch_size, hpc=False, file_format="HDF5"):
    time_run_started = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'Transformer'

    # Log the device that's being used
    if hpc:
        with open('./logs/device.txt', "w") as f:
            f.write(str(device))

    print(f"Using device {device}")

    # Load data
    cwd = Path.cwd()  # Current working directory
    if file_format == 'zarr':
        rel_path = 'data/hcp1200.zarr.zip'
    elif file_format == 'HDF5':
        rel_path = 'data/timeseries_max_all_subjects.hdf5'
    else:
        raise NotImplementedError
    file_path = (cwd / rel_path).resolve()
    data = load_data(file_path, format=file_format, number_patients=num_patients, normalize=True, verbose=True)

    # data is a dict of numpy arrays, extract the relevant entries
    autocorr_features = data['autocorrelation_and_variation']
    same_subject = data['same_subject']
    diff_subject = data['diff_subject']
    print(f"Raw features have shape {autocorr_features.shape}")

    # Train test split with deterministic RNG
    # The seed was generated by calling
    # import secrets; secrets.randbits(128)
    data_split = train_test_split(data, perc=.75, seed=251668716030294078557169461317962359616)

    same_subject_train = data_split['train']['same_subject']
    same_subject_val = data_split['val']['same_subject']
    diff_subject_train = data_split['train']['diff_subject']
    diff_subject_val = data_split['val']['diff_subject']
    autocorr_features_train = data_split['train']['autocorrelation_and_variation']
    autocorr_features_val = data_split['val']['autocorrelation_and_variation']

    # Convert to dataset and dataloader
    dataset_train = ourDataset(autocorr_features_train, device=device)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_val = ourDataset(autocorr_features_val, device=device)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    # Free up memory
    del data

    print("here")

    # Hyperparameters
    length = autocorr_features.shape[1]
    d_init = autocorr_features.shape[2]
    d_model = 45  # 10
    n_hidden = 20  # 10
    # d_model must be divisable by n_head
    n_head = 5  # 5
    n_layers = 3
    lr = 1e-5
    eps = 0.1

    # Instantiate model, optimiser and learning rate scheduler
    model = VisionTransformer(length, d_init, d_model, n_hidden, n_head, n_layers, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    # Testing model output
    test_num = min(10, autocorr_features.shape[0])
    output = model(autocorr_features[:test_num].to(device))
    print(f'Tested successfully. Model output has shape {output.shape}')

    # Set up for logging training metrics
    losses = []

    # Training loop
    start_time = time.time()
    pbar = tqdm(range(num_epochs))
    for _ in pbar:
        avg_loss = []  # average loss across batches
        avg_gn = []    # average gradient norm across batches
        # iterate over all data
        for (d, batch_idx) in dataloader_train:
            batch_idx = batch_idx.detach().numpy()
            # get submatrices of same and diff
            # ----------
            # TO-DO: Check if using the same_subject matrix as opposed to the same_subject_train matrix is correct!
            # ----------
            same = torch.tensor(same_subject[batch_idx[:, None], batch_idx[None, :]]).to(device)
            diff = torch.tensor(diff_subject[batch_idx[:, None], batch_idx[None, :]]).to(device)
            output = model.forward(d)
            loss = contr_loss_simple(output, same, diff, eps)
            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            optimizer.step()
            # Remember loss and gradient norm per batch
            avg_loss.append(loss.detach().item())
            avg_gn.append(gn.detach().item())

        # Evaluate model on validation set
        #with torch.no_grad():
        #    avg_loss = []  # We hope this approximates the true loss well but likely misses lots of interactions
        #    for (d, batch_idx) in dataloader_val:
        #        batch_idx = batch_idx.detach().numpy()
        #        same = torch.tensor(same_subject[batch_idx[:, None], batch_idx[None, :]]).to(device)
        #        diff = torch.tensor(diff_subject[batch_idx[:, None], batch_idx[None, :]]).to(device)
        #        output = model.forward(d)
        #        loss = contr_loss_simple(output, same, diff, eps)
        #        avg_loss.append(loss.detach().item())

        #    val_loss = np.array(avg_loss).mean()

        # Update learning rate
        #scheduler.step(val_loss)

        # Update progress bar
        description = (
                        f'Loss {np.array(avg_loss).mean():.2f} | '  
                        f'grad norm {np.array(avg_gn).mean():.2f} | '
                        f'learning rate {optimizer.param_groups[0]["lr"]:.9f}'
        )
        pbar.set_description(description)

        # Logging
        losses.append(np.array(avg_loss).mean())

    end_time = time.time()

    print(f"Training loop ({num_epochs} epochs) executed in {end_time-start_time:.2f}s, or {(end_time-start_time)/num_epochs:.2f}s per epoch.")

    # Save losses to txt file
    filename = f"./logs/loss_autocorr_{time_run_started}.txt"
    with open(filename, "w") as f:
        np.savetxt(f, np.array(losses))

    filename = f"./logs/loss_autocorr_{time_run_started}.png"
    plt.plot(losses)
    plt.title('Losses (averaged over batches) by epoch')
    if hpc:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

    print('Evaluating model performance:')
    compute_eval_metrics(
        dataloader_train,
        same_subject_train,
        diff_subject_train,
        dataloader_val,
        same_subject_val,
        diff_subject_val,
        model_name,
        model
    )

    # Garbage collection
    del dataset_val, dataloader_val, dataset_train, dataloader_train
    gc.collect()

