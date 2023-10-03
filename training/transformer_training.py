from data.dataloading import load_data, ourDataset
from pathlib import Path
import torch
from models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader
from models.losses.contrastive_losses import contr_loss_simple
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
import gc


def transformer_train(num_epochs, num_patients, batch_size, hpc=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log the device that's being used
    if hpc:
        with open('./logs/device.txt', "w") as f:
            f.write(str(device))

    print(f"Using device {device}")

    # Load data
    cwd = Path.cwd()  # Current working directory
    rel_path = 'data/timeseries_max_all_subjects.hdf5'  # Relative path from project directory, depends on where you store the data
    file_path = (cwd / rel_path).resolve()
    #file_path = (cwd.parent / rel_path).resolve()
    data = load_data(file_path, number_patients=num_patients)  # Load only subset of patients for testing for now

    # data is a dict of numpy arrays, extract the relevant entries
    raw_features = data['raw']
    same_subject = data['same_subject']
    diff_subject = data['diff_subject']
    #print(raw_features.shape)

    # Split into train/test sets
    # Create train and test datasets
    perc_train = 75
    nr_samples = raw_features.shape[0]
    num_train = int(np.round(nr_samples / 100 * perc_train))
    print(f'Total number of scans = {nr_samples}, num of scans in training set = {num_train}, num of scans in testing set = {nr_samples - num_train}.')
    # I want to select a set of patients as train and the rest as val (point being, both should contain enough same patients)
    idxs = np.arange(nr_samples)
    idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
    # shuffle after selecting
    np.random.shuffle(idxs_train)
    np.random.shuffle(idxs_val)
    same_subject_train = same_subject[idxs_train[:, None], idxs_train[None, :]]
    same_subject_val = same_subject[idxs_val[:, None], idxs_val[None, :]]
    diff_subject_train = diff_subject[idxs_train[:, None], idxs_train[None, :]]
    diff_subject_val = diff_subject[idxs_val[:, None], idxs_val[None, :]]
    raw_features_train = raw_features[idxs_train, :]
    raw_features_val = raw_features[idxs_val, :]

    # Convert to dataset and dataloader
    dataset_train = ourDataset(raw_features_train, device=device)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)

    # Hyperparameters
    n_chans = raw_features.shape[1]
    d_init = raw_features.shape[2]
    d_model = 30  # 10
    n_hidden = 15  # 10
    # d_model must be divisable by n_head
    n_head = 5  # 5
    n_layers = 1
    lr = 1e-5
    eps = 10

    # Instantiate model, optimiser and learning rate scheduler
    model = VisionTransformer(n_chans, d_init, d_model, n_hidden, n_head, n_layers, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 0.0)

    # Testing model output
    test_num = min(10, raw_features.shape[0])
    output = model(torch.tensor(raw_features[:test_num]).to(device))
    #print(output.shape)

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
            same = torch.tensor(same_subject[batch_idx[:, None], batch_idx[None, :]]).to(device)
            diff = torch.tensor(diff_subject[batch_idx[:, None], batch_idx[None, :]]).to(device)
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

    end_time = time.time()

    print(f"Training loop ({num_epochs} epochs) executed in {end_time-start_time:.2f}s, or {(end_time-start_time)/num_epochs:.2f}s per epoch.")

    if hpc:
        with open('./logs/loss.txt', "w") as f:
            np.savetxt(f, np.array(losses))


    # Plot losses and save to figure
    plt.plot(losses)
    plt.title('Losses (averaged over batches) by epoch')
    if hpc:
        plt.savefig('./logs/loss.png')
        plt.close()
    else:
        plt.show()


    print('Evaluating model performance:')

    # Evaluate model performance on training set
    # Pass through model
    output_train = []
    index_train = []
    for it, (d, batch_idx) in enumerate(dataloader_train):
        output = model.forward(d)
        output_train.append(output.detach().cpu().numpy())
        index_train.append(batch_idx.detach().cpu().numpy())
    output_train = np.vstack(output_train)
    index_train = np.hstack(index_train)

    dist_train = cdist(output_train, output_train)
    # get submatrices of same and diff
    same_train_true = same_subject_train[index_train[:, None], index_train[None, :]]
    diff_train_true = diff_subject_train[index_train[:, None], index_train[None, :]]

    # Evaluation 0: mean / median plot;
    # We want the distance for different to be very different from distance for same
    sames = dist_train[same_train_true]
    diffs = dist_train[diff_train_true]
    plt.hist(sames, bins=100, density=True, alpha=.5)
    plt.hist(diffs, bins=100, density=True, alpha=.5)
    plt.xlabel("Distance on validation data")
    plt.legend(["Same subject", "Different subject"])
    plt.title(
        f"Histogram of training set\nDifference of medians: {np.median(sames) - np.median(diffs):.2f}\nDifference of means: {np.mean(sames) - np.mean(diffs):.2f}")
    plt.savefig('./figures/histogram_train.png', bbox_inches='tight')
    plt.close()

    # Evaluation 1: we compute the accuracy of the same/diff classification
    threshold = eps  # REQUIRES LOTS OF TUNING
    same_train_pred = (dist_train <= threshold)  # True if above threshold
    diff_train_pred = (dist_train > threshold)

    accuracy_same = np.sum((same_train_pred == same_train_true) & (same_train_true == True)) / np.sum(same_train_true)
    accuracy_diff = np.sum((diff_train_true == diff_train_pred) & (diff_train_true == True)) / np.sum(diff_train_true)
    print(f'Training set: accuracy on same: {accuracy_same:.4f}')
    print(f'Training set: accuracy on different: {accuracy_diff:.4f}')

    # Garbage collection
    del dataset_train, dataloader_train
    gc.collect()


    # Evaluate model performance on testing set
    dataset_val = ourDataset(raw_features_val, device=device)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size)

    # Pass through model
    output_val = []
    index_val = []
    for it, (d, batch_idx) in enumerate(dataloader_val):
        output = model.forward(d)
        output_val.append(output.detach().cpu().numpy())
        index_val.append(batch_idx.detach().cpu().numpy())
    output_val = np.vstack(output_val)
    index_val = np.hstack(index_val)

    dist_val = cdist(output_val, output_val)
    # get submatrices of same and diff
    same_val_true = same_subject_val[index_val[:, None], index_val[None, :]]
    diff_val_true = diff_subject_val[index_val[:, None], index_val[None, :]]

    # Evaluation 0: mean / median plot;
    # We want the distance for different to be very different from distance for same
    sames = dist_val[same_val_true]
    diffs = dist_val[diff_val_true]
    plt.hist(sames, bins=100, density=True, alpha=.5)
    plt.hist(diffs, bins=100, density=True, alpha=.5)
    plt.xlabel("Distance on validation data")
    plt.legend(["Same subject", "Different subject"])
    plt.title(
        f"Histogram of test set\nDifference of medians: {np.median(sames) - np.median(diffs):.2f}\nDifference of means: {np.mean(sames) - np.mean(diffs):.2f}")
    plt.savefig('./figures/histogram_val.png', bbox_inches='tight')
    plt.close()

    # Evaluation 1: we compute the accuracy of the same/diff classification
    threshold = eps  # REQUIRES LOTS OF TUNING
    same_val_pred = (dist_val <= threshold)  # True if above threshold
    diff_val_pred = (dist_val > threshold)

    accuracy_same = np.sum((same_val_pred == same_val_true) & (same_val_true == True)) / np.sum(same_val_true)
    accuracy_diff = np.sum((diff_val_true == diff_val_pred) & (diff_val_true == True)) / np.sum(diff_val_true)
    print(f'Testing set:  accuracy on same: {accuracy_same:.4f}')
    print(f'Testing set:  accuracy on different: {accuracy_diff:.4f}')

    # Garbage collection
    del dataset_val, dataloader_val
    gc.collect()

