import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from data.dataloading import ourDataset, DataLoader


def compute_eval_metrics(
        data,
        model,
        device,
        batch_size,
        threshold=0.1,
        metric=None,
        normalise=False,  # Normalise output of model to [0,1]; only makes sense if output is 1D
):
    print('Evaluating model performance...')

    if metric not in ['euclidean', 'cosine']:
        raise NotImplementedError("A metric must be chosen for the loss: either 'euclidean' or 'cosine'.")
    if normalise and metric == 'cosine':
        raise NotImplementedError("Using cosine similarity and normalising does not make sense.")

    # Extract data, create dataloaders
    same_subject_train = data['train']['same_subject']
    same_subject_val = data['val']['same_subject']
    diff_subject_train = data['train']['diff_subject']
    diff_subject_val = data['val']['diff_subject']
    autocorr_features_train = data['train']['autocorrelation_and_variation']
    autocorr_features_val = data['val']['autocorrelation_and_variation']

    # Convert to dataset and dataloader
    dataset_train = ourDataset(autocorr_features_train, device=device)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_val = ourDataset(autocorr_features_val, device=device)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    # Evaluate model performance on training set
    output_train = []
    index_train = []
    for it, (d, batch_idx) in enumerate(dataloader_train):
        output = model.forward(d)
        output_train.append(output.detach().cpu())
        index_train.append(batch_idx.detach().cpu())
    output_train = torch.vstack(output_train)
    index_train = torch.hstack(index_train)

    if normalise:
        # Normalise output to [0,1]
        output_train = output_train - np.min(output_train)
        output_train = output_train / np.max(output_train)

    if metric == 'euclidean':
        # Euclidean distance between embeddings
        dist_train = torch.cdist(output_train, output_train, p=2)
    elif metric == 'cosine':
        # Cosine similarity between embeddings
        dist_train = nn.CosineSimilarity(dim=-1)(output_train[..., None, :, :], output_train[..., :, None, :])
    dist_train = dist_train.detach().numpy()
    # get submatrices of same and diff
    same_train_true = same_subject_train[index_train[:, None], index_train[None, :]]
    diff_train_true = diff_subject_train[index_train[:, None], index_train[None, :]]

    if metric == 'euclidean':
        same_train_pred = (dist_train <= threshold)
        diff_train_pred = (dist_train > threshold)
    elif metric == 'cosine':
        same_train_pred = (dist_train >= 1 - 2 * threshold)
        diff_train_pred = (dist_train < 1 - 2 * threshold)

    accuracy_same_train = np.sum((same_train_pred == same_train_true) & (same_train_true == True)) / np.sum(same_train_true)
    accuracy_diff_train = np.sum((diff_train_true == diff_train_pred) & (diff_train_true == True)) / np.sum(diff_train_true)
    print(f'Training set: accuracy on same: {accuracy_same_train:.4f}')
    print(f'Training set: accuracy on different: {accuracy_diff_train:.4f}')

    # Evaluate model performance on testing set
    # Pass through model
    output_val = []
    index_val = []
    for it, (d, batch_idx) in enumerate(dataloader_val):
        output = model.forward(d)
        output_val.append(output.detach().cpu())
        index_val.append(batch_idx.detach().cpu())
    output_val = torch.vstack(output_val)
    index_val = torch.hstack(index_val)

    if normalise:
        # Normalise output to [0,1]
        output_val = output_val - np.min(output_val)
        output_val = output_val / np.max(output_val)

    if metric == 'euclidean':
        # Euclidean distance between embeddings
        dist_val = torch.cdist(output_val, output_val, p=2)
    elif metric == 'cosine':
        # Cosine similarity between embeddings
        dist_val = nn.CosineSimilarity(dim=-1)(output_val[..., None, :, :], output_val[..., :, None, :])
    dist_val = dist_val.detach().numpy()
    # get submatrices of same and diff
    same_val_true = same_subject_val[index_val[:, None], index_val[None, :]]
    diff_val_true = diff_subject_val[index_val[:, None], index_val[None, :]]

    if metric == 'euclidean':
        same_val_pred = (dist_val <= threshold)
        diff_val_pred = (dist_val > threshold)
    elif metric == 'cosine':
        same_val_pred = (dist_val >= 1 - 2 * threshold)
        diff_val_pred = (dist_val < 1 - 2 * threshold)

    accuracy_same_val = np.sum((same_val_pred == same_val_true) & (same_val_true == True)) / np.sum(same_val_true)
    accuracy_diff_val = np.sum((diff_val_true == diff_val_pred) & (diff_val_true == True)) / np.sum(diff_val_true)
    print(f'Testing set:  accuracy on same: {accuracy_same_val:.4f}')
    print(f'Testing set:  accuracy on different: {accuracy_diff_val:.4f}')

    # Make histograms for training and test set
    fig, axs = plt.subplots(1, 2)

    # For testing set
    sames = dist_val[same_val_true]
    diffs = dist_val[diff_val_true]
    axs[0].hist(sames, bins=100, density=True, alpha=.5)
    axs[0].hist(diffs, bins=100, density=True, alpha=.5)
    axs[0].set_xlabel(("Distance" if metric == 'euclidean' else "Cosine similarity") + " on validation data")
    axs[0].legend(["Same subject", "Different subject"])
    title = "Testing set\n"
    title += f"diff of medians: {np.median(sames) - np.median(diffs):.2f}"
    title += f" | diff of means: {np.mean(sames) - np.mean(diffs):.2f}\n"
    title += f"acc on same: {accuracy_same_val:.4f} | acc on different: {accuracy_diff_val:.4f}"
    axs[0].set_title(title)

    # For training set
    sames = dist_train[same_train_true]
    diffs = dist_train[diff_train_true]
    axs[1].hist(sames, bins=100, density=True, alpha=.5)
    axs[1].hist(diffs, bins=100, density=True, alpha=.5)
    axs[1].set_xlabel(("Distance" if metric == 'euclidean' else "Cosine similarity") + " on training data")
    axs[1].legend(["Same subject", "Different subject"])
    title = "Training set\n"
    title += f"diff of medians: {np.median(sames) - np.median(diffs):.2f}"
    title += f" | diff of means: {np.mean(sames) - np.mean(diffs):.2f}\n"
    title += f"acc on same: {accuracy_same_train:.4f} | acc on different: {accuracy_diff_train:.4f}"
    axs[1].set_title(title)

    fig.set_size_inches(15, 7)
    plt.suptitle(f"Histograms for {str(model)} model")
    plt.savefig(f'./figures/histogram_{str(model)}_autocorr_combined.png', bbox_inches='tight')
    plt.close()

    # Print estimated density of the output for training and testing set
    fig, ax = plt.subplots(1, 1)
    sns.kdeplot(data=output_train, ax=ax, palette=['blue'], label='Training')
    sns.kdeplot(data=output_val, ax=ax, palette=['red'], label='Validation')
    plt.title('Approximate density of model output')
    plt.legend()
    plt.savefig(f'./figures/model_{str(model)}_density_autocorr.png', bbox_inches='tight')
    plt.close()