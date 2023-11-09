import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn


def compute_eval_metrics(
        dataloader_train,
        same_subject_train,
        diff_subject_train,
        dataloader_val,
        same_subject_val,
        diff_subject_val,
        model_name,
        model,
        threshold=0.1,
        metric = None,
        normalise=False,  # Normalise output of model to [0,1]; only makes sense if output is 1D
):
    if metric not in ['euclidean', 'cosine']:
        raise NotImplementedError("A metric must be chosen for the loss: either 'euclidean' or 'cosine'.")

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
    # get submatrices of same and diff
    same_train_true = same_subject_train[index_train[:, None], index_train[None, :]]
    diff_train_true = diff_subject_train[index_train[:, None], index_train[None, :]]

    # Evaluation 1: we compute the accuracy of the same/diff classification
    same_train_pred = (dist_train <= threshold)  # True if above threshold
    diff_train_pred = (dist_train > threshold)

    accuracy_same = np.sum((same_train_pred == same_train_true) & (same_train_true == True)) / np.sum(same_train_true)
    accuracy_diff = np.sum((diff_train_true == diff_train_pred) & (diff_train_true == True)) / np.sum(diff_train_true)
    print(f'Training set: accuracy on same: {accuracy_same:.4f}')
    print(f'Training set: accuracy on different: {accuracy_diff:.4f}')

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
    # get submatrices of same and diff
    same_val_true = same_subject_val[index_val[:, None], index_val[None, :]]
    diff_val_true = diff_subject_val[index_val[:, None], index_val[None, :]]

    # Evaluation 1: we compute the accuracy of the same/diff classification
    same_val_pred = (dist_val <= threshold)  # True if above threshold
    diff_val_pred = (dist_val > threshold)

    accuracy_same = np.sum((same_val_pred == same_val_true) & (same_val_true == True)) / np.sum(same_val_true)
    accuracy_diff = np.sum((diff_val_true == diff_val_pred) & (diff_val_true == True)) / np.sum(diff_val_true)
    print(f'Testing set:  accuracy on same: {accuracy_same:.4f}')
    print(f'Testing set:  accuracy on different: {accuracy_diff:.4f}')

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
    title += f"diff of medians: {torch.median(sames) - torch.median(diffs):.2f}"
    title += f" | diff of means: {torch.mean(sames) - torch.mean(diffs):.2f}\n"
    title += f"acc on same: {accuracy_same:.4f} | acc on different: {accuracy_diff:.4f}"
    axs[0].set_title(title)

    # For training set
    sames = dist_train[same_train_true]
    diffs = dist_train[diff_train_true]
    axs[1].hist(sames, bins=100, density=True, alpha=.5)
    axs[1].hist(diffs, bins=100, density=True, alpha=.5)
    axs[1].set_xlabel(("Distance" if metric == 'euclidean' else "Cosine similarity") + " on training data")
    axs[1].legend(["Same subject", "Different subject"])
    title = "Training set\n"
    title += f"diff of medians: {torch.median(sames) - torch.median(diffs):.2f}"
    title += f" | diff of means: {torch.mean(sames) - torch.mean(diffs):.2f}\n"
    title += f"acc on same: {accuracy_same:.4f} | acc on different: {accuracy_diff:.4f}"
    axs[1].set_title(title)

    fig.set_size_inches(15, 7)
    plt.suptitle(f"Histograms for {model_name} model")
    plt.savefig(f'./figures/histogram_{model_name}_autocorr_combined.png', bbox_inches='tight')
    plt.close()

    # Print estimated density of the output for training and testing set
    fig, ax = plt.subplots(1, 1)
    sns.kdeplot(data=output_train, ax=ax, palette=['blue'], label='Training')
    sns.kdeplot(data=output_val, ax=ax, palette=['red'], label='Validation')
    plt.title('Approximate density of model output')
    plt.legend()
    plt.savefig(f'./figures/model_{model_name}_density_autocorr.png', bbox_inches='tight')
    plt.close()