import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from data.dataloading import ourDataset, DataLoader
from metrics.icc import icc_full
from utils.utils import compute_same_diff_from_label
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, average_precision_score, RocCurveDisplay, roc_auc_score
from collections import Counter


def compute_eval_metrics(
        data,
        model,
        device,
        batch_size,
        metric=None,
        normalise=False,  # Normalise output of model to [0,1]; only makes sense if output is 1D
        create_figures=True,
        feature_name=None,
):
    print('Evaluating model performance...')

    if metric not in ['euclidean', 'cosine']:
        raise NotImplementedError("A metric must be chosen for the loss: either 'euclidean' or 'cosine'.")
    if normalise and metric == 'cosine':
        raise NotImplementedError("Using cosine similarity and normalising does not make sense.")
    if feature_name is None or not isinstance(feature_name, str):
        raise ValueError("Provide name of the features the model was trained on.")

    # Set threshold values
    # These values are percentages of the standard deviation of the latent space
    threshold_percentages = [0.1, 0.25, 0.5, 0.75, 1.]

    # Extract data, create dataloaders
    label_train = data['train']['label']
    label_val = data['val']['label']
    same_subject_train, diff_subject_train = compute_same_diff_from_label(label_train, label_train)
    same_subject_val, diff_subject_val = compute_same_diff_from_label(label_val, label_val)
    features_train = data['train']['features']
    features_val = data['val']['features']

    # Convert to dataset and dataloader
    dataset_train = ourDataset(features_train, device=device)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_val = ourDataset(features_val, device=device)
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
    subject_number_train = np.asarray(data['train']['label'][:, 0], dtype=int)[index_train]

    if normalise:
        # Normalise output to [0,1]
        output_train = output_train - torch.min(output_train)
        output_train = output_train / torch.max(output_train)

    if metric == 'euclidean':
        # Euclidean distance between embeddings
        #dist_train = torch.cdist(output_train, output_train, p=2)
        dist_train = torch.cdist(output_train.view([-1, 1]), output_train.view([-1, 1]), p=2)
    elif metric == 'cosine':
        # Cosine similarity between embeddings
        dist_train = nn.CosineSimilarity(dim=-1)(output_train[..., None, :, :], output_train[..., :, None, :])
    dist_train = dist_train.detach().numpy()
    # get submatrices of same and diff
    same_train_true = same_subject_train[index_train[:, None], index_train[None, :]]
    same_train_true = same_train_true.detach().numpy()
    diff_train_true = diff_subject_train[index_train[:, None], index_train[None, :]]
    diff_train_true = diff_train_true.detach().numpy()

    tnrs_train = []
    tprs_train = []
    f1s_train = []
    for threshold_perc in threshold_percentages:
        threshold = (threshold_perc * torch.std(output_train)).numpy()
        if metric == 'euclidean':
            same_train_pred = (dist_train <= threshold)
            diff_train_pred = (dist_train > threshold)
        elif metric == 'cosine':
            same_train_pred = (dist_train >= 1 - 2 * threshold)
            diff_train_pred = (dist_train < 1 - 2 * threshold)
        tnr_train = np.sum(same_train_pred & same_train_true) / np.sum(same_train_true)
        tpr_train = np.sum(diff_train_pred & diff_train_true) / np.sum(diff_train_true)
        prec_train = np.sum(diff_train_pred & diff_train_true) / np.sum(diff_train_pred)
        f1 = 2 * tpr_train * prec_train / (tpr_train + prec_train)
        tnrs_train.append(tnr_train)
        tprs_train.append(tpr_train)
        f1s_train.append(f1)

    # Compute intra-rater reliability (ICC)
    if metric == 'euclidean':
        values = np.ndarray.flatten(output_train.detach().cpu().numpy())
        icc_train = icc_full(subject_number_train, values)[0]
    elif metric == 'cosine':
        # Here we have a problem: ICC is only defined for one-dimensional ratings
        # However, our model output is at least two dimensional (i.e. a point on some hypersphere).
        # Therefore, we cheat a little bit and compute the ICC of each dimension of the model output separately
        # and then average over the dimensions... this is likely not a very smart thing to do but hey.
        icc_average = 0.
        for i in range(output_train.shape[-1]):
            values = output_train[:, i].detach().cpu().numpy()
            icc_average += icc_full(subject_number_train, values)[0]
        icc_average /= output_train.shape[-1]
        icc_train = icc_average

    # Compute ROC AUC for training set
    roc_auc_train = roc_auc_score(1 - same_train_true.flatten(), dist_train.flatten())

    # Compute average precision for training set
    average_precision_train = average_precision_score(1 - same_train_true.flatten(), dist_train.flatten())

    # Compute chance level for training set (i.e. likelihood of classifying something as different correctly by chance)
    chance_level_train = np.sum(1 - same_train_true) / same_train_true.size

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
    subject_number_val = np.asarray(data['val']['label'][:, 0], dtype=int)[index_val]

    if normalise:
        # Normalise output to [0,1]
        output_val = output_val - torch.min(output_val)
        output_val = output_val / torch.max(output_val)

    if metric == 'euclidean':
        # Euclidean distance between embeddings
        # dist_val = torch.cdist(output_val, output_val, p=2)
        dist_val = torch.cdist(output_val.view([-1, 1]), output_val.view([-1, 1]), p=2)
    elif metric == 'cosine':
        # Cosine similarity between embeddings
        dist_val = nn.CosineSimilarity(dim=-1)(output_val[..., None, :, :], output_val[..., :, None, :])
    dist_val = dist_val.detach().numpy()
    # get submatrices of same and diff
    same_val_true = same_subject_val[index_val[:, None], index_val[None, :]]
    same_val_true = same_val_true.detach().numpy()
    diff_val_true = diff_subject_val[index_val[:, None], index_val[None, :]]
    diff_val_true = diff_val_true.detach().numpy()

    tnrs_val = []
    tprs_val = []
    f1s_val = []
    for threshold_perc in threshold_percentages:
        threshold = (threshold_perc * torch.std(output_val)).numpy()
        if metric == 'euclidean':
            same_val_pred = (dist_val <= threshold)
            diff_val_pred = (dist_val > threshold)
        elif metric == 'cosine':
            same_val_pred = (dist_val >= 1 - 2 * threshold)
            diff_val_pred = (dist_val < 1 - 2 * threshold)
        tnr_val = np.sum(same_val_pred & same_val_true) / np.sum(same_val_true)
        tpr_val = np.sum(diff_val_pred & diff_val_true) / np.sum(diff_val_true)
        prec_val = np.sum(diff_val_pred & diff_val_true) / np.sum(diff_val_pred)
        f1 = 2 * tpr_val * prec_val / (tpr_val + prec_val)
        tnrs_val.append(tnr_val)
        tprs_val.append(tpr_val)
        f1s_val.append(f1)

    # Compute intra-rater reliability (ICC)
    if metric == 'euclidean':
        values = np.ndarray.flatten(output_val.detach().cpu().numpy())
        icc_val = icc_full(subject_number_val, values)[0]
    elif metric == 'cosine':
        # Here we have a problem: ICC is only defined for one-dimensional ratings
        # However, our model output is at least two dimensional (i.e. a point on some hypersphere).
        # Therefore, we cheat a little bit and compute the ICC of each dimension of the model output separately
        # and then average over the dimensions... this is likely not a very smart thing to do but hey.
        icc_average = 0.
        for i in range(output_val.shape[-1]):
            values = output_val[:, i].detach().cpu().numpy()
            icc_average += icc_full(subject_number_val, values)[0]
        icc_average /= output_val.shape[-1]
        icc_val = icc_average

    # Compute ROC AUC for test set
    roc_auc_val = roc_auc_score(1 - same_val_true.flatten(), dist_val.flatten())

    # Compute average precision (similar to AUC for precision-recall curve)
    average_precision_val = average_precision_score(1 - same_val_true.flatten(), dist_val.flatten())

    # Compute chance level for testing set
    chance_level_val = np.sum(1 - same_val_true) / same_val_true.size

    if create_figures:
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
        title += f"acc on same: {tpr_val:.4f} | acc on different: {tnr_val:.4f}\n"
        title += f"ICC {icc_val:.2f}"
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
        title += f"acc on same: {tpr_train:.4f} | acc on different: {tnr_train:.4f}\n"
        title += f"ICC {icc_train:.2f}"
        axs[1].set_title(title)

        filename = f'./figures/histogram_{str(model)}_' + feature_name + '.png'
        fig.set_size_inches(15, 7)
        plt.suptitle(f"{str(model)} model")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        # Print estimated density of the output for training and testing set
        filename = f'./figures/density_{str(model)}_' + feature_name + '.png'
        fig, ax = plt.subplots(1, 1)
        sns.kdeplot(data=output_train, ax=ax, palette=['blue'], label='Training')
        sns.kdeplot(data=output_val, ax=ax, palette=['red'], label='Validation')
        plt.title('Approximate density of model output')
        plt.legend()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        # Make recall-precision curve for training set
        filename = f'./figures/recall_precision_train_{str(model)}_' + feature_name + '.png'
        precision, recall, threshold = precision_recall_curve(1-same_train_true.flatten(), dist_train.flatten())
        f1_train = 2 * precision * recall / (precision + recall)
        max_idx_train = np.argmax(f1_train)
        max_f1_train = f1_train[max_idx_train]
        max_f1_threshold_train = threshold[max_idx_train]
        display = PrecisionRecallDisplay(
            recall=recall,
            precision=precision,
            average_precision=average_precision_train,
            prevalence_pos_label=Counter((1-same_train_true).flatten().ravel())[1] / same_train_true.size,
        )
        display.plot(plot_chance_level=True)
        _ = display.ax_.set_title("Precision-recall curve | train")
        plt.legend()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        # Make recall-precision curve for testing set
        filename = f'./figures/recall_precision_test_{str(model)}_' + feature_name + '.png'
        precision, recall, threshold = precision_recall_curve(1 - same_val_true.flatten(), dist_val.flatten())
        f1_val = 2 * precision * recall / (precision + recall)
        max_idx_val = np.argmax(f1_val)
        max_f1_val = f1_val[max_idx_val]
        max_f1_threshold_val = threshold[max_idx_val]
        display = PrecisionRecallDisplay(
            recall=recall,
            precision=precision,
            average_precision=average_precision_val,
            prevalence_pos_label=Counter((1 - same_val_true).flatten().ravel())[1] / same_val_true.size,
        )
        display.plot(plot_chance_level=True)
        _ = display.ax_.set_title("Precision-recall curve | test")
        plt.legend()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        # Make ROC curve for training set
        filename = f'./figures/ROC_train_{str(model)}_' + feature_name + '.png'
        display = RocCurveDisplay.from_predictions(
            1 - same_train_true.flatten(),
            dist_train.flatten(),
            name=str(model),
            plot_chance_level=True,
        )
        _ = display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Receiver Operating Characteristic | training set\nTPR = (# pairs identified as different)/(# different pairs)",
        )
        plt.legend()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        # Make ROC curve for testing set
        filename = f'./figures/ROC_test_{str(model)}_' + feature_name + '.png'
        display = RocCurveDisplay.from_predictions(
            1 - same_val_true.flatten(),
            dist_val.flatten(),
            name=str(model),
            plot_chance_level=True,
        )
        _ = display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Receiver Operating Characteristic | testing set\nTPR = (# pairs identified as different)/(# different pairs)",
        )
        plt.legend()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    # Construct return dictionary
    train_dict = {
        'Chance level': chance_level_train,
        'ICC': icc_train,
        'ROC AUC': roc_auc_train,
        'Average precision': average_precision_train,
        f"Maximum F1 (at threshold = {max_f1_threshold_train})": max_f1_train,
    }
    train_dict.update(dict(zip([f"TNR (threshold {int(t*100)}% of std)" for t in threshold_percentages], tnrs_train)))
    train_dict.update(dict(zip([f"TPR (threshold {int(t * 100)}% of std)" for t in threshold_percentages], tprs_train)))
    train_dict.update(dict(zip([f"F1 (threshold {int(t * 100)}% of std)" for t in threshold_percentages], f1s_train)))
    # Here, TNR = (# correctly classified pairs from same subject) / (# pairs from same subjects)
    # TPR = (# correctly classified pairs from different subject) / (# pairs from different subjects)

    val_dict = {
        'Chance level': chance_level_val,
        'ICC': icc_val,
        'ROC AUC': roc_auc_val,
        'Average precision': average_precision_val,
        f"Maximum F1 (at threshold = {max_f1_threshold_val})": max_f1_val,
    }
    val_dict.update(dict(zip([f"TNR (threshold {int(t*100)}% of std)" for t in threshold_percentages], tnrs_val)))
    val_dict.update(dict(zip([f"TPR (threshold {int(t * 100)}% of std)" for t in threshold_percentages], tprs_val)))
    val_dict.update(dict(zip([f"F1 (threshold {int(t * 100)}% of std)" for t in threshold_percentages], f1s_val)))

    return_dict = {
        'train': train_dict,
        'val': val_dict,
    }

    return return_dict

