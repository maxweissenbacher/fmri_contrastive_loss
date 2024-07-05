import pickle
from data.dataloading import load_data, ourDataset, train_test_split
from pathlib import Path
import torch
from models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader
from models.losses.contrastive_losses import contr_loss_simple
import numpy as np
import optuna


def objective(trial, num_patients=450, batch_size=512, num_epochs=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    cwd = Path.cwd()  # Current working directory
    rel_path = 'data/timeseries_max_all_subjects.hdf5'  # Relative path from project directory, depends on where you store the data
    file_path = (cwd / rel_path).resolve()
    data = load_data(file_path, number_patients=num_patients, normalize=True)
    data_split = train_test_split(data, perc=.75)  # Random train test split

    autocorr_features_train = data_split['train']['autocorrelation_and_variation']
    autocorr_features_val = data_split['val']['autocorrelation_and_variation']
    same_subject_train = data_split['train']['same_subject']
    same_subject_val = data_split['val']['same_subject']
    diff_subject_train = data_split['train']['diff_subject']
    diff_subject_val = data_split['val']['diff_subject']

    # Construct dataset and dataloader
    dataset_train = ourDataset(autocorr_features_train, device=device)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
    dataset_val = ourDataset(autocorr_features_val, device=device)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size)

    # Hyperparameters
    length = autocorr_features_train.shape[1]
    d_init = autocorr_features_train.shape[2]
    d_model = trial.suggest_int("d_model", 10, 50, step=5)
    n_hidden = trial.suggest_int("n_hidden", 10, 50, step=5)
    n_head = 5  # d_model must be dividable by n_head
    n_layers = trial.suggest_int("n_layers", 1, 3)
    lr = 1e-5  # Tuned by fixing other parameters, this seems like a reasonable choice
    eps = trial.suggest_float("eps", 0.1, 10)

    # Instantiate model, optimiser and learning rate scheduler
    model = VisionTransformer(length, d_init, d_model, n_hidden, n_head, n_layers, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # TO-DO: be careful about lr scheduler.... this might not be the right choice!
    # Scheduler is deactivated for now
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 0.01 * lr)

    # Training loop
    for epoch in range(num_epochs):
        # Iterate over the entire dataset
        for (d, batch_idx) in dataloader_train:
            batch_idx = batch_idx.detach().numpy()
            same = torch.tensor(same_subject_train[batch_idx[:, None], batch_idx[None, :]]).to(device)
            diff = torch.tensor(diff_subject_train[batch_idx[:, None], batch_idx[None, :]]).to(device)
            output = model.forward(d)
            loss = contr_loss_simple(output, same, diff, eps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            optimizer.step()

        # Evaluate model on validation set
        with torch.no_grad():
            avg_loss = []  # We hope this approximates the true loss well but likely misses lots of interactions
            for (d, batch_idx) in dataloader_val:
                batch_idx = batch_idx.detach().numpy()
                same = torch.tensor(same_subject_val[batch_idx[:, None], batch_idx[None, :]]).to(device)
                diff = torch.tensor(diff_subject_val[batch_idx[:, None], batch_idx[None, :]]).to(device)
                output = model.forward(d)
                loss = contr_loss_simple(output, same, diff, eps)
                avg_loss.append(loss.detach().item())

        val_loss = np.array(avg_loss).mean()
        trial.report(val_loss, epoch)

        # Update learning rate
        #scheduler.step()

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


def transformer_tuning_autocorr():
    study = optuna.create_study(direction="minimize", study_name="Transformer tuning all params autocorr")
    study.optimize(objective, n_trials=100, timeout=72000)  # Timeout is in seconds, 20 hours

    filename = "./tuning/tuning_study_autocorr.pkl"
    with open(filename, "wb") as f:
        pickle.dump(study, f)
    print(f"Saved study '{study.study_name}' to pickle {filename}.")
