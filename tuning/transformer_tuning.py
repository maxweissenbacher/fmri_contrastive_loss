from data.dataloading import load_data, ourDataset, train_test_split
from pathlib import Path
import torch
from models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader
from models.losses.contrastive_losses import contr_loss_simple
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import optuna
from optuna.trial import TrialState


def objective(trial, num_patients=10, batch_size=256, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    cwd = Path.cwd()  # Current working directory
    rel_path = 'data/timeseries_max_all_subjects.hdf5'  # Relative path from project directory, depends on where you store the data
    #file_path = (cwd / rel_path).resolve()
    file_path = (cwd.parent / rel_path).resolve()
    data = load_data(file_path, number_patients=num_patients)  # Load only subset of patients for testing for now

    # data is a dict of numpy arrays, extract the relevant entries
    # remove this
    raw_features = data['raw']
    same_subject = data['same_subject']
    diff_subject = data['diff_subject']

    data_split = train_test_split(data, .75)

    raw_features_train = data_split['train']['raw']
    raw_features_val = data_split['val']['raw']

    # Construct dataset and dataloader
    dataset_train = ourDataset(raw_features_train, device=device)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
    dataset_val = ourDataset(raw_features_val, device=device)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size)

    # Hyperparameters
    length = raw_features.shape[1]
    d_init = raw_features.shape[2]
    d_model = trial.suggest_int("d_model", 10, 100, step=10)
    n_hidden = trial.suggest_int("n_hidden", 10, 50, step=10)
    n_head = 5  # d_model must be dividable by n_head
    n_layers = trial.suggest_int("n_layers", 1, 5)
    lr = 1e-5
    eps = trial.suggest_float("eps", 0.1, 10)

    # Instantiate model, optimiser and learning rate scheduler
    model = VisionTransformer(length, d_init, d_model, n_hidden, n_head, n_layers, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 0.0)

    # Training loop
    for epoch in range(num_epochs):
        # Iterate over the entire dataset
        for (d, batch_idx) in dataloader_train:
            batch_idx = batch_idx.detach().numpy()
            same = torch.tensor(same_subject[batch_idx[:, None], batch_idx[None, :]]).to(device)
            diff = torch.tensor(diff_subject[batch_idx[:, None], batch_idx[None, :]]).to(device)
            output = model.forward(d)
            loss = contr_loss_simple(output, same, diff, eps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            optimizer.step()

        # Update learning rate
        scheduler.step()

        # Evaluate model on validation set
        with torch.no_grad():
            avg_loss = []  # We hope this approximates the true loss well but likely misses lots of interactions
            for (d, batch_idx) in dataloader_val:
                batch_idx = batch_idx.detach().numpy()
                same = torch.tensor(same_subject[batch_idx[:, None], batch_idx[None, :]]).to(device)
                diff = torch.tensor(diff_subject[batch_idx[:, None], batch_idx[None, :]]).to(device)
                output = model.forward(d)
                loss = contr_loss_simple(output, same, diff, eps)
                avg_loss.append(loss.detach().item())

        val_loss = np.array(avg_loss).mean()
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


if __name__ == '__main__':
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, timeout=600)  # Timeout is in seconds

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))