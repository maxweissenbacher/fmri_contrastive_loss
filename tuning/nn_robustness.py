import pickle
from data.dataloading import train_test_split, load_features, k_fold_split
import torch
import optuna
from training.trainer import Trainer
from models.nn import Net
from metrics.evaluation import compute_eval_metrics
import numpy as np
from functools import partial


def objective(trial, feature_names, batch_size=512, num_epochs=3000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # feature_names = ['mean']  # Fix feature selection for now - probs do this for a few different ones
    num_features = len(feature_names)

    # Choose hyperparameters
    width = trial.suggest_categorical("width", [8, 16, 32, 64, 128, 256, 512])
    depth = trial.suggest_int("depth", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    eps = trial.suggest_float("eps", 0.01, 10., log=True)
    alpha = trial.suggest_float("alpha", 0.5, 1.5)

    # Load data
    data = load_features(path="data", names=feature_names)
    data_split = k_fold_split(data, k=5, seed=2352937509)
    # data_split = train_test_split(data, perc=.75)  # Random train test split

    # Hyperparameters
    model_params = {
        'dim': 360 * num_features,
        'width': width,
        'depth': depth,
        'nenc': 1,
    }
    loss_params = {
        'eps': eps,
        'alpha': alpha,
    }

    average_losses = []
    average_metrics = []
    for d in data_split:
        # Training
        trainer = Trainer(
            model=Net,
            model_params=model_params,
            loss_params=loss_params,
            labels=d['train']['label'],
            features=d['train']['features'],
            device=device,
            lr=lr,
            batch_size=batch_size,
            within_batch=False,
        )
        losses = trainer.train(num_epochs)
        average_losses.append(np.asarray(losses))

        # Evaluating model performance
        metrics = compute_eval_metrics(
            data=d,
            model=trainer.model,
            device=device,
            batch_size=batch_size,
            metric='euclidean',
            feature_name='+'.join(feature_names),
        )

        average_metrics.append(metrics['icc_val'])

    # Log average losses (averaged over folds)
    average_losses = np.asarray(average_losses)
    average_losses = average_losses.mean(axis=0)
    for epoch, l in enumerate(average_losses):
        trial.report(l, epoch)

    average_metrics = np.asarray(average_metrics)

    return np.mean(average_metrics)


def nn_tuning(feature_names):
    study = optuna.create_study(
        direction="maximize",
        study_name="NN_hyperparameter_tuning_mean",
        sampler=optuna.samplers.TPESampler()
    )
    study.optimize(partial(objective, feature_names=feature_names), n_trials=1000, timeout=86400)  # Timeout is 24 hours

    filename = "./tuning/outputs/nn_tuning_" + '_'.join(feature_names) + ".pkl"
    with open(filename, "wb") as f:
        pickle.dump(study, f)
    print(f"Saved study '{study.study_name}' to pickle {filename}.")
