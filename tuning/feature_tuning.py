import pickle
from data.dataloading import ourDataset, train_test_split, load_features
import torch
from models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader
from models.losses.contrastive_losses import contr_loss_simple
import numpy as np
import optuna
from training.trainer import Trainer
from models.nn import Net
from metrics.evaluation import compute_eval_metrics


def objective(trial, batch_size=512, num_epochs=2500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = trial.suggest_categorical('feature', ['mean', 'var', 'std', 'skew', 'kurt', 'ar1', 'ar2', 'ar3'])
    feature_names = [name]

    # Load data
    data = load_features("data", feature_names)
    data_split = train_test_split(data, perc=.75)  # Random train test split

    # Hyperparameters
    model_params = {
        'dim': 360,
        'width': 64,
        'depth': 2,
        'nenc': 1,
    }
    loss_params = {
        'eps': 1.4,
        'alpha': 0.8,
    }

    # Training
    trainer = Trainer(
        model=Net,
        model_params=model_params,
        loss_params=loss_params,
        labels=data_split['train']['label'],
        features=data_split['train']['features'],
        device=device,
        lr=1e-3,
        batch_size=batch_size,
    )
    losses = trainer.train(num_epochs)

    for epoch, l in enumerate(losses):
        trial.report(l, epoch)

    # Evaluating model performance
    metrics = compute_eval_metrics(
        data=data_split,
        model=trainer.model,
        device=device,
        batch_size=batch_size,
        metric='euclidean',
        feature_name='+'.join(feature_names),
    )

    return metrics['icc_val']


def feature_tuning():
    search_space = {'feature': ['mean', 'var', 'std', 'skew', 'kurt', 'ar1', 'ar2', 'ar3']}
    study = optuna.create_study(
        direction="maximize",
        study_name="Feature tuning",
        sampler=optuna.samplers.GridSampler(search_space)
    )
    study.optimize(objective, n_trials=5, timeout=72000)  # Timeout is in seconds, 20 hours

    filename = "./tuning/feature_tuning_study.pkl"
    with open(filename, "wb") as f:
        pickle.dump(study, f)
    print(f"Saved study '{study.study_name}' to pickle {filename}.")
