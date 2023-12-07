import pickle
from data.dataloading import train_test_split, load_features
import torch
import optuna
from training.trainer import Trainer
from models.nn import Net
from metrics.evaluation import compute_eval_metrics
import itertools
from warnings import simplefilter


MAX_NUM_COMBINED_FEATURES = 3  # We consider combinations of up to this many features


def create_feature_combinations(n):
    feature_names = ['mean', 'var', 'std', 'skew', 'kurt', 'ar1', 'ar2', 'ar3']
    feature_combinations = []
    for i in range(1, n + 1):
        feature_combinations += list(itertools.combinations(feature_names, i))
    return feature_combinations


def objective(trial, batch_size=512, num_epochs=2000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_combinations = create_feature_combinations(MAX_NUM_COMBINED_FEATURES)
    feature_names = trial.suggest_categorical('feature', feature_combinations)
    num_features = len(feature_names)

    # Load data
    data = load_features("data", feature_names)
    data_split = train_test_split(data, perc=.75)  # Random train test split

    # Hyperparameters
    model_params = {
        'dim': 360 * num_features,
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
    # Suppress UserWarnings from Optuna  #rogueCoding
    simplefilter("ignore", category=UserWarning)

    feature_combinations = create_feature_combinations(MAX_NUM_COMBINED_FEATURES)
    search_space = {'feature': feature_combinations}
    study = optuna.create_study(
        direction="maximize",
        study_name="Feature tuning",
        sampler=optuna.samplers.GridSampler(search_space)
    )
    study.optimize(objective, n_trials=len(feature_combinations), timeout=72000)  # Timeout is in seconds, 20 hours

    filename = "./tuning/feature_tuning_study.pkl"
    with open(filename, "wb") as f:
        pickle.dump(study, f)
    print(f"Saved study '{study.study_name}' to pickle {filename}.")
