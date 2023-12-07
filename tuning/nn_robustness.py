import pickle
from data.dataloading import train_test_split, load_features
import torch
import optuna
from training.trainer import Trainer
from models.nn import Net
from metrics.evaluation import compute_eval_metrics


def objective(trial, batch_size=512, num_epochs=3000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_names = ['mean', 'ar1']  # Fix feature selection for now - probs do this for a few different ones
    num_features = len(feature_names)

    # Choose hyperparameters
    width = trial.suggest_int("width", 32, 256, 32)
    depth = trial.suggest_int("depth", 1, 5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    # Load data
    data = load_features(path="data", names=feature_names)
    data_split = train_test_split(data, perc=.75)  # Random train test split

    # Hyperparameters
    model_params = {
        'dim': 360 * num_features,
        'width': width,
        'depth': depth,
        'nenc': 1,
    }
    loss_params = {
        'eps': 1.5,
        'alpha': 1.0,
    }

    # Training
    trainer = Trainer(
        model=Net,
        model_params=model_params,
        loss_params=loss_params,
        labels=data_split['train']['label'],
        features=data_split['train']['features'],
        device=device,
        lr=lr,
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


def nn_robustness():
    study = optuna.create_study(
        direction="maximize",
        study_name="NN_hyperparameter_robustness",
        sampler=optuna.samplers.RandomSampler()
    )
    study.optimize(objective, n_trials=100, timeout=72000)  # Timeout is in seconds, 20 hours

    filename = "./tuning/nn_robustness_study.pkl"
    with open(filename, "wb") as f:
        pickle.dump(study, f)
    print(f"Saved study '{study.study_name}' to pickle {filename}.")
