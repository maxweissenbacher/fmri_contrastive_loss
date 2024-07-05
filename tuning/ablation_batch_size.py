import pickle
from data.dataloading import train_test_split, load_features
import torch
import optuna
from training.trainer import Trainer
from models.nn import Net
from metrics.evaluation import compute_eval_metrics


def objective(trial, num_epochs=2000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_names = ['mean', 'ar1']  # Fix feature selection for now - probs do this for a few different ones
    num_features = len(feature_names)

    # Choose hyperparameters
    # Batch size between 32 and 4096 (i.e. full dataset)
    batch_size = int(trial.suggest_categorical("batch_size", [2**i for i in range(6, 13)]))

    # Load data
    data = load_features(path="data", names=feature_names)
    data_split = train_test_split(data, perc=.75)  # Random train test split

    # Hyperparameters
    model_params = {
        'dim': 360 * num_features,
        'width': 64,
        'depth': 2,
        'nenc': 1,
    }
    loss_params = {
        'eps': 1.5,
        'alpha': 1.0,
    }
    lr = 1e-3

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
        within_batch=True,  # Make sure loss is computed only within each batch
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


def ablation_study_batch_size():
    search_space = {'batch_size': [2**i for i in range(6, 13)]}
    study = optuna.create_study(
        direction="maximize",
        study_name="Effect of batch size for within-batch loss computation",
        sampler=optuna.samplers.GridSampler(search_space)
    )
    study.optimize(objective, n_trials=len(search_space['batch_size']), timeout=72000)  # Timeout is 20 hours

    filename = "./tuning/outputs/batch_size_study.pkl"
    with open(filename, "wb") as f:
        pickle.dump(study, f)
    print(f"Saved study '{study.study_name}' to pickle {filename}.")
