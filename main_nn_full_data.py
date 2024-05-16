from data.dataloading import load_data, ourDataset, train_test_split, load_features
from pathlib import Path
import torch
from models.nn import Net
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from metrics.evaluation import compute_eval_metrics
from training.trainer import Trainer
import argparse


if __name__ == '__main__':
    time_run_started = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

    # Parse feature name from command line
    parser = argparse.ArgumentParser(description='NN training')
    parser.add_argument('feature_name', type=str, help='Name of feature')
    args = parser.parse_args()
    name = args.feature_name

    # Training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 512
    compute_loss_within_batch = False  # Compute loss over entire train set if False, only within batch if True
    num_epochs = 3000

    # Tuned hyperparameters
    model_params = {
        'dim': 360,
        'width': 256 if name == 'ar1' else 512,
        'depth': 1,
        'nenc': 1,
    }
    loss_params = {
        'eps': 100.,
        'alpha': 1.0,
    }
    feature_names = [name]

    print(f"Using device {device}")

    # Load data
    data = load_features("data", feature_names)
    # Train test split with deterministic RNG
    data_split = train_test_split(data, perc=.75, seed=513670296)

    # Training
    trainer = Trainer(
        model=Net,
        model_params=model_params,
        loss_params=loss_params,
        labels=data['label'],
        features=data['features'],
        device=device,
        lr=1e-3,
        batch_size=batch_size,
        within_batch=compute_loss_within_batch,
    )
    losses = trainer.train(num_epochs)

    # Save losses to txt file
    filename = f"./logs/loss_{str(trainer.model)}_autocorr_{time_run_started}.txt"
    with open(filename, "w") as f:
        np.savetxt(f, np.array(losses))
    # Plot losses and save to figure
    filename = f"./logs/loss_{str(trainer.model)}_autocorr_{time_run_started}.png"
    plt.plot(losses)
    plt.title('Losses (averaged over batches) by epoch')
    plt.savefig(filename)
    plt.close()

    # Evaluating model performance
    compute_eval_metrics(
        data=data_split,
        model=trainer.model,
        device=device,
        batch_size=batch_size,
        metric='euclidean',
        feature_name='+'.join(feature_names),
    )

    # Save model
    filename = f"outputs/model_ENTIREDATASET_{str(trainer.model)}"
    filename += f"_WIDTH-{model_params['width']}_DEPTH-{model_params['depth']}"
    filename += f"_FEATURES-{'+'.join(feature_names)}.pt"
    torch.save(trainer.model.state_dict(), filename)

    # Done
    print(f'Finished executing. Model saved to {filename}.')

