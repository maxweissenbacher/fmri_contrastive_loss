from data.dataloading import load_data, ourDataset, train_test_split
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from metrics.evaluation import compute_eval_metrics
from training.trainer import Trainer
from models.linear import LinearLayer


if __name__ == '__main__':
    time_run_started = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

    # Training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 512
    num_patients = None
    num_epochs = 2000
    file_format = 'zarr'

    # Hyperparameters
    model_params = {
        'in_dim': 720,  # = 2 * 360
        'out_dim': 10,
    }
    loss_params = {
        'eps': 1.4,
        'alpha': 0.8,
    }

    print(f"Using device {device}")

    # Load data
    if file_format == 'zarr':
        rel_path = 'data/hcp1200.zarr.zip'
    elif file_format == 'HDF5':
        rel_path = 'data/timeseries_max_all_subjects.hdf5'
    else:
        raise NotImplementedError
    cwd = Path.cwd()
    file_path = (cwd / rel_path).resolve()
    data = load_data(file_path, number_patients=num_patients, normalize=True, verbose=True)

    # Train test split with deterministic RNG
    data_split = train_test_split(data, perc=.75, seed=251668716030294078557169461317962359616)
    del data

    # Training
    trainer = Trainer(
        model=LinearLayer,
        model_params=model_params,
        loss_params=loss_params,
        data=data_split['train'],
        device=device,
        lr=1e-5,
        batch_size=batch_size,
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
        metric='cosine',
    )

print('Finished executing.')

