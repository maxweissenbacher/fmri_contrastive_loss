from data.dataloading import load_data, ourDataset, train_test_split
from pathlib import Path
import torch
from datetime import datetime
from metrics.evaluation import compute_eval_metrics
from models.random_features import RandomFeatures


if __name__ == '__main__':
    time_run_started = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

    # Training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 512
    num_patients = None
    file_format = 'zarr'

    # Hyperparameters
    model_params = {
        'num_features': 10,
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

    model = RandomFeatures(**model_params)

    # Evaluating model performance
    compute_eval_metrics(
        data=data_split,
        model=model,
        device=device,
        batch_size=batch_size,
        metric='cosine',
    )

    print('Finished executing.')

