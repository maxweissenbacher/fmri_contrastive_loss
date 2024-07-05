from data.dataloading import load_data, ourDataset, train_test_split
from pathlib import Path
import torch
import torch.nn as nn
from datetime import datetime
from metrics.evaluation import compute_eval_metrics


if __name__ == '__main__':
    time_run_started = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

    # Training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 512
    num_patients = None
    file_format = 'zarr'

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

    class OneFeature(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.flatten = nn.Flatten()
            self.idx=idx

        def forward(self, input):
            x = self.flatten(input)  # equivalently, x = x.view(x.size()[0], -1)
            x = x[..., self.idx]
            return x.view(-1, 1)

        def __repr__(self):
            return f"one_feature_idx_{self.idx}"

    max_acc = -torch.inf
    best_idx = None
    for i in range(720):
        model = OneFeature(idx=i)
        metrics = compute_eval_metrics(
            data=data_split,
            model=model,
            device=device,
            batch_size=batch_size,
            metric='euclidean',
            normalise=True,
            create_figures=False,
        )
        # Finding the index that gives maximal recall on validation data for different subjects
        if metrics['acc_diff_val'] > max_acc:
            max_acc = metrics['acc_diff_val']
            best_idx = i

    print(f"Best feature (maximal recall on different subjects on validation set) index is {best_idx}.")

    model = OneFeature(idx=best_idx)
    metrics = compute_eval_metrics(
        data=data_split,
        model=model,
        device=device,
        batch_size=batch_size,
        metric='euclidean',
        normalise=True,
        create_figures=True,
    )

    print('Finished executing.')

