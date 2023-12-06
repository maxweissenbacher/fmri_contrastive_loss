import zarr
import numpy as np
import torch
import scipy
from pathlib import Path


cwd = Path.cwd()  # Current working directory
rel_path = 'data/hcp1200.zarr.zip'
file_path = (cwd.parent / rel_path).resolve()
store = zarr.ZipStore(file_path, mode='r')
data = zarr.group(store=store)['subjects']
subjects = list(sorted(data.keys()))
valid_subjects = []
timeseries = []
for i, s in enumerate(subjects):
    if not data[s]['functional/is_complete'][()]:
        continue
    tss = [data[s]['functional'][str(i + 1)]['timeseries']
           for i in range(0, 4)]
    timeseries.append(np.asarray(tss))
    valid_subjects.append(s)

timeseries = np.asarray(timeseries)

print('Loaded full timeseries.')

feature_bank = []
label = []
for i, subj in enumerate(valid_subjects):
    print(f"Processing subject {i}")
    for scan in range(0, 4):
        ts_np = timeseries[i][scan]
        mean = np.mean(ts_np, axis=1)
        var = np.var(ts_np, axis=1)
        std = np.std(ts_np, axis=1)
        skew = scipy.stats.skew(ts_np, axis=1)
        kurt = scipy.stats.kurtosis(ts_np, axis=1)
        ars = [np.asarray([np.corrcoef(ts_np[i, k:], ts_np[i, :-k])[0, 1] for i in range(0, len(ts_np))]) for k in
               range(1, 10)]
        features = np.concatenate([[mean, var, std, skew, kurt], ars])
        feature_bank.append(features)
        label.append((i, scan))

print('Created feature bank and labels.')

np.save("feature_bank.npy", feature_bank)
np.save("labels.npy", label)

print('Saved feature bank and labels.')
