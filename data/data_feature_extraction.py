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
    if i > 50:
        break
    if not data[s]['functional/is_complete'][()]:
        continue
    tss = [data[s]['functional'][str(i + 1)]['timeseries']
           for i in range(0, 4)]
    timeseries.append(np.asarray(tss))
    valid_subjects.append(s)

print('stop here')

timeseries = np.asarray(timeseries)
ts2 = timeseries.reshape(-1, timeseries.shape[2], timeseries.shape[3])
assert np.all(ts2[1] == timeseries[0, 1])
assert np.all(ts2[4] == timeseries[1, 0])

feature_bank = []
for i, subj in enumerate(subjects):
    print(i)
    if i > 49:
        break
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

print('stop here')

np.save("feature_bank.npy", feature_bank)

scannum = np.tile([0, 1, 2, 3], timeseries.shape[0])
subjnum = np.repeat(range(0, timeseries.shape[0]), 4)
label = list(zip(subjnum, scannum))