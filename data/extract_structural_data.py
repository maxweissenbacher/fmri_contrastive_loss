import zarr
import numpy as np
import torch
import scipy
from pathlib import Path

# data['852455/structural/thickness_parcellated']

# first key: patient id, then structural, then use .keys() to access the different keys


if __name__ == '__main__':
    cwd = Path.cwd()  # Current working directory
    rel_path = 'data/hcp1200.zarr.zip'
    file_path = (cwd.parent / rel_path).resolve()
    store = zarr.ZipStore(file_path, mode='r')
    data = zarr.group(store=store)['subjects']
    subjects = list(sorted(data.keys()))
    valid_subjects = []
    timeseries = []

    # Throw out incomplete scans
    for i, s in enumerate(subjects):
        if not data[s]['functional/is_complete'][()]:
            continue
        """
        tss = [data[s]['functional'][str(i + 1)]['timeseries']
               for i in range(0, 4)]
        timeseries.append(np.asarray(tss))
        """
        valid_subjects.append(s)

    # timeseries = np.asarray(timeseries)

    # Get list of keys for structural data
    subj = valid_subjects[0]
    structural_keys = list(data[f"{subj}/structural"].keys())

    structural_dir = (cwd.parent / "data/structural_data").resolve()
    for structural_key in structural_keys:
        print(structural_key)
        l = []
        for subj in valid_subjects:
            l.append(np.asarray(data[f'{subj}/structural/{structural_key}']))
        l = np.stack(l, axis=1).T
        # Save to file
        file_path = (structural_dir / f"{structural_key}.npy").resolve()
        with open(file_path, "wb") as f:
            np.savetxt(f, l)


    print('Loaded full timeseries.')