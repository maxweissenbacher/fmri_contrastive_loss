import secrets

import h5py
import zarr
import numpy as np
import spatiotemporal
from torch.utils.data import DataLoader, Dataset
import torch
import time
from pathlib import Path
import matplotlib.pyplot as plt

def load_data(path, format=None, number_patients=None, normalize=False, verbose=False):
    """
    path points to wherever the dataset is (this depends on where the function is called from)
    number_patients is Int, number of patients to be loaded. If None, all patients are loaded
    """
    start_time = time.time()

    if format == 'HDF5':
        print('Loading from HDF5 format.')
    elif format == 'zarr':
        print('Loading from zarr format.')
    else:
        raise NotImplementedError("Specify format to be either 'HDF5' or 'zarr'.")

    # Load dataset
    if format == 'HDF5':
        f = h5py.File(path, "r")
        tss = f['timeseries'] # Subject timeseries, four sessions of 360 timeseries for each subject
        nr_subj = number_patients if number_patients is not None else len(tss.keys())
    elif format == 'zarr':
        store = zarr.ZipStore('hcp1200.zarr.zip', mode='r')
        data = zarr.group(store=store)['subjects']
        subjects = list(data.keys())
        nr_subj = number_patients if number_patients is not None else len(subjects)

    raw_features = []
    all_features = []
    subjnum = []
    scannum = []

    # Extract data
    if format == 'HDF5':
        for i, subj in enumerate(tss.keys()):
            if i > nr_subj:
                break
            for j, scan in enumerate(tss[subj].keys()):  # 4 scans roughly per subject
                # extract raw time series
                ts_np = np.asarray(tss[subj][scan]).T  # 360 (=nr_voxels) x 1100
                raw_features += [ts_np]
                # extract variance and AR(1) coefficient (roughly 1st and 2nd term of ACF)
                ar1s = spatiotemporal.stats.temporal_autocorrelation(ts_np)
                var = np.var(tss[subj][scan], axis=0)
                var_norm = var / np.max(var)
                #all_features.append(np.concatenate([ar1s, var_norm]))  # Gives shape nr_scans x 720
                all_features.append(np.vstack((ar1s, var_norm)))  # Give shape nr_scans x 2 x 360
                subjnum.append(i)
                scannum.append(j)
    elif format == 'zarr':
        for i, s in enumerate(subjects):
            if i > nr_subj:
                break
            if not data[s]['functional/is_complete'][()]:
                err_string = f"Data for subject number {i} (id {s}) is incomplete"
                for j in range(0, 4):
                    try:
                        ts_np = np.asarray(data[s]['functional'][str(j + 1)]['timeseries'])
                        if not np.isfinite(ts_np).all():
                            err_string += " | Found NaN"
                    except KeyError:
                        err_string += f" | missing scan {j+1}"
                print(err_string)
                continue
            for j in range(0, 4):
                # extract variance and AR(1) coefficient (roughly 1st and 2nd term of ACF)
                ts_np = np.asarray(data[s]['functional'][str(j + 1)]['timeseries'])  # 360 x 1200
                ar1s = spatiotemporal.stats.temporal_autocorrelation(ts_np)
                var = np.var(ts_np.T, axis=0)
                var_norm = var / np.max(var)
                # all_features.append(np.concatenate([ar1s, var_norm]))  # Gives shape nr_scans x 720
                all_features.append(np.vstack((ar1s, var_norm)))  # Give shape nr_scans x 2 x 360
                raw_features.append(ts_np)
                subjnum.append(i)
                scannum.append(j)

    # transform to arrays
    all_features = torch.tensor(np.asarray(all_features), dtype=torch.float)  # (nr_scans) x (nr_voxels * nr_features)
    # Optional: swap the last two dimensions, so that output will have shape nr_scans x 360 x 2
    all_features = torch.transpose(all_features, 2, 1)
    subjnum = np.asarray(subjnum)
    scannum = np.asarray(scannum)
    raw_features = np.asarray(raw_features)  # (nr_scans * nr_subjects) x (nr_voxels) x (length ts)
    raw_features = raw_features.transpose((0, 2, 1))
    raw_features = torch.tensor(raw_features, dtype=torch.float)

    if normalize:
        raw_means = torch.mean(raw_features, dim=1, keepdim=True)
        raw_stds = torch.std(raw_features, dim=1, keepdim=True)
        raw_features = (raw_features - raw_means)/raw_stds

        # Normalise the norm - this is probably undesirable
        #raw_features = torch.nn.functional.normalize(raw_features, dim=1)

    # Create a matrix that will allow to query same or different pairs;
    # element (i,j) of the matrix = True if same subject and False if different subject
    same_subject = (subjnum[:, None] == subjnum[None, :]) & (scannum[:, None] != scannum[None, :])
    np.fill_diagonal(same_subject, np.ones(same_subject.shape[0], dtype=bool))  # modifies diagonal in-place
    # element (i,j) of the matrix = True if different subjects and False if same
    diff_subject = (subjnum[:, None] != subjnum[None, :])

    d = {'raw': raw_features,
         'autocorrelation_and_variation': all_features,
         'subject_number': subjnum,
         'scan_number': scannum,
         'same_subject': same_subject,
         'diff_subject': diff_subject
         }

    end_time = time.time()
    if verbose:
        print(f'Data loading complete ({nr_subj} patients, {end_time - start_time:.2f}s.).')

    return d


class ourDataset(Dataset):
    def __init__(self, data, device):
        self.dataset = data  # Assumed to be a torch.Tensor
        self.the_device = device

    def __len__(self):
        return (self.dataset).shape[0]

    def __getitem__(self, idx):
        # Load the data
        datapoint = self.dataset[idx,:]
        batch_idx = idx
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        datapoint = datapoint.to(self.the_device)
        return datapoint, batch_idx


def train_test_split(data, perc, seed=None, verbose=False):
    if seed is None:  # If user did not specify seed, choose random seed
        seed = secrets.randbits(128)
    rng = np.random.default_rng(seed)  # Ensure reproducibility of train test split
    subjects = np.array(list(set(data['subject_number'])))
    nr_subjects = len(subjects)
    nr_subjects_train = int(perc * nr_subjects)
    subjects_train = rng.choice(subjects, nr_subjects_train, replace=False)
    idxs_train = np.array([s in subjects_train for s in data['subject_number']])
    idxs_val = np.logical_not(idxs_train)

    if verbose:
        print(f"Total number of scans = {data['raw'].shape[0]}, num of scans in training set = {idxs_train.sum()}, num of scans in testing set = {idxs_val.sum()}.")

    d_train = {}
    d_train['raw'] = data['raw'][idxs_train, :]
    d_train['autocorrelation_and_variation'] = data['autocorrelation_and_variation'][idxs_train, :]
    d_train['same_subject'] = data['same_subject'][idxs_train, :][:, idxs_train]
    d_train['diff_subject'] = data['diff_subject'][idxs_train, :][:, idxs_train]
    d_val = {}
    d_val['raw'] = data['raw'][idxs_val, :]
    d_val['autocorrelation_and_variation'] = data['autocorrelation_and_variation'][idxs_val, :]
    d_val['same_subject'] = data['same_subject'][idxs_val, :][:, idxs_val]
    d_val['diff_subject'] = data['diff_subject'][idxs_val, :][:, idxs_val]
    d = {'train': d_train, 'val': d_val}

    return d


if __name__ == '__main__':
    # Some basic data exploration
    # Load data
    cwd = Path.cwd()  # Current working directory
    rel_path = 'data/timeseries_max_all_subjects.hdf5'  # Relative path from project directory, depends on where you store the data
    file_path = (cwd.parent / rel_path).resolve()
    data = load_data(file_path, format='zarr', number_patients=100, verbose=True)

    print('here')

    print(data['raw'].shape)

    x = data['raw']
    xx = torch.nn.functional.normalize(torch.tensor(data['raw']), dim=1)

    # For each patient, plot histogram of L2 norm of each region
    # i.e. for each patient, and each region, compute the L2 norm of the time series, then plot a histogram over the different regions
    # The distributions are all fairly similar
    num_patients = 20
    for patient_id in range(num_patients):
        if len(np.where(data['subject_number'] == patient_id)[0]) == 0:
            continue
        idx = np.where(data['subject_number'] == patient_id)[0][0]
        plt.hist(torch.linalg.norm(torch.tensor(x), dim=1)[idx, :], histtype='step', bins=15)
        #plt.title(f"Patient id {data['subject_number'][i]}")
    plt.legend([f"Patient id {i}" for i in range(num_patients)])
    plt.show()

