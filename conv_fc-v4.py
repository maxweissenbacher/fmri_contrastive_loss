import numpy as np
import matplotlib.pyplot as plt
import zarr
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import scipy
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import spatiotemporal
import math
import os
import random 
import pingouin
import pandas
from collections import Counter
import numpy as np
import json

### THIS ONE USES THE NEW VERSION OF DATA WHERE MEAN IS NOT REMOVED
### HERE I ALLOW FILTERS TO BE BEYOND JUST MEAN INIT

# # fix seeds 
# seed = 123
# torch.manual_seed(seed)
# random.seed(seed)
# np.random.seed(seed)
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)
# g = torch.Generator()
# g.manual_seed(seed)

# This will be the convolutional layer (with averaging as init) + fully-connected part

def icc_full(subjects, values, version="ICC1"):
    counts = Counter(subjects)
    assert len(set(counts.values())) == 1, "Different numbers of subject ratings in ICC"
    df = pandas.DataFrame({"subject": subjects, "value": values})
    df.sort_values("subject", inplace=True, kind="mergesort")  # mergesort is only stable sort
    df['rater'] = np.tile(range(0, len(subjects) // len(set(subjects))), len(set(subjects)))
    iccs = pingouin.intraclass_corr(data=df, targets="subject", raters="rater", ratings="value")
    iccs.set_index('Type', inplace=True)
    return iccs.loc[version]['ICC'], tuple(iccs.loc[version]['CI95%']), iccs.loc[version]['pval']

def contr_loss_simple(output, same, diff, eps):
    # note that when computing the gradient here, we thus need to iterate over all N^2 pairs (N is batchsize)
    dist = torch.cdist(output, output, p=2)  # gives matrix with (i,j) = l2 norm of (output[i:]-output[j:])
    loss_same = torch.mean(torch.pow(torch.masked_select(dist, same), 2))
    loss_diff = torch.mean(torch.pow(torch.clamp(eps - torch.masked_select(dist, diff), 0), 2))
    # return (loss_same + loss_diff)**2 / (dist.shape[0]*dist.shape[1])
    return loss_same + loss_diff

def contr_loss_simple_np(output, same, diff, eps):
    # note that when computing the gradient here, we thus need to iterate over all N^2 pairs (N is batchsize)
    dist = cdist(output, output)  # gives matrix with (i,j) = l2 norm of (output[i:]-output[j:])
    loss_same = np.mean(dist[same]**2)
    loss_diff = np.mean(np.maximum(eps - dist[diff], 0)**2)
    # return (loss_same + loss_diff)**2 / (dist.shape[0]*dist.shape[1])
    return loss_same + loss_diff

def load_data(file_path, nr_subj, cut_size):
    
    # Extract data
    raw_features = []
    #all_features = []
    subjnum = []
    scannum = []
    label = []

    store = zarr.ZipStore(file_path, mode='r') # hcp1200.zarr.zip
    data = zarr.group(store=store)['subjects'] # Subject timeseries, four sessions of 360 timeseries for each subject

    for i,subj in enumerate(data.keys()):
        if not data[subj]['functional/is_complete'][()]:
            continue
        if i%100==0:
            print(i)
        if i>nr_subj: break # Just x subjects so that it runs faster
        for j in range(0,4): # 4 scans roughly per subject
            # extract raw time series
            ts_np = np.asarray(data[subj]['functional'][str(j+1)]['timeseries']) # 360 (=nr_voxels) x 1200
            # cut the time series into chunks
            nr_chunks = ts_np.shape[1]//cut_size
            for chunkit in range(nr_chunks):
                ts_np_cut = ts_np[:,chunkit*cut_size:(chunkit+1)*cut_size]
                # add to raw features
                raw_features += [ts_np_cut]
                # extract variance and AR(1) coefficient (roughly 1st and 2nd term of ACF)
                #ar1s = spatiotemporal.stats.temporal_autocorrelation(ts_np)
                #var = np.var(tss[subj][scan], axis=0)
                #var_norm = var/np.max(var)
                #all_features.append(np.concatenate([ar1s, var_norm]))
                subjnum.append(i)
                scannum.append(j)
                label += [(i,j)]

    # transform to arrays
    #all_features = np.asarray(all_features) # (nr_scans * nr_subjects) x (nr_voxels * nr_features)
    subjnum = np.asarray(subjnum)
    scannum = np.asarray(scannum)
    label = np.asarray(label) # [[nr of subject, nr of scan],...]
    raw_features = np.asarray(raw_features) # (nr_scans * nr_subjects x nr_chunks) x (nr_voxels) x (length ts)
    
    # Create a matrix that will allow to query same or different pairs; 
    # element (i,j) of the matrix = True if same subject and False if different subject
    same_subject = (subjnum[:,None] == subjnum[None,:]) & (scannum[:,None] != scannum[None,:])
    # element (i,j) of the matrix = True if different subjects and False if same
    diff_subject = (subjnum[:,None] != subjnum[None,:])

    d = {'raw': raw_features,
         #'autocorrelation_and_variation': all_features,
         'subject_number': subjnum,
         'scan_number': scannum,
         'same_subject': same_subject,
         'diff_subject': diff_subject
         }

    print('Data loading complete.')

    return d

class ourDataset(Dataset):
    def __init__(self, data, device):
        self.dataset = data
        self.the_device = device

    def __len__(self):
        return (self.dataset).shape[0]

    def __getitem__(self, idx):
        # Load the data
        datapoint = self.dataset[idx]
        batch_idx = idx
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        datapoint = torch.tensor(datapoint, dtype=torch.float) #.to(self.the_device)
        return datapoint, batch_idx

class ConvFCModel(nn.Module):
    def __init__(self, batch_size, d_chans, d_init, d_conv, range_of_freqs, c_conv, d_model, n_layers, nenc):
        '''
        batch_size: batch size
        d_chans: number of channels in input data (channels here being the fmri voxels)
        d_init: initial timeseries length
        d_conv: output size i want after the convolutional layer
        c_conv: number of filers i apply in first conv layer
        range_of_freqs: list of frequencies we want to use in the conv inits
        d_model: size of the data after fc layer & every consecutive layer
        n_layers: nr of hidden layers
        nenc: output of final layer; encoding dimension
        '''
        super().__init__()

        # CONV PART: instead of the first layer being linear it can also be a convolutional layer 
        kernel_width = d_init + 1 - d_conv
        kernel_size = (1, kernel_width) # we set this to output sth of size d_conv
        self.conv1 = nn.Conv2d(1, c_conv, kernel_size) # c_conv is nr channels
        print(self.conv1.weight.shape)
        # The idea is to create c_conv filters consisting of the mean filter, the gradient filter, cos/sine filters
        filler_tensor = np.zeros([c_conv,1,1,kernel_width])
        # create the gradient filter
        the_gradient_filter = np.ones(kernel_width)
        the_gradient_filter[::2] = -1 #gives sth like array([-1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1.])
        filler_tensor[0,0,0,:] = the_gradient_filter
        # create the mean filter
        the_mean_filer = np.ones(kernel_width) * (1/kernel_width)
        filler_tensor[1,0,0,:] = the_mean_filer
        # create the cos & sine filters
        x_grid = np.linspace(0,2*np.pi,kernel_width) # do i need the full [0,2pi] range?
        it_freq = 2 
        for freq in range_of_freqs:
            filler_tensor[it_freq,0,0,:] = np.sin(x_grid*freq) # sine with some frequency sin(x[n]*freq)
            filler_tensor[it_freq+1,0,0,:] = np.cos(x_grid*freq) # cos with some frequency cos(x[n]*freq)
            it_freq += 2
        print(filler_tensor.shape)
        with torch.no_grad():
            # this will be something of shape [c_conv,1,1,kernel_width]
            self.conv1.weight.data = torch.tensor(filler_tensor, dtype = torch.float32)
        
        # TRAINABLE PART!!! it is optional: set to False to make conv1 layers not trainable
        self.conv1.bias.requires_grad = True
        self.conv1.weight.requires_grad = True

        self.flatten = nn.Flatten(start_dim=1) #flatten from second dimension only
        self.linear1 = nn.Linear(int(d_chans*d_conv*c_conv), d_model) #maps cnn dimension to d_model
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for it in range(n_layers)])
        self.final_layer = nn.Linear(int(d_model), nenc)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(d_model, affine=False) #affine=False means nonlearnable
        self.norm2 = nn.BatchNorm2d(c_conv, affine=True)
    
    def forward(self, x):
        x = self.conv1(x) # output is [batch_size, c_conv, d_chans, d_model]
        x = self.norm2(x)
        x = self.flatten(x) # output should be [batch_size, d_chans x d_conv x c_conv]
        x = self.linear1(x) # output is [batch_size, d_model]
        x = self.relu(x)
        x = self.norm1(x)
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
            x = self.norm1(x)
        # final output is [batch_size, d_model]
        # then we pass it through a linear layer to map to right embedding dimension
        x = self.final_layer(x)
        return x

def full_run(device, num_epochs, nenc, lr, eps, batch_size, d_model, n_layers, c_conv, d_conv, range_of_freqs, cut_size): 
    
    # Load data 
    data = load_data(file_path, nr_subj=nr_subj, cut_size=cut_size)
    
    # data is a dict, extract the relevant entries
    raw_features = data['raw']
    num_samps = raw_features.shape[0]
    d_chans = raw_features.shape[1]

    # OPTIONAL: transform time series component into the mean and use this data; THEN NEED TO COMMENT OUT CONV LAYER
    # raw_features = np.mean(raw_features, axis = 2).reshape(num_samps, d_chans, 1) # dim: [nr_samples, nr_voxels, 1]

    # Pass the data through the model
    d_init = raw_features.shape[2] 
    raw_features = raw_features.reshape(num_samps, 1, d_chans, d_init) # only needed when doing convolutions
    print(raw_features.shape)
    
    same_subject = data['same_subject']
    diff_subject = data['diff_subject']
    subj_num = data['subject_number']
    scan_num = data['scan_number']

    # Create train and test datasets based on a random selection of subjects
    nr_samples = raw_features.shape[0]
    nr_patients_train = int((nr_subj // 4) * 3)
    print('Number of train patients: ' + str(nr_patients_train))
    # I want to select a set of patients as train and the rest as val (point being, both should contain enough same patients)
    patients_train = random.sample(list(set(subj_num)), nr_patients_train)
    patients_val = [pat for pat in list(set(subj_num)) if pat not in patients_train]
    # print('Patients train: ' + str(patients_train))
    # print('Patients val: ' + str(patients_val))
    print('Number of total samples: '+ str(nr_samples))
    idxs_train = np.array([idx for idx in range(0,nr_samples) if subj_num[idx] in patients_train])
    idxs_val = np.array([idx for idx in range(0,nr_samples) if subj_num[idx] in patients_val])

    # # Create train and test datasets
    # perc_train = 75
    # nr_samples = raw_features.shape[0]
    # print('Number of total samples= '+ str(nr_samples))
    # num_train = int(np.round(nr_samples / 100 * perc_train))
    # # I want to select a set of patients as train and the rest as val (point being, both should contain enough same patients)
    # idxs = np.arange(nr_samples)
    # idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
    # print(idxs_train, idxs_val)

    same_subject_train = same_subject[idxs_train[:,None], idxs_train[None,:]]
    same_subject_val = same_subject[idxs_val[:,None], idxs_val[None,:]]
    diff_subject_train = diff_subject[idxs_train[:,None], idxs_train[None,:]]
    diff_subject_val = diff_subject[idxs_val[:,None], idxs_val[None,:]]
    raw_features_train = raw_features[idxs_train,:,:,:]
    raw_features_val = raw_features[idxs_val,:,:,:]
    subj_num_train = subj_num[idxs_train]
    subj_num_val = subj_num[idxs_val]
    scan_num_train = scan_num[idxs_train]
    scan_num_val = scan_num[idxs_val]

    # Load data
    dataset_train = ourDataset(raw_features_train, device=device)
    #train_dataloader = DataLoader(dataset_train, batch_size=batch_size, worker_init_fn=seed_worker,generator=g)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    dataset_val = ourDataset(raw_features_val, device=device)
    #val_dataloader = DataLoader(dataset_val, batch_size=batch_size, worker_init_fn=seed_worker,generator=g)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size)

    # Make model & optimiser
    model = ConvFCModel(batch_size, d_chans, d_init, d_conv, range_of_freqs, c_conv, d_model, n_layers, nenc).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # optimizer = optim.Adam([
    #         {'params': model.conv1.weight, 'lr':1e-4}, # i want to change these very slowly now
    #         {'params': model.conv1.bias, 'lr':1e-4},
    #         {'params': model.linear1.weight},
    #         {'params': model.linear1.bias},
    #         {'params': model.layers.parameters()},
    #         {'params': model.final_layer.weight},
    #         {'params': model.final_layer.bias},
    #         {'params': model.norm2.parameters()}
    #     ], lr=lr)

    # # Testing model output
    # output = model(torch.tensor(raw_features).to(device))
    # print(output.shape)

    # Training loop
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        # iterate over all data 
        for (d, batch_idx) in train_dataloader:
            batch_idx = batch_idx.detach().numpy()
            # get submatrices of same and diff
            same = torch.tensor(same_subject_train[batch_idx[:, None], batch_idx[None, :]]).to(device)
            diff = torch.tensor(diff_subject_train[batch_idx[:, None], batch_idx[None, :]]).to(device)
            # pass through model
            output = model.forward(d.to(device))
            # Compute the loss value
            # Currently uses the 'simple' contrastive loss!
            loss = contr_loss_simple(output, same, diff, eps)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Compute the gradients
            loss.backward()
            # Gradient clipping
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            # Take the optimisation step
            optimizer.step()

        # Update progress bar
        description = (
                        f'Loss {loss.item():.2f} | '
                        f'grad norm {gn.item():.2f} | '
                        f'learning rate {optimizer.param_groups[0]["lr"]:.9f}'
        )
        pbar.set_description(description)

        if epoch==num_epochs-1:
            # STORE the model 
            torch.save(model.state_dict(), 'models/model_'+str(epoch)+'_cut_size='+str(cut_size)\
                       +'_d_conv='+str(d_conv)+'_c_conv='+str(c_conv)\
                       + '_d_model='+str(d_model)+'_n_layers='+str(n_layers)+'_eps='+str(eps)\
                       +'.pt')
            d_chans, d_init, d_conv, c_conv, d_model, n_layers

    # print conv layer params as a check that they aren't updated 
    # for name, parameter in model.named_parameters():
    #     if name == 'conv1.weight':
    #         print(name, parameter)
    # print(model.conv1.weight)

    # Evaluation 00: mean / median plot on TRAIN
    # We want the distance for different to be very different from distance for same

    sames = []
    diffs = []
    losses = []
    subj_array = [] # for icc
    output_array = [] # for icc
    idxs_for_embeddings = []
    scan_array = []

    for it, (d,batch_idx) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        output = model.forward(d.to(device))
        dist_train = torch.cdist(output, output).detach().to('cpu').numpy()
        labels_same = same_subject_train[batch_idx[:,None],batch_idx[None,:]]
        labels_diff = diff_subject_train[batch_idx[:,None],batch_idx[None,:]]
        sames.append(dist_train[labels_same])
        diffs.append(dist_train[labels_diff])
        idxs_for_embeddings += [int(idxs_train[ix]) for ix in batch_idx]
        subj_array += [int(subj_num_train[ix]) for ix in batch_idx] # for icc
        scan_array += [int(scan_num_train[ix]) for ix in batch_idx]
        output_array += [float(element) for element in output.detach().to('cpu').numpy()[:,0]] # for icc
        losses += [contr_loss_simple_np(dist_train, labels_same, labels_diff, eps)]
    sames = np.hstack(sames)
    diffs = np.hstack(diffs)

    plt.hist(sames, bins=100, density=True, alpha=.5)
    plt.hist(diffs, bins=100, density=True, alpha=.5)
    plt.xlabel("Distance on train")
    plt.legend(["Same subject", "Different subject"])
    plt.title(f"Median of distance: {np.median(sames)-np.median(diffs):.2f}\nDifference of means: {np.mean(sames)-np.mean(diffs):.2f}")
    plt.savefig('results/train_cut_size='+str(cut_size)+'_d_conv='+str(d_conv)+'_c_conv='+str(c_conv)+'_d_model='+str(d_model)+'_n_layers='+str(n_layers)+'_eps='+str(eps)+'.png')
    plt.clf()
    median_train = np.median(sames)-np.median(diffs)
    mean_train = np.mean(sames)-np.mean(diffs)
    icc_train_val, icc_train_ci, icc_train_p = icc_full(subj_array, output_array, version="ICC1")
    print('TRAIN: Median {} Mean {} ICC {}'.format(median_train, mean_train, icc_train_val))

    # COMPUTE VALIDATION SET THINGS
                                
    # pass through model
    sames_val = []
    diffs_val = []
    losses_val = []
    subj_array_val = [] # for icc
    output_array_val = [] # for icc
    idxs_for_embeddings_val = []
    scan_array_val = []
    for it, (d,batch_idx) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        output = model.forward(d.to(device))
        dist_val = torch.cdist(output, output).detach().to('cpu').numpy()
        labels_same = same_subject_val[batch_idx[:,None],batch_idx[None,:]]
        labels_diff = diff_subject_val[batch_idx[:,None],batch_idx[None,:]]
        sames_val.append(dist_val[labels_same])
        diffs_val.append(dist_val[labels_diff])
        idxs_for_embeddings_val += [int(idxs_val[ix]) for ix in batch_idx]
        subj_array_val += [int(subj_num_val[ix]) for ix in batch_idx] # for icc
        scan_array_val += [int(scan_num_val[ix]) for ix in batch_idx]
        output_array_val += [float(element) for element in output.detach().to('cpu').numpy()[:,0]] # for icc
        losses_val += [contr_loss_simple_np(dist_val, labels_same, labels_diff, eps)]
    sames_val = np.hstack(sames_val)
    diffs_val = np.hstack(diffs_val)

    # Evaluation 0: mean / median plot; 
    # We want the distance for different to be very different from distance for same
    plt.hist(sames_val, bins=100, density=True, alpha=.5)
    plt.hist(diffs_val, bins=100, density=True, alpha=.5)
    plt.xlabel("Distance on validation data")
    plt.legend(["Same subject", "Different subject"])
    plt.title(f"Difference of medians: {np.median(sames_val)-np.median(diffs_val):.2f}\nDifference of means: {np.mean(sames_val)-np.mean(diffs_val):.2f}")
    plt.savefig('results/val_cut_size='+str(cut_size)+'_d_conv='+str(d_conv)+'_c_conv='+str(c_conv)+'_d_model='+str(d_model)+'_n_layers='+str(n_layers)+'_eps='+str(eps)+'.png')
    median_val = np.median(sames_val)-np.median(diffs_val)
    mean_val = np.mean(sames_val)-np.mean(diffs_val)
    icc_val_val, icc_val_ci, icc_val_p = icc_full(subj_array_val, output_array_val, version="ICC1")
    print('VAL: Median {} Mean {} ICC {}'.format(median_val, mean_val, icc_val_val))

    # add to result json
    json_output = {'hyperparams':{'cut_size':int(cut_size), 'd_conv':int(d_conv), 'c_conv':int(c_conv), 'd_model':int(d_model), 'n_layers':int(n_layers), 'eps':int(eps)},\
                    'embeddings_train':{'index':idxs_for_embeddings, 'subject': subj_array, 'scan':scan_array, 'embedding':output_array},\
                    'embeddings_val':{'index':idxs_for_embeddings_val, 'subject': subj_array_val, 'scan':scan_array_val, 'embedding':output_array_val},\
                    'train_patients':[int(pat) for pat in patients_train], 'val_patients':[int(pat) for pat in patients_val],\
                   'loss_train':float(np.mean(losses)), 'loss_val':float(np.mean(losses_val)), 'median_train':float(median_train), 'median_val':float(median_val), \
                    'mean_train':float(mean_train), 'mean_val':float(mean_val), 'icc_train':float(icc_train_val), 'icc_val':float(icc_val_val)
                }
    with open('results/result_'+'cut_size='+str(cut_size)+'_d_conv='+str(d_conv)+'_c_conv='+str(c_conv)+'_d_model='+str(d_model)+'_n_layers='+str(n_layers)+'_eps='+str(eps)+'.json', 'w') as fp:
        json.dump(json_output, fp)

    return 'Success!'

if __name__=="__main__":
    # Load my dataset 
    os.chdir('/Users/anastasia/Documents/O. Github/Brain connectivity analysis with Max/') # change this to your directory; to store outputs & load data
    file_path = "data/hcp1200.zarr.zip"
    #file_path = 'hcp1200.zarr.zip'
    nr_subj = 10 #len(tss.keys()) #883 in total
    
    # define some params 
    device = 'cpu' #'cuda' or 'cpu'
    num_epochs = 150
    nenc = 1
    lr = 1e-3
    eps = 10
    batch_size = 256 # to do full-batch GD use num_samps

    d_model_list = [10,100] #havent done more than 10 yet
    n_layers_list = [1,2,4]
    c_conv_list = [8] # I FIX THE NUMBER OF C_CONV'S AS I INIT EVERY ONE INDIVIDUALLY; DON'T CHANGE THIS unless you also change range_of_freqs
    d_conv_list = [2,64,128]
    # needs to be divisors of 1200: 10, 20, 60, 100, 200, 240, 300, 400, 600, 1200
    cut_size_list = [200,400,1200] 
    # this is the frequences we initialise filters to in the CNN part; the len of the list is the number of filters. 
    range_of_freqs = [1,2,3]

    for d_model in d_model_list: 
        for n_layers in n_layers_list: 
            for c_conv in c_conv_list: 
                for d_conv in d_conv_list:
                    for cut_size in cut_size_list:
                        print('Running config d_model: {} n_layers: {} c_conv: {} d_conv: {} cut_size: {}'.format(d_model, n_layers, c_conv, d_conv, cut_size))
                        output = full_run(device, num_epochs, nenc, lr, eps, batch_size, d_model, n_layers, c_conv, d_conv, range_of_freqs,  cut_size)
                        print(output)
    

    
