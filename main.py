import numpy as np
import matplotlib.pyplot as plt
#import h5py # https://pypi.org/project/h5py/
import spatiotemporal # https://pypi.org/project/spatiotemporal/
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import zarr




store = zarr.ZipStore('/home/max/Research_data/other/hcp_mni/hcp1200.zarr.zip', mode='r')
data = zarr.group(store=store)['subjects']
subjects = list(sorted(data.keys()))
valid_subjects = []
timeseries = []
for s in subjects:
    if not data[s]['functional/is_complete'][()]:
        continue
    tss = [data[s]['functional'][str(i+1)]['timeseries']
                 for i in range(0,4)]
    timeseries.append(np.asarray(tss))
    valid_subjects.append(s)

timeseries = np.asarray(timeseries)
ts2 = timeseries.reshape(-1,timeseries.shape[2], timeseries.shape[3])
assert np.all(ts2[1] == timeseries[0,1])
assert np.all(ts2[4] == timeseries[1,0])

feature_bank = []
for i,subj in enumerate(subjects):
    print(i)
    for scan in range(0, 4):
        ts_np = timeseries[i][scan]
        mean = np.mean(ts_np, axis=1)
        var = np.var(ts_np, axis=1)
        std = np.std(ts_np, axis=1)
        skew = scipy.stats.skew(ts_np, axis=1)
        kurt = scipy.stats.kurtosis(ts_np, axis=1)
        ars = [np.asarray([np.corrcoef(ts_np[i,k:], ts_np[i,:-k])[0,1] for i in range(0, len(ts_np))]) for k in range(1, 10)]
        features = np.concatenate([[mean, var, std, skew, kurt], ars])
        feature_bank.append(features)
        
np.save("feature_bank.npy", feature_bank)

scannum = np.tile([0, 1, 2, 3], timeseries.shape[0])
subjnum = np.repeat(range(0, timeseries.shape[0]), 4)
label = list(zip(subjnum,scannum))


# raw_features = []
# all_features = []
# subjnum = []
# scannum = []
# label = []
# nr_subj = len(tss.keys())
# for i,subj in enumerate(tss.keys()):
#     if i>nr_subj: break # Just x subjects so that it runs faster
#     for j,scan in enumerate(tss[subj].keys()):
#         ts_np = np.asarray(tss[subj][scan]).T # 360 x 1100
#         raw_features += [ts_np]
#         # feature extraction
#         ar1s = spatiotemporal.stats.temporal_autocorrelation(ts_np)
#         var = np.var(tss[subj][scan], axis=0)
#         var_norm = var/np.max(var)
#         all_features.append(np.concatenate([ar1s, var_norm]))
#         subjnum.append(i)
#         scannum.append(j)
#         label += [(i,j)]
        
feature_bank = np.asarray(feature_bank) # (nr_scans * nr_subjects) x (360 * nr_features)
subjnum = np.asarray(subjnum)
scannum = np.asarray(scannum)
label = np.asarray(label) # [[nr of subject, nr of scan],...]
raw_features = np.asarray(raw_features)
print(raw_features.shape)

def contr_loss(batch1, label1, batch2, label2, eps): 
    subjnum1 = label1[:,0]
    scannum1 = label1[:,1]
    subjnum2 = label2[:,0]
    scannum2 = label2[:,1]
    same_subjects = (subjnum1[:,None] == subjnum2[None,:]) & (scannum1[:,None] != scannum2[None,:])
    diff_subjects = (subjnum1[:,None] != subjnum2[None,:])
    dist = torch.cdist(batch1, batch2, p=2) # gives matrix with (i,j) = l2 norm of (batch1_[i:]-batch2_[j:])
    loss_same = torch.sum(torch.square(dist[same_subjects]))
    loss_diff = torch.sum(torch.square(torch.clamp(eps - dist[diff_subjects],0)))
    return loss_same + loss_diff

class Net(nn.Module):
  def __init__(self, dim, nenc, width, depth):
    super().__init__()
    self.flatten = nn.Flatten()
    modules = []
    modules.append(nn.Linear(dim, width))
    modules.append(nn.ReLU())
    for dp in range(depth):
        modules.append(nn.Linear(width, width))
        modules.append(nn.ReLU())
    modules.append(nn.Linear(width, nenc)) # output is thus something of size nenc, the size of the encoding
    self.sequential = nn.Sequential(*modules)
  
  def forward(self, input):
    x = self.flatten(input) # equivalently, x = x.view(x.size()[0], -1)
    output = self.sequential.forward(x)
    return output

def run_network(all_features, name, bs=16, width=64, depth=2, nenc=1, eps=100, nr_its=1000):
    dim = all_features.shape[1]
    net = Net(dim, nenc, width, depth)
    optimizer = optim.SGD(net.parameters(), lr=0.00000001)
    print('Number of total samples= '+ str(all_features.shape[0]))
     
    nr_train = round(all_features.shape[0]//4*.75)*4 # 4 preserves subject boundaries
    input_t = torch.tensor(all_features[:nr_train,:], dtype=torch.float) # (nr_scans * nr_subjects) x (360 * nr_features)
    label_t = torch.tensor(label[:nr_train,:]) # [[nr of subject, nr of scan],...]
    input_test = torch.tensor(all_features[nr_train:,:], dtype=torch.float)
    label_test = torch.tensor(label[nr_train:,:])
    
    print("Starting", name)
    for epoch in range(nr_its):
        # X is a torch Variable
        permutation = torch.randperm(input_t.shape[0])
        for i in range(0,input_t.shape[0], bs):
            indices = permutation[i:i+bs]
            input_in = input_t[indices,:]
            label_in = label_t[indices,:]
    
            # Compute the output for all the inputs in the batch_size; dim: bs x nenc
            outputs = net.forward(input_in)
            outputs_all = net.forward(input_t)
    
            # Compute the loss value
            loss = contr_loss(outputs, label_in, outputs_all, label_t, eps)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Compute the gradients
            loss.backward()
            # Take the optimisation step
            optimizer.step()
    
        if epoch%10==0:
            print(epoch, loss.item())
    
    print('Finished training', name)
    print('Final loss ' + str(loss.item()))
    outputs_all = net.forward(torch.tensor(all_features, dtype=torch.float)).detach().numpy()
    np.savez_compressed(f"encoding_layer_{name}.npz", network_output=outputs_all, subjnum=subjnum, scannum=scannum)
    
    # compute model on all data 
    outputs_train = net.forward(input_t)
    outputs_test = net.forward(input_test)
    
    # select data 
    subjnum1 = label_t[:,0]
    scannum1 = label_t[:,1]
    subjnum2 = label_test[:,0]
    scannum2 = label_test[:,1]
    same_subject_train = (subjnum1[:,None] == subjnum1[None,:]) & (scannum1[:,None] != scannum1[None,:])
    diff_subject_train = (subjnum1[:,None] != subjnum1[None,:])
    same_subject_test = (subjnum2[:,None] == subjnum2[None,:]) & (scannum2[:,None] != scannum2[None,:])
    diff_subject_test = (subjnum2[:,None] != subjnum2[None,:])
    
    # plot histogram on train
    dist_train = torch.cdist(outputs_train, outputs_train, p=2).detach().numpy()
    sames = dist_train[same_subject_train]
    diffs = dist_train[diff_subject_train]
    plt.hist(sames, bins=np.linspace(0, 20000, 100), density=True, alpha=.5)
    plt.hist(diffs, bins=np.linspace(0, 20000, 100), density=True, alpha=.5)
    plt.xlabel("Similarity on train")
    plt.legend(["Same subject", "Different subject"])
    plt.title(f"Difference of medians: {np.median(sames)-np.median(diffs):.2f}\nDifference of means: {np.mean(sames)-np.mean(diffs):.2f}")
    plt.savefig(f"trainset-{name}.png")
    plt.show()
    
    # histogram on test
    dist_test = torch.cdist(outputs_test, outputs_test, p=2).detach().numpy()
    sames = dist_test[same_subject_test]
    diffs = dist_test[diff_subject_test]
    plt.hist(sames, bins=np.linspace(0, 20000, 100), density=True, alpha=.5)
    plt.hist(diffs, bins=np.linspace(0, 20000, 100), density=True, alpha=.5)
    plt.xlabel("Similarity on test")
    plt.legend(["Same subject", "Different subject"])
    plt.title(f"Difference of medians: {np.median(sames)-np.median(diffs):.2f}\nDifference of means: {np.mean(sames)-np.mean(diffs):.2f}")
    plt.savefig(f"testset-{name}.png")
    plt.show()
    
