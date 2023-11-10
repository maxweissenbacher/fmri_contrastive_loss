import torch
import torch.nn as nn


def contr_loss_simple(output, same, diff, eps, alpha=1., metric=None):
    # note that when computing the gradient here, we thus need to iterate over all N^2 pairs (N is batchsize)
    if metric == 'euclidean':
        # Euclidean distance between embeddings
        dist = torch.cdist(output, output, p=2)  # gives matrix with (i,j) = l2 norm of (output[i:]-output[j:])
    elif metric == 'cosine':
        # Cosine similarity between embeddings
        dist = nn.CosineSimilarity(dim=-1)(output[..., None, :, :], output[..., :, None, :])
    else:
        raise NotImplementedError("A metric must be chosen for the loss: either 'euclidean' or 'cosine'.")
    loss_same = torch.mean(torch.pow(torch.masked_select(dist, same), 2))
    loss_diff = torch.mean(torch.pow(torch.clamp(eps - torch.masked_select(dist, diff), 0), 2))
    # return (loss_same + loss_diff)**2 / (dist.shape[0]*dist.shape[1])
    return loss_same + alpha * loss_diff


def contr_loss_lifted(output, same, diff, eps):
    dist = torch.cdist(output, output, p=2)  # gives matrix with (i,j) = l2 norm of (output[i:]-output[j:])
    pos_indices = torch.argwhere(same)  # size (nr_pos, 2)

    # for each pair in the positive pairs, compute negative distances
    pos_pair1 = pos_indices[:, 0]  # get first positive pair index
    pos_pair2 = pos_indices[:, 1]  # get second positive pair index
    # find the largest distance between first/second positive index and its negative partner
    # note: dist[pos_pair1,:] is a matrix of size (nr_pos_pairs, nr_pairs) & diff[pos_pair1,:] is matrix w/ True/False
    dist_pospair1 = dist[pos_pair1, :]
    dist_pospair1[diff[pos_pair1, :] == False] = np.inf
    max1 = torch.exp(eps - dist_pospair1)
    dist_pospair2 = dist[pos_pair2, :]
    dist_pospair2[diff[pos_pair2, :] == False] = np.inf
    max2 = torch.exp(eps - dist_pospair2)
    max_12 = torch.log(
        torch.sum(max1, axis=1) + torch.sum(max2, axis=1))  # sum over all pairs -> output is [nr_pos_pairs1]
    sum_all = torch.max(torch.tensor(0), max_12 + torch.masked_select(dist, same))  # should be size
    return torch.sum(sum_all) / (2 * pos_indices.shape[0])