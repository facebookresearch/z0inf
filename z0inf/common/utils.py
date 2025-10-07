# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from scipy import stats
import numpy as np

def get_gramm_weights(weights: torch.Tensor):
    '''
    Compute Gram matrix G = weights @ weights.T for a small number of epochs and moderate model size.
    
    Inputs:
        weights: (epochs, num_parameters) tensor on GPU
    '''
    return weights @ weights.T


def get_gramm_weights_large(X, row_block=1, col_block=500000):
    """
    Compute Gram matrix G = X @ X.T blocking by rows and columns to fit in CPU memory.
    X: (n, d) tensor on CPU
    row_block: number of rows to load per block
    col_block: number of features to load per chunk
    
    """
    n, d = X.shape
    
    device = X.device
    
    G = torch.zeros((n, n), dtype=X.dtype)

    for i in range(0, n, row_block):
        Xi = torch.zeros((min(row_block, n - i), d), dtype=X.dtype, device=device)
        Xi.copy_(X[i:i+row_block])  # copy row block
        for j in range(i, n, row_block):  # use symmetry: only compute j ≥ i
            Xj = torch.zeros((min(row_block, n - j), d), dtype=X.dtype, device=device)
            Xj.copy_(X[j:j+row_block])

            # accumulate across feature chunks
            Gij = torch.zeros((Xi.shape[0], Xj.shape[0]), dtype=X.dtype, device=device)
            for k in range(0, d, col_block):
                Xi_chunk = Xi[:, k:k+col_block].cuda()
                Xj_chunk = Xj[:, k:k+col_block].cuda()
                Gij += Xi_chunk @ Xj_chunk.T

            Gij = Gij.cpu()
            G[i:i+Xi.shape[0], j:j+Xj.shape[0]] = Gij
            if i != j:
                G[j:j+Xj.shape[0], i:i+Xi.shape[0]] = Gij.T  # fill symmetric part

    return G

def get_weights_norms(weights: torch.Tensor):
    epochs = weights.shape[0]
    W_norms = torch.zeros((epochs, epochs),
                          dtype=weights.dtype,
                          device=weights.device)

    for epoch in range(epochs):
        for epoch_inner in range(epoch):
            if epoch_inner == epoch:
                continue
            W_diff = weights[epoch] - weights[epoch_inner]
            W_norm = torch.sum(W_diff * W_diff)
            W_norms[epoch][epoch_inner] = W_norm
    
    # copy symmetric part
    W_norms = W_norms + W_norms.T
    
    return W_norms


def spearmanr_batch(t1: torch.Tensor,
                    t2: torch.Tensor):
    """
    Compute Spearman correlation between columns of t1 and t2.
    
    t1: [epochs, batch_size1]
    t2: [epochs, batch_size2]
    
    Returns: [batch_size1, batch_size2] Spearman correlation matrix
    """
    # Convert to numpy for ranking
    t1_np = t1.detach().cpu().numpy()
    t2_np = t2.detach().cpu().numpy()
    
    # Rank along rows (axis=0 are samples)
    t1_ranks = np.apply_along_axis(stats.rankdata, 0, t1_np)
    t2_ranks = np.apply_along_axis(stats.rankdata, 0, t2_np)
    
    # Convert back to torch
    t1_ranks = torch.tensor(t1_ranks, dtype=t1.dtype, device=t1.device)
    t2_ranks = torch.tensor(t2_ranks, dtype=t2.dtype, device=t2.device)
    
    # Center
    t1_centered = t1_ranks - t1_ranks.mean(dim=0, keepdim=True)
    t2_centered = t2_ranks - t2_ranks.mean(dim=0, keepdim=True)
    
    # Pearson correlation on ranks
    cov = (t1_centered.T @ t2_centered) / (t1.shape[0] - 1)
    std_t1 = t1_centered.std(dim=0, unbiased=True).unsqueeze(1)
    std_t2 = t2_centered.std(dim=0, unbiased=True).unsqueeze(0)
    
    corr = cov / (std_t1 @ std_t2)
    return corr
