# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import fire
import torch
from tqdm import tqdm

def z_self_inf_var(losses):
    '''
    Variance-based self-influence approximation.

    Inputs
    -----------
    losses (epochs, num_examples): Tensor of losses at different epochs.
    
    Returns
    -----------
    self-influence scores: (num_examples, ) Tensor of influence scores.
    
    '''
    
    return torch.var(losses, dim=0)
    
 
def z_self_inf_mean(losses):
    '''
    Mean-based self-influence approximation.

    Inputs
    -----------
    losses (epochs, num_examples): Tensor of losses at different epochs.
    
    Returns
    -----------
    self-influence scores: (num_examples, ) Tensor of influence scores.

    '''
    return torch.mean(losses, dim=0)
 
   
def z_self_inf_finite_diff(weights,
               losses,
               batch_size=200,
               neighbors=4):
    '''
    Zeroth-order self-influence approximation using finite differences method 
    
    Inputs
    -----------
    weights (epochs, w_d): Tensor of model weights at different epochs.
    losses (epochs, num_examples): Tensor of losses at different epochs.
    batch_size(int): Batch size for processing examples.
    neighbors(int): Number of neighboring checkpoints to consider.
    
    Returns
    -----------
    self-influence scores: (num_examples, ) Tensor of self-influence scores.
  
    '''
    device = weights.device
    epochs, num_examples = losses.shape
    _, w_d = weights.shape

    approx_grads_all = torch.zeros(num_examples)
    for epoch in tqdm(range(epochs), desc='Outer Loop of Epochs'):
        batch_size_prev = 0
        batch_size_current = batch_size

        train_bar = tqdm(desc="Progress within epoch", total=num_examples)
        while batch_size_prev < num_examples:
            approx_grads = torch.zeros(min(batch_size, num_examples - batch_size_prev), w_d, device=device)

            # TODO add proper neighbor selection based on weight distances
            for epoch_inner in range(max(0, epochs - neighbors // 2), min(epochs, epochs + neighbors // 2)):
                if (epoch == epoch_inner):
                    continue
                weight_diff = weights[epoch_inner] - weights[epoch]
                output_diff = losses[epoch_inner] - losses[epoch]
                weight_diff_norm = torch.sum(weight_diff * weight_diff)
                output_diff_partial = output_diff[batch_size_prev : batch_size_current]

                approx = output_diff_partial[:, None] * (weight_diff / weight_diff_norm)
                approx_grads += approx

            approx_grads = approx_grads / neighbors
            approx_grads_sum = (approx_grads * approx_grads).sum(axis=-1).detach().cpu()
            approx_grads_all[batch_size_prev : batch_size_current] += approx_grads_sum

            batch_size_prev = batch_size_current
            batch_size_current += batch_size
            
            train_bar.update(min(batch_size, num_examples - batch_size_prev))
            train_bar.set_postfix({"Step": batch_size_prev // batch_size, 
                                   "Remaining": num_examples - batch_size_prev})

        train_bar.close()
    
    return approx_grads_all


def preprocess(weights_path: str = 'exp-results/all_weights.pt',
               losses_path: str = 'exp-results/outputs_train.pt',
               method: str = 'var', # 'var', 'mean', 'finite_diff'
               batch_size: int = 200,
               save_fl: str = 'exp-results/z_self_resnet18.pt',
               neighbors: int = 4,
               ):
    '''
    Preprocess and compute self-influence scores.

    Inputs
    -----------
    weights_path (str): Path to the file containing model weights at different epochs.
    losses_path (str): Path to the file containing losses at different epochs.
    batch_size(int): Batch size for processing examples.
    neighbors(int): Number of neighboring checkpoints to consider for finite differences.
    method(str): Method to use for self-influence approximation ('var', 'mean', 'finite_diff').
    save_fl(str): Path to save the computed self-influence scores.
    
    Returns
    -----------
    None. Saves the self-influence scores to the specified path.
  
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = torch.load(losses_path).to(device)

    if method == 'var':
        scores = z_self_inf_var(losses, save_fl)
    elif method == 'mean':
        scores = z_self_inf_mean(losses, save_fl)
    elif method == 'finite_diff':
        weights = torch.load(weights_path).to(device)
        scores = z_self_inf_finite_diff(weights, losses, batch_size, neighbors)
    else:
        raise ValueError("Invalid method. Choose from 'var', 'mean', or 'finite_diff'.")

    torch.save(scores, save_fl)

if __name__ == '__main__':
    fire.Fire(preprocess)