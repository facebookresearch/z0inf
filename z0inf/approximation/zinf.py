# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import fire
import torch
import time
import numpy as np
from z0inf.common.utils import get_gramm_weights, get_weights_norms, spearmanr_batch
from tqdm import tqdm

def zinf_train_test_inner(gramm_W: torch.Tensor,
                          norms_W: torch.Tensor,
                          outputs_train: torch.Tensor,
                          outputs_test: torch.Tensor,
                          batch_size: int=3000,
                          epochs: int=30,
                          neighbors: int=4,
                          save_path: str='exp-results/',
                          save_intermediate: bool=True,
                          center_losses: bool=False, #TODO try wuth false
                          eps: float = 1e-12):
    '''
    Zeroth-order train-test influence approximation using finite differences method

    Inputs
    -----------
    gramm_W (epochs, epochs): Tensor of Gramm matrix of model weights at different epochs.
    norms_W (epochs, epochs): Tensor of norms matrix of model weights at different epochs.
    outputs_train_fl: Path to the Tensor of model outputs on training data at different epochs.
    outputs_test_fl: Path to the Tensor of model outputs on test data at different epochs.
    train_batch_size(int): Batch size for processing training examples.
    test_batch_size(int): Batch size for processing test examples.
    save_path: Path to save the results.
    save_intermediate(bool): Whether to save intermediate results after each epoch.
    save_influence_fl: Path to save the final influence scores.
    center_losses(bool): Whether to center the losses by subtracting the mean loss.
    eps(float): Small value to avoid division by zero.

    Returns
    -----------
    influence scores: (num_examples_train, num_examples_test) Tensor of influence scores.
    '''
    
    device = gramm_W.device

    if center_losses:
        outputs_train = outputs_train - outputs_train.mean(dim=1, keepdim=True)
        outputs_test = outputs_test - outputs_test.mean(dim=1, keepdim=True)
    
    _, num_examples_train = outputs_train.shape
    _, num_examples_test = outputs_test.shape

    approx_grads_all = torch.zeros(num_examples_train, num_examples_test).to(device)
    for epoch in tqdm(range(epochs), desc='Outer Loop of Epochs'):
        print(f'Processing epoch {epoch + 1} of {epochs}')
        batch_size_prev_train = 0
        batch_size_current_train = batch_size
        tight_neighbors = range(max(0, epochs - neighbors // 2), min(epochs, epochs + neighbors // 2))  #  #select_tight_neighbors(epoch, weights, lower_q=lower_q, upper_q=upper_q) # np.arange(epochs) # lower_q=0.62, upper_q=0.85 # np.arange(max(0, epoch - 3), max(epoch + 3, epochs)) 
        approx_grads_epoch = torch.zeros(num_examples_train, num_examples_test).to(device)
        
        train_bar = tqdm(desc="Progress within epoch", total=num_examples_train)
        while batch_size_prev_train < num_examples_train:
            batch_size_prev_test = 0
            batch_size_current_test = batch_size
            while batch_size_prev_test < num_examples_test:
                for epoch_inner in tight_neighbors:
                    if epoch_inner == epoch:
                        continue
                    output_diff_train = outputs_train[epoch] - outputs_train[epoch_inner]
                    output_diff_partial_train = output_diff_train[batch_size_prev_train : batch_size_current_train]

                    for epoch_inner_inner in tight_neighbors:
                        if epoch_inner_inner == epoch:
                            continue
                        output_diff_test = outputs_test[epoch] - outputs_test[epoch_inner_inner]
                        output_diff_partial_test = output_diff_test[batch_size_prev_test : batch_size_current_test]
                        alpha = (1.0 / (norms_W[epoch][epoch_inner] + eps)) * ((1.0 / (norms_W[epoch][epoch_inner_inner] + eps)))
                        sim = alpha * torch.outer(output_diff_partial_train, output_diff_partial_test) * \
                                           (gramm_W[epoch][epoch] - \
                                            gramm_W[epoch][epoch_inner] - \
                                            gramm_W[epoch][epoch_inner_inner] + \
                                            gramm_W[epoch_inner][epoch_inner_inner]) / (norms_W[epoch][epoch_inner] * norms_W[epoch][epoch_inner_inner])
                        approx_grads_all[batch_size_prev_train : batch_size_current_train,
                                 batch_size_prev_test : batch_size_current_test] += sim
                        approx_grads_epoch[batch_size_prev_train : batch_size_current_train,
                                    batch_size_prev_test : batch_size_current_test] += sim

                batch_size_prev_test = batch_size_current_test
                batch_size_current_test += batch_size

            batch_size_prev_train = batch_size_current_train
            batch_size_current_train += batch_size

            train_bar.update(batch_size)
            train_bar.set_postfix({"Step": batch_size_prev_train // batch_size, 
                                   "Remaining": num_examples_train - batch_size_prev_train})
        
        train_bar.close()     
        #save the epoch
        if save_intermediate:
            torch.save(approx_grads_epoch, f'{save_path}/epoch-grads/{epoch}.pt')

    approx_grads_all *= 1 / epochs
    return approx_grads_all


def zinf_train_test(train_path: str,
                    test_path: str,
                    save_path: str,
                    weights_path: str = 'all_weights.pt',
                    method: str = 'finite_diff', # 'finite_diff' or 'corr'
                    bacth_size: int = 512,
                    epochs: int = 30,
                    neighbors: int = 4,
                    do_precompute: bool = True,
                    save_intermediate: bool = True,
                    center_losses: bool = True,
                    weight_by_correlation=True,
                    save_fl='approx_grads_train_test.pt',
                    verbose: bool = True):
    outputs_train = torch.load(train_path).cuda() 
    outputs_test = torch.load(test_path).cuda()
    if verbose:
        print('neighbors: ', neighbors)
    
    if method == 'corr':
        torch.save(spearmanr_batch(outputs_train, outputs_test), f'{save_path}/{save_fl}')
        return

    if do_precompute:
        weights = torch.load(weights_path).cuda()
        gramm_W = get_gramm_weights(weights)
        norms_W = get_weights_norms(weights)

        torch.save(gramm_W, f'{save_path}/gramm_weights.pt')
        torch.save(norms_W, f'{save_path}/norms_weights.pt')

        print('Precomputed gramm and norms weights and saved to disk.')
    else:
        gramm_W = torch.load(f'{save_path}/gramm_weights.pt')
        norms_W = torch.load(f'{save_path}/norms_weights.pt')

    outputs_train = torch.load(train_path).cuda() 
    outputs_test = torch.load(test_path).cuda()
    
    approx_grads_all = zinf_train_test_inner(gramm_W,
                                             norms_W,
                                             outputs_train,
                                             outputs_test,
                                             batch_size=bacth_size,
                                             epochs=epochs,
                                             neighbors=neighbors,
                                             save_path=save_path,
                                             save_intermediate=save_intermediate,
                                             center_losses=center_losses)
    
    if weight_by_correlation:
        print('Weighting by correlation')
        approx_grads_all *= spearmanr_batch(outputs_train, outputs_test)
        
    torch.save(approx_grads_all, f'{save_path}/{save_fl}')


# zinf.py WEIGHTS_PATH TRAIN_PATH TEST_PATH SAVE_PATH <flags>
# python -m approximation.zinf --weights_path examples/exp-results/all_weights.pt \
#                              --train_path  examples/exp-results/losses_train.pt \
#                              --test_path  examples/exp-results/losses_test.pt \
#                              --save_path examples/exp-results/
if __name__ == '__main__':
    fire.Fire(zinf_train_test)