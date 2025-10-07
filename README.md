# Z0: Zeroth-Order Approximation for Data Influence

This project provides methods for zeroth-order approximation for data influence and demonstrates how to use them on various models.


## Installation
Installation Requirements

* Python >= 3.6
* PyTorch >= 2.0.0

Create a new conda environment and activate it.
```
conda create --name z0inf python=3.10.14
conda activate z0inf

conda install pytorch==2.4.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install fire matplotlib "tqdm>=4.42.1" -c conda-forge
conda install numpy scipy
```
Skip torchvision and torchaudio, if vision and audio is not required by the model.

Install the package locally.

```
pip install -e .
```


## Zeroth-Order Quantities
Before running zeroth-order approximation, let's precompute and store the quantities s.a. the weight matrix, train and test losses. 
These quantaties will be reused during zeroth-order estimation estimation.
The tutorial below demonstrates how to do that for the toy ResNet18 model and CIFAR-10 dataset.


## ZSInf - Zeroth-Order Self-Influence

```
python -m approximation.zsinf --method <METHOD> \
                              --losses_path <TRAIN_DATA_LOSS_PATH> \ 
                              --save_fl <SELF_INFLUENCE_SCORES_FILE_PATH>
                              --batch_size <TRAINING_DATA_BATCH_SIZE>
                              --neighbors <NUMBER_OF_NEARST_NEIGHBOR_CHECKPOINTS>
```

## ZInf - Zeroth-Order Train-Test Influence

```
python -m approximation.zinf --weights_path <WEIGHTS_PATH>  \
                             --train_path <TRAIN_DATA_LOSS_PATH> \
                             --test_path <TEST_DATA_LOSS_PATH> \
                             --save_fl <INFLUENCE_SCORES_FILE_PATH> \
                             --do_precompute <BOOL_FLAG_TO_ENFORCE_PRECOMPUTATION_OF_WEIGHT_AND_LOSS_QUANTITIES> \
                             --save_intermediate <BOOL_FLAG_TO_SAVE_THE_APPROXIMATIONS_PER_EPOCH> \ 
                             --center_losses <BOOL_FLAG_TO_CENTER_LOSS_SCORES_BEFORE_APPROXIMATION> \
                             --weight_by_correlation <BOOL_FLAG_TO_WEIGHT_APPROXIMATION_BY_LOSS_CORRELATION>
```

# A demo using Resnet-18 model on CIFAR-10 dataset.

## A sample script to train the model from scratch


```
cd examples
python cifar_train_resnet.py --checkpoint_dir data/checkpoints

```

The checkpoints will be stored under: `examples/data/checkpoints` 

## A tutorial on how to run Zeroth-Order approximation on CIFAR-10 and the model that we trained with the script above.


### Store the weights
```
python cifar_influence.py store_weights --checkpoint_dir data/checkpoints \
                                        --num_epochs 30 \
                                        --save_fl data/exp-results/all_weights.pt

```

### Store the losses both train and test
```
python cifar_influence.py store_losses --checkpoint_dir data/checkpoints \
                                        --num_epochs 30 \
                                        --save_fl data/exp-results/train_losses.pt \
                                        --train True
```

```
python cifar_influence.py store_losses --checkpoint_dir data/checkpoints \
                                        --num_epochs 30 \
                                        --save_fl data/exp-results/test_losses.pt \
                                        --train False
```

The demo below provides examples on how to compute influence scores.
It visualizes the influence scores and the data points associated with it

```

examples/z0_Inf_cifar_demo.ipynb

```

## Citation


## License
Z0Inf is licensed under the CC-BY-NC 4.0 license.
