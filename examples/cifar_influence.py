# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import fire
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from models import ResNet

def store_weights(checkpoint_dir: str,
                  num_epochs: int = 40,
                  save_fl: str = 'exp-results/weights.pt'):
    all_last_weights = []
    for epoch in range(0, num_epochs):
        print('Epoch: ', epoch)

        weights_all = []
        model_path = f'{checkpoint_dir}/model_epoch_{epoch}.pth'
        optimizer_state_dict = torch.load(model_path, weights_only=True)['model_state_dict']
        for _, weights in optimizer_state_dict.items():
            weights_all.append(weights.view(-1))
        gradients_all = torch.cat(weights_all, dim=-1)

        all_last_weights.append(gradients_all)
    all_last_weights = torch.stack(all_last_weights)

    print('Stacked weight matrix shape: ', all_last_weights.shape)

    torch.save(all_last_weights, save_fl)


def store_losses(checkpoint_dir: str,
                 num_epochs: int = 40,
                 save_fl: str = 'losses.pt',
                 train: bool = True):
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))])
    batch_size = 128
    
    dataloader = None
    if train:
        print('Using train set')
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
    else:
        print('Using test set')
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=False, transform=transform)
        dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    models = []
    with torch.no_grad():
        for epoch in range(0, num_epochs):
            print('Epoch: ', epoch)
            model_path = f'{checkpoint_dir}/model_epoch_{epoch}.pth'
            lr = torch.load(model_path)['optimizer_state_dict']['param_groups'][0]['lr']
            net = ResNet()
            net = torch.nn.DataParallel(net.model)
            net.cuda()
            net.load_state_dict(torch.load(model_path, weights_only=True)['model_state_dict'])

            net.eval()
            models.append(net)

    all_preds = [ [] for _ in range(num_epochs) ]
    with torch.no_grad():
        for data, target in dataloader: #TODO: add tqdm
            for i, model in enumerate(models):
                data, target = data.cuda(), target.cuda()
                output = model(data)
                all_preds[i].append(loss_fn(output, target).detach().cpu())
    
    all_preds = [ torch.cat(epoch_preds) for epoch_preds in all_preds ]
    all_preds = torch.stack(all_preds)

    torch.save(all_preds, save_fl)


if __name__ == "__main__":
    fire.Fire()
