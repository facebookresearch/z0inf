# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import fire
import torch
import torchvision
import torchvision.transforms as transforms
from models import ResNet as Net


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))])
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


print('Train Set length: ', len(trainset))
print('Test Set length: ', len(testset))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_epochs = 40


net = Net()
net.model = net.model.to(device)
net = torch.nn.DataParallel(net.model)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(),
                            lr=0.1,
                            momentum=0.9,
                            weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


import math

parameters = net.parameters()
num_parameters = 0
for param in parameters:
    num_parameters += math.prod(param.shape)

print('Total number of model parameters: ', num_parameters)


def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Epoch: ', epoch, batch_idx, '/', len(trainloader), 'Train Avg Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                  % (train_loss/(batch_idx+1), 100. * correct / total, correct, total))


def test(epoch,
         net,
         checkpoint_dir):    
    test_loss = 0
    correct = 0
    total = 0

    net.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)            
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            
            correct += predicted.eq(targets).sum().item()

        print('Epoch: ', epoch, batch_idx, '/',  len(trainloader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
               % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    
    checkpoint_path = f"{checkpoint_dir}/model_epoch_{epoch}.pth"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)


def train_and_eval(checkpoint_dir: str = 'data/checkpoints'):
    for epoch in range(start_epoch, start_epoch + num_epochs):
        train(epoch)
        test(epoch, net, checkpoint_dir)

if __name__ == '__main__':
    fire.Fire(train_and_eval)