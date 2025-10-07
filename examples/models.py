# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision


from torch import nn

class ResNet:
    def __init__(self, num_classes=10):
        self.model = torchvision.models.resnet18(weights=None)
        self.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = torch.nn.Identity()  # remove the 3x3/2 maxpool
        # adapt final FC
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input):
        return self.forward(input)

