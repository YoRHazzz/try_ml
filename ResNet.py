import os
import argparse

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from standard_image_classification import download_FashionMNIST_dataloader, standard_image_classification
from utils import Config

NET_NAME = os.path.splitext(os.path.basename(__file__))[0]


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

    @property
    def depth(self):
        return 2


def main():
    parser = argparse.ArgumentParser(description='ResNet Example')
    parser.add_argument('--config', type=str, default="default.yaml", metavar='CONFIG',
                        help='config file path')
    parser.add_argument('--test', action='store_true', default=False,
                        help='For testing the saved Model')
    args = parser.parse_args()
    config = Config(args.config).set_sub_config('ResNet')

    devices = [index for index in config['gpu'] if index < torch.cuda.device_count()]
    print(f"use gpu: {devices}")
    train_dataloader, test_dataloader = \
        download_FashionMNIST_dataloader(batch_size=config['batch_size'] * len(devices),
                                         test_batch_size=config['test_batch_size'] * len(devices),
                                         resize=config['resize'],
                                         num_workers=config['num_workers'])

    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def resnet_block(input_channels, num_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    model = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10))

    kwargs = dict(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        dataset_name='FashionMNIST',
        net_name=NET_NAME,
        devices=devices,
        learning_rate=config['lr'],
        num_epochs=config['num_epochs'],
        eval_interval=config['eval_interval'],
        random_seed=config['random_seed'],
        image_size=(config['num_channels'], config['resize'], config['resize']),
        train=not args.test
    )
    standard_image_classification(**kwargs)


if __name__ == "__main__":
    main()
"""
use gpu: [0, 1]
input train shape torch.Size([256, 1, 224, 224]) torch.Size([256])
input test shape torch.Size([2560, 1, 224, 224]) torch.Size([2560])
ResNet-18 Depth=18 Linear=1 Conv=20
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [256, 10]                 --
├─Sequential: 1-1                        [256, 64, 56, 56]         --
│    └─Conv2d: 2-1                       [256, 64, 112, 112]       3,200
│    └─BatchNorm2d: 2-2                  [256, 64, 112, 112]       128
│    └─ReLU: 2-3                         [256, 64, 112, 112]       --
│    └─MaxPool2d: 2-4                    [256, 64, 56, 56]         --
├─Sequential: 1-2                        [256, 64, 56, 56]         --
│    └─Residual: 2-5                     [256, 64, 56, 56]         --
│    │    └─Conv2d: 3-1                  [256, 64, 56, 56]         36,928
│    │    └─BatchNorm2d: 3-2             [256, 64, 56, 56]         128
│    │    └─Conv2d: 3-3                  [256, 64, 56, 56]         36,928
│    │    └─BatchNorm2d: 3-4             [256, 64, 56, 56]         128
│    └─Residual: 2-6                     [256, 64, 56, 56]         --
│    │    └─Conv2d: 3-5                  [256, 64, 56, 56]         36,928
│    │    └─BatchNorm2d: 3-6             [256, 64, 56, 56]         128
│    │    └─Conv2d: 3-7                  [256, 64, 56, 56]         36,928
│    │    └─BatchNorm2d: 3-8             [256, 64, 56, 56]         128
├─Sequential: 1-3                        [256, 128, 28, 28]        --
│    └─Residual: 2-7                     [256, 128, 28, 28]        --
│    │    └─Conv2d: 3-9                  [256, 128, 28, 28]        73,856
│    │    └─BatchNorm2d: 3-10            [256, 128, 28, 28]        256
│    │    └─Conv2d: 3-11                 [256, 128, 28, 28]        147,584
│    │    └─BatchNorm2d: 3-12            [256, 128, 28, 28]        256
│    │    └─Conv2d: 3-13                 [256, 128, 28, 28]        8,320
│    └─Residual: 2-8                     [256, 128, 28, 28]        --
│    │    └─Conv2d: 3-14                 [256, 128, 28, 28]        147,584
│    │    └─BatchNorm2d: 3-15            [256, 128, 28, 28]        256
│    │    └─Conv2d: 3-16                 [256, 128, 28, 28]        147,584
│    │    └─BatchNorm2d: 3-17            [256, 128, 28, 28]        256
├─Sequential: 1-4                        [256, 256, 14, 14]        --
│    └─Residual: 2-9                     [256, 256, 14, 14]        --
│    │    └─Conv2d: 3-18                 [256, 256, 14, 14]        295,168
│    │    └─BatchNorm2d: 3-19            [256, 256, 14, 14]        512
│    │    └─Conv2d: 3-20                 [256, 256, 14, 14]        590,080
│    │    └─BatchNorm2d: 3-21            [256, 256, 14, 14]        512
│    │    └─Conv2d: 3-22                 [256, 256, 14, 14]        33,024
│    └─Residual: 2-10                    [256, 256, 14, 14]        --
│    │    └─Conv2d: 3-23                 [256, 256, 14, 14]        590,080
│    │    └─BatchNorm2d: 3-24            [256, 256, 14, 14]        512
│    │    └─Conv2d: 3-25                 [256, 256, 14, 14]        590,080
│    │    └─BatchNorm2d: 3-26            [256, 256, 14, 14]        512
├─Sequential: 1-5                        [256, 512, 7, 7]          --
│    └─Residual: 2-11                    [256, 512, 7, 7]          --
│    │    └─Conv2d: 3-27                 [256, 512, 7, 7]          1,180,160
│    │    └─BatchNorm2d: 3-28            [256, 512, 7, 7]          1,024
│    │    └─Conv2d: 3-29                 [256, 512, 7, 7]          2,359,808
│    │    └─BatchNorm2d: 3-30            [256, 512, 7, 7]          1,024
│    │    └─Conv2d: 3-31                 [256, 512, 7, 7]          131,584
│    └─Residual: 2-12                    [256, 512, 7, 7]          --
│    │    └─Conv2d: 3-32                 [256, 512, 7, 7]          2,359,808
│    │    └─BatchNorm2d: 3-33            [256, 512, 7, 7]          1,024
│    │    └─Conv2d: 3-34                 [256, 512, 7, 7]          2,359,808
│    │    └─BatchNorm2d: 3-35            [256, 512, 7, 7]          1,024
├─AdaptiveAvgPool2d: 1-6                 [256, 512, 1, 1]          --
├─Flatten: 1-7                           [256, 512]                --
├─Linear: 1-8                            [256, 10]                 5,130
==========================================================================================
Total params: 11,178,378
Trainable params: 11,178,378
Non-trainable params: 0
Total mult-adds (G): 444.77
==========================================================================================
Input size (MB): 51.38
Forward/backward pass size (MB): 9813.64
Params size (MB): 44.71
Estimated Total Size (MB): 9909.74
==========================================================================================
train epoch 01/10 100%|===================>[60000/60000] Time: 47.13s->47.13s/7m51.31s | Loss: 0.4439281901041667
evaluate train set 100%|===================>[60000/60000] train accuracy [51745/60000 = 86.24%] train loss = 0.3832
evaluate test set 100%|===================>[10000/10000] test accuracy [8524/10000 = 85.24%] test loss = 0.4194
------------------------------------------------------------------------------------------
train epoch 02/10 100%|===================>[60000/60000] Time: 46.36s->1m33.49s/7m45.47s | Loss: 0.24281984049479166
evaluate train set 100%|===================>[60000/60000] train accuracy [54473/60000 = 90.79%] train loss = 0.2536
evaluate test set 100%|===================>[10000/10000] test accuracy [8926/10000 = 89.26%] test loss = 0.3030
------------------------------------------------------------------------------------------
train epoch 03/10 100%|===================>[60000/60000] Time: 46.24s->2m19.73s/7m44.40s | Loss: 0.19359166666666666
evaluate train set 100%|===================>[60000/60000] train accuracy [56543/60000 = 94.24%] train loss = 0.1616
evaluate test set 100%|===================>[10000/10000] test accuracy [9223/10000 = 92.23%] test loss = 0.2277
------------------------------------------------------------------------------------------
train epoch 04/10 100%|===================>[60000/60000] Time: 45.33s->3m5.05s/7m37.07s | Loss: 0.16384207356770833
evaluate train set 100%|===================>[60000/60000] train accuracy [55965/60000 = 93.28%] train loss = 0.1859
evaluate test set 100%|===================>[10000/10000] test accuracy [9078/10000 = 90.78%] test loss = 0.2592
------------------------------------------------------------------------------------------
train epoch 05/10 100%|===================>[60000/60000] Time: 45.22s->3m50.27s/7m36.47s | Loss: 0.136979345703125
evaluate train set 100%|===================>[60000/60000] train accuracy [56933/60000 = 94.89%] train loss = 0.1405
evaluate test set 100%|===================>[10000/10000] test accuracy [9170/10000 = 91.70%] test loss = 0.2509
------------------------------------------------------------------------------------------
train epoch 06/10 100%|===================>[60000/60000] Time: 45.16s->4m35.43s/7m36.20s | Loss: 0.107924560546875
evaluate train set 100%|===================>[60000/60000] train accuracy [57121/60000 = 95.20%] train loss = 0.1342
evaluate test set 100%|===================>[10000/10000] test accuracy [9108/10000 = 91.08%] test loss = 0.2772
------------------------------------------------------------------------------------------
train epoch 07/10 100%|===================>[60000/60000] Time: 45.06s->5m20.49s/7m35.78s | Loss: 0.0911444580078125
evaluate train set 100%|===================>[60000/60000] train accuracy [58241/60000 = 97.07%] train loss = 0.0814
evaluate test set 100%|===================>[10000/10000] test accuracy [9219/10000 = 92.19%] test loss = 0.2349
------------------------------------------------------------------------------------------
train epoch 08/10 100%|===================>[60000/60000] Time: 45.10s->6m5.60s/7m35.93s | Loss: 0.06606308186848958
evaluate train set 100%|===================>[60000/60000] train accuracy [58251/60000 = 97.08%] train loss = 0.0765
evaluate test set 100%|===================>[10000/10000] test accuracy [9184/10000 = 91.84%] test loss = 0.2814
------------------------------------------------------------------------------------------
train epoch 09/10 100%|===================>[60000/60000] Time: 45.08s->6m50.67s/7m35.83s | Loss: 0.048214595540364585
evaluate train set 100%|===================>[60000/60000] train accuracy [58486/60000 = 97.48%] train loss = 0.0667
evaluate test set 100%|===================>[10000/10000] test accuracy [9189/10000 = 91.89%] test loss = 0.2903
------------------------------------------------------------------------------------------
train epoch 10/10 100%|===================>[60000/60000] Time: 45.20s->7m35.87s/7m35.87s | Loss: 0.03711707763671875
evaluate train set 100%|===================>[60000/60000] train accuracy [58421/60000 = 97.37%] train loss = 0.0721
evaluate test set 100%|===================>[10000/10000] test accuracy [9117/10000 = 91.17%] test loss = 0.3661
------------------------------------------------------------------------------------------
training done!
evaluate train set 100%|===================>[60000/60000] train accuracy [58421/60000 = 97.37%] train loss = 0.0721
evaluate test set 100%|===================>[10000/10000] test accuracy [9117/10000 = 91.17%] test loss = 0.3661
model saved to pretrained_model/ResNet.pt
"""
