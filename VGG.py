import os
import argparse

import torch
from torch import nn

from standard_image_classification import download_FashionMNIST_dataloader, standard_image_classification
from utils import Config

NET_NAME = os.path.splitext(os.path.basename(__file__))[0]


class VGGBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_convs):
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.layers.append(nn.ReLU())
            in_channels = out_channels
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    parser = argparse.ArgumentParser(description='VGG Example')
    parser.add_argument('--config', type=str, default="default.yaml", metavar='CONFIG',
                        help='config file path')
    parser.add_argument('--test', action='store_true', default=False,
                        help='For testing the saved Model')
    args = parser.parse_args()
    config = Config(args.config).set_sub_config('VGG')

    devices = [index for index in config['gpu'] if index < torch.cuda.device_count()]
    print(f"use gpu: {devices}")
    train_dataloader, test_dataloader = \
        download_FashionMNIST_dataloader(batch_size=config['batch_size'] * len(devices),
                                         test_batch_size=config['test_batch_size'] * len(devices),
                                         resize=config['resize'],
                                         num_workers=config['num_workers'])

    model = nn.Sequential(
        VGGBlock(1, 1, 64),
        VGGBlock(1, 64, 128),
        VGGBlock(2, 128, 256),
        VGGBlock(2, 256, 512),
        VGGBlock(2, 512, 512), nn.Flatten(),
        nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 10)
    )

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
input test shape torch.Size([512, 1, 224, 224]) torch.Size([512])
VGG-11 Depth=11 Linear=3 Conv=8
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [256, 10]                 --
├─VGGBlock: 1-1                          [256, 64, 112, 112]       --
│    └─ModuleList: 2-1                   --                        --
│    │    └─Conv2d: 3-1                  [256, 64, 224, 224]       640
│    │    └─ReLU: 3-2                    [256, 64, 224, 224]       --
│    │    └─MaxPool2d: 3-3               [256, 64, 112, 112]       --
├─VGGBlock: 1-2                          [256, 128, 56, 56]        --
│    └─ModuleList: 2-2                   --                        --
│    │    └─Conv2d: 3-4                  [256, 128, 112, 112]      73,856
│    │    └─ReLU: 3-5                    [256, 128, 112, 112]      --
│    │    └─MaxPool2d: 3-6               [256, 128, 56, 56]        --
├─VGGBlock: 1-3                          [256, 256, 28, 28]        --
│    └─ModuleList: 2-3                   --                        --
│    │    └─Conv2d: 3-7                  [256, 256, 56, 56]        295,168
│    │    └─ReLU: 3-8                    [256, 256, 56, 56]        --
│    │    └─Conv2d: 3-9                  [256, 256, 56, 56]        590,080
│    │    └─ReLU: 3-10                   [256, 256, 56, 56]        --
│    │    └─MaxPool2d: 3-11              [256, 256, 28, 28]        --
├─VGGBlock: 1-4                          [256, 512, 14, 14]        --
│    └─ModuleList: 2-4                   --                        --
│    │    └─Conv2d: 3-12                 [256, 512, 28, 28]        1,180,160
│    │    └─ReLU: 3-13                   [256, 512, 28, 28]        --
│    │    └─Conv2d: 3-14                 [256, 512, 28, 28]        2,359,808
│    │    └─ReLU: 3-15                   [256, 512, 28, 28]        --
│    │    └─MaxPool2d: 3-16              [256, 512, 14, 14]        --
├─VGGBlock: 1-5                          [256, 512, 7, 7]          --
│    └─ModuleList: 2-5                   --                        --
│    │    └─Conv2d: 3-17                 [256, 512, 14, 14]        2,359,808
│    │    └─ReLU: 3-18                   [256, 512, 14, 14]        --
│    │    └─Conv2d: 3-19                 [256, 512, 14, 14]        2,359,808
│    │    └─ReLU: 3-20                   [256, 512, 14, 14]        --
│    │    └─MaxPool2d: 3-21              [256, 512, 7, 7]          --
├─Flatten: 1-6                           [256, 25088]              --
├─Linear: 1-7                            [256, 4096]               102,764,544
├─ReLU: 1-8                              [256, 4096]               --
├─Dropout: 1-9                           [256, 4096]               --
├─Linear: 1-10                           [256, 4096]               16,781,312
├─ReLU: 1-11                             [256, 4096]               --
├─Dropout: 1-12                          [256, 4096]               --
├─Linear: 1-13                           [256, 10]                 40,970
==========================================================================================
Total params: 128,806,154
Trainable params: 128,806,154
Non-trainable params: 0
Total mult-adds (T): 1.93
==========================================================================================
Input size (MB): 51.38
Forward/backward pass size (MB): 15225.34
Params size (MB): 515.22
Estimated Total Size (MB): 15791.95
==========================================================================================
train epoch 01/10 100%|===================>[60000/60000] Time: 2m13.33s->2m13.33s/22m13.26s | Loss: 0.5331194661458334
evaluate train set 100%|===================>[60000/60000] train accuracy [54220/60000 = 90.37%] train loss = 0.2651
evaluate test set 100%|===================>[10000/10000] test accuracy [8935/10000 = 89.35%] test loss = 0.2893
------------------------------------------------------------------------------------------
train epoch 02/10 100%|===================>[60000/60000] Time: 2m9.20s->4m22.52s/21m36.09s | Loss: 0.25981728515625
evaluate train set 100%|===================>[60000/60000] train accuracy [55686/60000 = 92.81%] train loss = 0.2084
evaluate test set 100%|===================>[10000/10000] test accuracy [9126/10000 = 91.26%] test loss = 0.2466
------------------------------------------------------------------------------------------
train epoch 03/10 100%|===================>[60000/60000] Time: 2m8.68s->6m31.21s/21m31.99s | Loss: 0.20838564453125
evaluate train set 100%|===================>[60000/60000] train accuracy [56464/60000 = 94.11%] train loss = 0.1619
evaluate test set 100%|===================>[10000/10000] test accuracy [9213/10000 = 92.13%] test loss = 0.2186
------------------------------------------------------------------------------------------
train epoch 04/10 100%|===================>[60000/60000] Time: 2m8.44s->8m39.64s/21m30.27s | Loss: 0.17379952799479167
evaluate train set 100%|===================>[60000/60000] train accuracy [57277/60000 = 95.46%] train loss = 0.1233
evaluate test set 100%|===================>[10000/10000] test accuracy [9288/10000 = 92.88%] test loss = 0.2022
------------------------------------------------------------------------------------------
train epoch 05/10 100%|===================>[60000/60000] Time: 2m8.52s->10m48.16s/21m30.78s | Loss: 0.14617067057291666
evaluate train set 100%|===================>[60000/60000] train accuracy [57748/60000 = 96.25%] train loss = 0.1028
evaluate test set 100%|===================>[10000/10000] test accuracy [9273/10000 = 92.73%] test loss = 0.2095
------------------------------------------------------------------------------------------
train epoch 06/10 100%|===================>[60000/60000] Time: 2m8.42s->12m56.58s/21m30.27s | Loss: 0.11877213541666666
evaluate train set 100%|===================>[60000/60000] train accuracy [58549/60000 = 97.58%] train loss = 0.0702
evaluate test set 100%|===================>[10000/10000] test accuracy [9322/10000 = 93.22%] test loss = 0.2099
------------------------------------------------------------------------------------------
train epoch 07/10 100%|===================>[60000/60000] Time: 2m8.43s->15m5.01s/21m30.35s | Loss: 0.09514195963541666
evaluate train set 100%|===================>[60000/60000] train accuracy [58843/60000 = 98.07%] train loss = 0.0604
evaluate test set 100%|===================>[10000/10000] test accuracy [9330/10000 = 93.30%] test loss = 0.2056
------------------------------------------------------------------------------------------
train epoch 08/10 100%|===================>[60000/60000] Time: 2m8.34s->17m13.35s/21m30.08s | Loss: 0.0743099853515625
evaluate train set 100%|===================>[60000/60000] train accuracy [58791/60000 = 97.99%] train loss = 0.0564
evaluate test set 100%|===================>[10000/10000] test accuracy [9207/10000 = 92.07%] test loss = 0.2880
------------------------------------------------------------------------------------------
train epoch 09/10 100%|===================>[60000/60000] Time: 2m8.27s->19m21.62s/21m29.91s | Loss: 0.06344943440755209
evaluate train set 100%|===================>[60000/60000] train accuracy [59231/60000 = 98.72%] train loss = 0.0358
evaluate test set 100%|===================>[10000/10000] test accuracy [9295/10000 = 92.95%] test loss = 0.2726
------------------------------------------------------------------------------------------
train epoch 10/10 100%|===================>[60000/60000] Time: 2m8.34s->21m29.96s/21m29.96s | Loss: 0.04519087727864583
evaluate train set 100%|===================>[60000/60000] train accuracy [59695/60000 = 99.49%] train loss = 0.0158
evaluate test set 100%|===================>[10000/10000] test accuracy [9319/10000 = 93.19%] test loss = 0.3009
------------------------------------------------------------------------------------------
training done!
evaluate train set 100%|===================>[60000/60000] train accuracy [59695/60000 = 99.49%] train loss = 0.0158
evaluate test set 100%|===================>[10000/10000] test accuracy [9319/10000 = 93.19%] test loss = 0.3009
model saved to pretrained_model/VGG.pt
"""
