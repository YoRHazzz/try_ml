import os
import argparse

import torch
from torch import nn
from torch.nn import functional as F

from standard_image_classification import download_FashionMNIST_dataloader, standard_image_classification
from utils import Config

NET_NAME = os.path.splitext(os.path.basename(__file__))[0]


class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)

    @property
    def depth(self):
        return 2


def main():
    parser = argparse.ArgumentParser(description='GoogLeNet Example')
    parser.add_argument('--config', type=str, default="default.yaml", metavar='CONFIG',
                        help='config file path')
    parser.add_argument('--test', action='store_true', default=False,
                        help='For testing the saved Model')
    args = parser.parse_args()
    config = Config(args.config).set_sub_config('GoogLeNet')

    devices = [index for index in config['gpu'] if index < torch.cuda.device_count()]
    print(f"use gpu: {devices}")
    train_dataloader, test_dataloader = \
        download_FashionMNIST_dataloader(batch_size=config['batch_size'] * len(devices),
                                         test_batch_size=config['test_batch_size'] * len(devices),
                                         resize=config['resize'],
                                         num_workers=config['num_workers'])

    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                       nn.ReLU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                       Inception(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                       Inception(512, 160, (112, 224), (24, 64), 64),
                       Inception(512, 128, (128, 256), (24, 64), 64),
                       Inception(512, 112, (144, 288), (32, 64), 64),
                       Inception(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                       Inception(832, 384, (192, 384), (48, 128), 128),
                       nn.AdaptiveAvgPool2d((1, 1)),
                       nn.Flatten())

    model = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

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
GoogLeNet-22 Depth=22 Linear=1 Conv=57
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [256, 10]                 --
├─Sequential: 1-1                        [256, 64, 56, 56]         --
│    └─Conv2d: 2-1                       [256, 64, 112, 112]       3,200
│    └─ReLU: 2-2                         [256, 64, 112, 112]       --
│    └─MaxPool2d: 2-3                    [256, 64, 56, 56]         --
├─Sequential: 1-2                        [256, 192, 28, 28]        --
│    └─Conv2d: 2-4                       [256, 64, 56, 56]         4,160
│    └─ReLU: 2-5                         [256, 64, 56, 56]         --
│    └─Conv2d: 2-6                       [256, 192, 56, 56]        110,784
│    └─ReLU: 2-7                         [256, 192, 56, 56]        --
│    └─MaxPool2d: 2-8                    [256, 192, 28, 28]        --
├─Sequential: 1-3                        [256, 480, 14, 14]        --
│    └─Inception: 2-9                    [256, 256, 28, 28]        --
│    │    └─Conv2d: 3-1                  [256, 64, 28, 28]         12,352
│    │    └─Conv2d: 3-2                  [256, 96, 28, 28]         18,528
│    │    └─Conv2d: 3-3                  [256, 128, 28, 28]        110,720
│    │    └─Conv2d: 3-4                  [256, 16, 28, 28]         3,088
│    │    └─Conv2d: 3-5                  [256, 32, 28, 28]         12,832
│    │    └─MaxPool2d: 3-6               [256, 192, 28, 28]        --
│    │    └─Conv2d: 3-7                  [256, 32, 28, 28]         6,176
│    └─Inception: 2-10                   [256, 480, 28, 28]        --
│    │    └─Conv2d: 3-8                  [256, 128, 28, 28]        32,896
│    │    └─Conv2d: 3-9                  [256, 128, 28, 28]        32,896
│    │    └─Conv2d: 3-10                 [256, 192, 28, 28]        221,376
│    │    └─Conv2d: 3-11                 [256, 32, 28, 28]         8,224
│    │    └─Conv2d: 3-12                 [256, 96, 28, 28]         76,896
│    │    └─MaxPool2d: 3-13              [256, 256, 28, 28]        --
│    │    └─Conv2d: 3-14                 [256, 64, 28, 28]         16,448
│    └─MaxPool2d: 2-11                   [256, 480, 14, 14]        --
├─Sequential: 1-4                        [256, 832, 7, 7]          --
│    └─Inception: 2-12                   [256, 512, 14, 14]        --
│    │    └─Conv2d: 3-15                 [256, 192, 14, 14]        92,352
│    │    └─Conv2d: 3-16                 [256, 96, 14, 14]         46,176
│    │    └─Conv2d: 3-17                 [256, 208, 14, 14]        179,920
│    │    └─Conv2d: 3-18                 [256, 16, 14, 14]         7,696
│    │    └─Conv2d: 3-19                 [256, 48, 14, 14]         19,248
│    │    └─MaxPool2d: 3-20              [256, 480, 14, 14]        --
│    │    └─Conv2d: 3-21                 [256, 64, 14, 14]         30,784
│    └─Inception: 2-13                   [256, 512, 14, 14]        --
│    │    └─Conv2d: 3-22                 [256, 160, 14, 14]        82,080
│    │    └─Conv2d: 3-23                 [256, 112, 14, 14]        57,456
│    │    └─Conv2d: 3-24                 [256, 224, 14, 14]        226,016
│    │    └─Conv2d: 3-25                 [256, 24, 14, 14]         12,312
│    │    └─Conv2d: 3-26                 [256, 64, 14, 14]         38,464
│    │    └─MaxPool2d: 3-27              [256, 512, 14, 14]        --
│    │    └─Conv2d: 3-28                 [256, 64, 14, 14]         32,832
│    └─Inception: 2-14                   [256, 512, 14, 14]        --
│    │    └─Conv2d: 3-29                 [256, 128, 14, 14]        65,664
│    │    └─Conv2d: 3-30                 [256, 128, 14, 14]        65,664
│    │    └─Conv2d: 3-31                 [256, 256, 14, 14]        295,168
│    │    └─Conv2d: 3-32                 [256, 24, 14, 14]         12,312
│    │    └─Conv2d: 3-33                 [256, 64, 14, 14]         38,464
│    │    └─MaxPool2d: 3-34              [256, 512, 14, 14]        --
│    │    └─Conv2d: 3-35                 [256, 64, 14, 14]         32,832
│    └─Inception: 2-15                   [256, 528, 14, 14]        --
│    │    └─Conv2d: 3-36                 [256, 112, 14, 14]        57,456
│    │    └─Conv2d: 3-37                 [256, 144, 14, 14]        73,872
│    │    └─Conv2d: 3-38                 [256, 288, 14, 14]        373,536
│    │    └─Conv2d: 3-39                 [256, 32, 14, 14]         16,416
│    │    └─Conv2d: 3-40                 [256, 64, 14, 14]         51,264
│    │    └─MaxPool2d: 3-41              [256, 512, 14, 14]        --
│    │    └─Conv2d: 3-42                 [256, 64, 14, 14]         32,832
│    └─Inception: 2-16                   [256, 832, 14, 14]        --
│    │    └─Conv2d: 3-43                 [256, 256, 14, 14]        135,424
│    │    └─Conv2d: 3-44                 [256, 160, 14, 14]        84,640
│    │    └─Conv2d: 3-45                 [256, 320, 14, 14]        461,120
│    │    └─Conv2d: 3-46                 [256, 32, 14, 14]         16,928
│    │    └─Conv2d: 3-47                 [256, 128, 14, 14]        102,528
│    │    └─MaxPool2d: 3-48              [256, 528, 14, 14]        --
│    │    └─Conv2d: 3-49                 [256, 128, 14, 14]        67,712
│    └─MaxPool2d: 2-17                   [256, 832, 7, 7]          --
├─Sequential: 1-5                        [256, 1024]               --
│    └─Inception: 2-18                   [256, 832, 7, 7]          --
│    │    └─Conv2d: 3-50                 [256, 256, 7, 7]          213,248
│    │    └─Conv2d: 3-51                 [256, 160, 7, 7]          133,280
│    │    └─Conv2d: 3-52                 [256, 320, 7, 7]          461,120
│    │    └─Conv2d: 3-53                 [256, 32, 7, 7]           26,656
│    │    └─Conv2d: 3-54                 [256, 128, 7, 7]          102,528
│    │    └─MaxPool2d: 3-55              [256, 832, 7, 7]          --
│    │    └─Conv2d: 3-56                 [256, 128, 7, 7]          106,624
│    └─Inception: 2-19                   [256, 1024, 7, 7]         --
│    │    └─Conv2d: 3-57                 [256, 384, 7, 7]          319,872
│    │    └─Conv2d: 3-58                 [256, 192, 7, 7]          159,936
│    │    └─Conv2d: 3-59                 [256, 384, 7, 7]          663,936
│    │    └─Conv2d: 3-60                 [256, 48, 7, 7]           39,984
│    │    └─Conv2d: 3-61                 [256, 128, 7, 7]          153,728
│    │    └─MaxPool2d: 3-62              [256, 832, 7, 7]          --
│    │    └─Conv2d: 3-63                 [256, 128, 7, 7]          106,624
│    └─AdaptiveAvgPool2d: 2-20           [256, 1024, 1, 1]         --
│    └─Flatten: 2-21                     [256, 1024]               --
├─Linear: 1-6                            [256, 10]                 10,250
==========================================================================================
Total params: 5,977,530
Trainable params: 5,977,530
Non-trainable params: 0
Total mult-adds (G): 385.59
==========================================================================================
Input size (MB): 51.38
Forward/backward pass size (MB): 6607.20
Params size (MB): 23.91
Estimated Total Size (MB): 6682.49
==========================================================================================
train epoch 01/10 100%|===================>[60000/60000] Time: 52.28s->52.28s/8m42.84s | Loss: 1.0666764973958334
evaluate train set 100%|===================>[60000/60000] train accuracy [47873/60000 = 79.79%] train loss = 0.5373
evaluate test set 100%|===================>[10000/10000] test accuracy [7907/10000 = 79.07%] test loss = 0.5609
------------------------------------------------------------------------------------------
train epoch 02/10 100%|===================>[60000/60000] Time: 50.81s->1m43.09s/8m29.81s | Loss: 0.46940426432291665
evaluate train set 100%|===================>[60000/60000] train accuracy [51076/60000 = 85.13%] train loss = 0.3942
evaluate test set 100%|===================>[10000/10000] test accuracy [8385/10000 = 83.85%] test loss = 0.4275
------------------------------------------------------------------------------------------
train epoch 03/10 100%|===================>[60000/60000] Time: 49.97s->2m33.07s/8m22.93s | Loss: 0.3674556966145833
evaluate train set 100%|===================>[60000/60000] train accuracy [52146/60000 = 86.91%] train loss = 0.3413
evaluate test set 100%|===================>[10000/10000] test accuracy [8572/10000 = 85.72%] test loss = 0.3819
------------------------------------------------------------------------------------------
train epoch 04/10 100%|===================>[60000/60000] Time: 49.84s->3m22.91s/8m22.03s | Loss: 0.3147691080729167
evaluate train set 100%|===================>[60000/60000] train accuracy [53929/60000 = 89.88%] train loss = 0.2674
evaluate test set 100%|===================>[10000/10000] test accuracy [8891/10000 = 88.91%] test loss = 0.3045
------------------------------------------------------------------------------------------
train epoch 05/10 100%|===================>[60000/60000] Time: 48.77s->4m11.68s/8m15.53s | Loss: 0.28608518880208333
evaluate train set 100%|===================>[60000/60000] train accuracy [53259/60000 = 88.76%] train loss = 0.2934
evaluate test set 100%|===================>[10000/10000] test accuracy [8770/10000 = 87.70%] test loss = 0.3340
------------------------------------------------------------------------------------------
train epoch 06/10 100%|===================>[60000/60000] Time: 48.67s->5m0.34s/8m15.01s | Loss: 0.24607120768229165
evaluate train set 100%|===================>[60000/60000] train accuracy [55098/60000 = 91.83%] train loss = 0.2179
evaluate test set 100%|===================>[10000/10000] test accuracy [9078/10000 = 90.78%] test loss = 0.2539
------------------------------------------------------------------------------------------
train epoch 07/10 100%|===================>[60000/60000] Time: 48.81s->5m49.15s/8m15.61s | Loss: 0.22895330403645833
evaluate train set 100%|===================>[60000/60000] train accuracy [55194/60000 = 91.99%] train loss = 0.2211
evaluate test set 100%|===================>[10000/10000] test accuracy [9076/10000 = 90.76%] test loss = 0.2558
------------------------------------------------------------------------------------------
train epoch 08/10 100%|===================>[60000/60000] Time: 49.26s->6m38.41s/8m17.09s | Loss: 0.215218359375
evaluate train set 100%|===================>[60000/60000] train accuracy [55654/60000 = 92.76%] train loss = 0.1940
evaluate test set 100%|===================>[10000/10000] test accuracy [9146/10000 = 91.46%] test loss = 0.2340
------------------------------------------------------------------------------------------
train epoch 09/10 100%|===================>[60000/60000] Time: 48.74s->7m27.15s/8m15.89s | Loss: 0.19618486328125
evaluate train set 100%|===================>[60000/60000] train accuracy [55682/60000 = 92.80%] train loss = 0.1923
evaluate test set 100%|===================>[10000/10000] test accuracy [9107/10000 = 91.07%] test loss = 0.2419
------------------------------------------------------------------------------------------
train epoch 10/10 100%|===================>[60000/60000] Time: 48.61s->8m15.76s/8m15.76s | Loss: 0.18421373697916665
evaluate train set 100%|===================>[60000/60000] train accuracy [55892/60000 = 93.15%] train loss = 0.1825
evaluate test set 100%|===================>[10000/10000] test accuracy [9140/10000 = 91.40%] test loss = 0.2333
------------------------------------------------------------------------------------------
training done!
evaluate train set 100%|===================>[60000/60000] train accuracy [55892/60000 = 93.15%] train loss = 0.1825
evaluate test set 100%|===================>[10000/10000] test accuracy [9140/10000 = 91.40%] test loss = 0.2333
model saved to pretrained_model/GoogLeNet.pt
"""
