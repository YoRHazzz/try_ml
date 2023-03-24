import os
import argparse

import torch
from torch import nn

from standard_image_classification import download_FashionMNIST_dataloader, standard_image_classification
from utils import Config

NET_NAME = os.path.splitext(os.path.basename(__file__))[0]


def NiNBlock(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
    )


def main():
    parser = argparse.ArgumentParser(description='NiN Example')
    parser.add_argument('--config', type=str, default="default.yaml", metavar='CONFIG',
                        help='config file path')
    parser.add_argument('--test', action='store_true', default=False,
                        help='For testing the saved Model')
    args = parser.parse_args()
    config = Config(args.config).set_sub_config('NiN')

    devices = [index for index in config['gpu'] if index < torch.cuda.device_count()]
    print(f"use gpu: {devices}")
    train_dataloader, test_dataloader = \
        download_FashionMNIST_dataloader(batch_size=config['batch_size'] * len(devices),
                                         test_batch_size=config['test_batch_size'] * len(devices),
                                         resize=config['resize'],
                                         num_workers=config['num_workers'])

    model = nn.Sequential(
        NiNBlock(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        NiNBlock(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        NiNBlock(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
        NiNBlock(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
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
input test shape torch.Size([2560, 1, 224, 224]) torch.Size([2560])
NiN-12 Depth=12 Linear=0 Conv=12
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [256, 10]                 --
├─Sequential: 1-1                        [256, 96, 54, 54]         --
│    └─Conv2d: 2-1                       [256, 96, 54, 54]         11,712
│    └─ReLU: 2-2                         [256, 96, 54, 54]         --
│    └─Conv2d: 2-3                       [256, 96, 54, 54]         9,312
│    └─ReLU: 2-4                         [256, 96, 54, 54]         --
│    └─Conv2d: 2-5                       [256, 96, 54, 54]         9,312
│    └─ReLU: 2-6                         [256, 96, 54, 54]         --
├─MaxPool2d: 1-2                         [256, 96, 26, 26]         --
├─Sequential: 1-3                        [256, 256, 26, 26]        --
│    └─Conv2d: 2-7                       [256, 256, 26, 26]        614,656
│    └─ReLU: 2-8                         [256, 256, 26, 26]        --
│    └─Conv2d: 2-9                       [256, 256, 26, 26]        65,792
│    └─ReLU: 2-10                        [256, 256, 26, 26]        --
│    └─Conv2d: 2-11                      [256, 256, 26, 26]        65,792
│    └─ReLU: 2-12                        [256, 256, 26, 26]        --
├─MaxPool2d: 1-4                         [256, 256, 12, 12]        --
├─Sequential: 1-5                        [256, 384, 12, 12]        --
│    └─Conv2d: 2-13                      [256, 384, 12, 12]        885,120
│    └─ReLU: 2-14                        [256, 384, 12, 12]        --
│    └─Conv2d: 2-15                      [256, 384, 12, 12]        147,840
│    └─ReLU: 2-16                        [256, 384, 12, 12]        --
│    └─Conv2d: 2-17                      [256, 384, 12, 12]        147,840
│    └─ReLU: 2-18                        [256, 384, 12, 12]        --
├─MaxPool2d: 1-6                         [256, 384, 5, 5]          --
├─Dropout: 1-7                           [256, 384, 5, 5]          --
├─Sequential: 1-8                        [256, 10, 5, 5]           --
│    └─Conv2d: 2-19                      [256, 10, 5, 5]           34,570
│    └─ReLU: 2-20                        [256, 10, 5, 5]           --
│    └─Conv2d: 2-21                      [256, 10, 5, 5]           110
│    └─ReLU: 2-22                        [256, 10, 5, 5]           --
│    └─Conv2d: 2-23                      [256, 10, 5, 5]           110
│    └─ReLU: 2-24                        [256, 10, 5, 5]           --
├─AdaptiveAvgPool2d: 1-9                 [256, 10, 1, 1]           --
├─Flatten: 1-10                          [256, 10]                 --
==========================================================================================
Total params: 1,992,166
Trainable params: 1,992,166
Non-trainable params: 0
Total mult-adds (G): 195.54
==========================================================================================
Input size (MB): 51.38
Forward/backward pass size (MB): 3124.46
Params size (MB): 7.97
Estimated Total Size (MB): 3183.81
==========================================================================================
train epoch 01/10 100%|===================>[60000/60000] Time: 19.96s->19.96s/3m19.64s | Loss: 1.7296080729166667
evaluate train set 100%|===================>[60000/60000] train accuracy [37318/60000 = 62.20%] train loss = 1.2531
evaluate test set 100%|===================>[10000/10000] test accuracy [6200/10000 = 62.00%] test loss = 1.2693
------------------------------------------------------------------------------------------
train epoch 02/10 100%|===================>[60000/60000] Time: 18.58s->38.55s/3m7.56s | Loss: 0.98047421875
evaluate train set 100%|===================>[60000/60000] train accuracy [48480/60000 = 80.80%] train loss = 0.5564
evaluate test set 100%|===================>[10000/10000] test accuracy [8023/10000 = 80.23%] test loss = 0.5773
------------------------------------------------------------------------------------------
train epoch 03/10 100%|===================>[60000/60000] Time: 18.47s->57.01s/3m6.88s | Loss: 0.52446572265625
evaluate train set 100%|===================>[60000/60000] train accuracy [50982/60000 = 84.97%] train loss = 0.4350
evaluate test set 100%|===================>[10000/10000] test accuracy [8364/10000 = 83.64%] test loss = 0.4703
------------------------------------------------------------------------------------------
train epoch 04/10 100%|===================>[60000/60000] Time: 18.76s->1m15.78s/3m9.38s | Loss: 0.42680149739583334
evaluate train set 100%|===================>[60000/60000] train accuracy [52167/60000 = 86.94%] train loss = 0.3835
evaluate test set 100%|===================>[10000/10000] test accuracy [8623/10000 = 86.23%] test loss = 0.4166
------------------------------------------------------------------------------------------
train epoch 05/10 100%|===================>[60000/60000] Time: 18.70s->1m34.47s/3m8.88s | Loss: 0.3772456705729167
evaluate train set 100%|===================>[60000/60000] train accuracy [52994/60000 = 88.32%] train loss = 0.3266
evaluate test set 100%|===================>[10000/10000] test accuracy [8745/10000 = 87.45%] test loss = 0.3552
------------------------------------------------------------------------------------------
train epoch 06/10 100%|===================>[60000/60000] Time: 18.90s->1m53.37s/3m8.95s | Loss: 0.33488141276041666
evaluate train set 100%|===================>[60000/60000] train accuracy [53675/60000 = 89.46%] train loss = 0.2939
evaluate test set 100%|===================>[10000/10000] test accuracy [8853/10000 = 88.53%] test loss = 0.3252
------------------------------------------------------------------------------------------
train epoch 07/10 100%|===================>[60000/60000] Time: 18.85s->2m12.22s/3m8.88s | Loss: 0.32124498697916665
evaluate train set 100%|===================>[60000/60000] train accuracy [53895/60000 = 89.82%] train loss = 0.2879
evaluate test set 100%|===================>[10000/10000] test accuracy [8869/10000 = 88.69%] test loss = 0.3273
------------------------------------------------------------------------------------------
train epoch 08/10 100%|===================>[60000/60000] Time: 17.97s->2m30.19s/3m6.18s | Loss: 0.30389508463541665
evaluate train set 100%|===================>[60000/60000] train accuracy [54418/60000 = 90.70%] train loss = 0.2629
evaluate test set 100%|===================>[10000/10000] test accuracy [8973/10000 = 89.73%] test loss = 0.2973
------------------------------------------------------------------------------------------
train epoch 09/10 100%|===================>[60000/60000] Time: 18.81s->2m49.00s/3m7.78s | Loss: 0.28689124348958334
evaluate train set 100%|===================>[60000/60000] train accuracy [54582/60000 = 90.97%] train loss = 0.2564
evaluate test set 100%|===================>[10000/10000] test accuracy [8989/10000 = 89.89%] test loss = 0.2957
------------------------------------------------------------------------------------------
train epoch 10/10 100%|===================>[60000/60000] Time: 18.44s->3m7.44s/3m7.44s | Loss: 0.27661363932291666
evaluate train set 100%|===================>[60000/60000] train accuracy [54879/60000 = 91.47%] train loss = 0.2427
evaluate test set 100%|===================>[10000/10000] test accuracy [9042/10000 = 90.42%] test loss = 0.2819
------------------------------------------------------------------------------------------
training done!
evaluate train set 100%|===================>[60000/60000] train accuracy [54879/60000 = 91.47%] train loss = 0.2427
evaluate test set 100%|===================>[10000/10000] test accuracy [9042/10000 = 90.42%] test loss = 0.2819
model saved to pretrained_model/NiN.pt
"""
