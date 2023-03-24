import os
import argparse

import torch
from torch import nn

from standard_image_classification import download_FashionMNIST_dataloader, standard_image_classification
from utils import Config

NET_NAME = os.path.splitext(os.path.basename(__file__))[0]


def main():
    parser = argparse.ArgumentParser(description='LeNet Example')
    parser.add_argument('--config', type=str, default="default.yaml", metavar='CONFIG',
                        help='config file path')
    parser.add_argument('--test', action='store_true', default=False,
                        help='For testing the saved Model')
    args = parser.parse_args()
    config = Config(args.config).set_sub_config('LeNet')

    devices = [index for index in config['gpu'] if index < torch.cuda.device_count()]
    print(f"use gpu: {devices}")
    train_dataloader, test_dataloader = \
        download_FashionMNIST_dataloader(batch_size=config['batch_size'] * len(devices),
                                         test_batch_size=config['test_batch_size'] * len(devices),
                                         resize=config['resize'],
                                         num_workers=config['num_workers'])

    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10)
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
input train shape torch.Size([256, 1, 28, 28]) torch.Size([256])
input test shape torch.Size([2560, 1, 28, 28]) torch.Size([2560])
LeNet-5 Depth=5 Linear=3 Conv=2
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [256, 10]                 --
├─Conv2d: 1-1                            [256, 6, 28, 28]          156
├─Sigmoid: 1-2                           [256, 6, 28, 28]          --
├─AvgPool2d: 1-3                         [256, 6, 14, 14]          --
├─Conv2d: 1-4                            [256, 16, 10, 10]         2,416
├─Sigmoid: 1-5                           [256, 16, 10, 10]         --
├─AvgPool2d: 1-6                         [256, 16, 5, 5]           --
├─Flatten: 1-7                           [256, 400]                --
├─Linear: 1-8                            [256, 120]                48,120
├─Sigmoid: 1-9                           [256, 120]                --
├─Linear: 1-10                           [256, 84]                 10,164
├─Sigmoid: 1-11                          [256, 84]                 --
├─Linear: 1-12                           [256, 10]                 850
==========================================================================================
Total params: 61,706
Trainable params: 61,706
Non-trainable params: 0
Total mult-adds (M): 108.30
==========================================================================================
Input size (MB): 0.80
Forward/backward pass size (MB): 13.35
Params size (MB): 0.25
Estimated Total Size (MB): 14.40
==========================================================================================
train epoch 01/10 100%|===================>[60000/60000] Time: 4.32s->4.32s/43.16s | Loss: 1.828145703125
evaluate train set 100%|===================>[60000/60000] train accuracy [37713/60000 = 62.85%] train loss = 1.0311
evaluate test set 100%|===================>[10000/10000] test accuracy [6271/10000 = 62.71%] test loss = 1.0357
------------------------------------------------------------------------------------------
train epoch 02/10 100%|===================>[60000/60000] Time: 3.14s->7.46s/33.15s | Loss: 0.8878319661458334
evaluate train set 100%|===================>[60000/60000] train accuracy [42511/60000 = 70.85%] train loss = 0.7959
evaluate test set 100%|===================>[10000/10000] test accuracy [7046/10000 = 70.46%] test loss = 0.8098
------------------------------------------------------------------------------------------
train epoch 03/10 100%|===================>[60000/60000] Time: 2.98s->10.44s/32.02s | Loss: 0.7355580078125
evaluate train set 100%|===================>[60000/60000] train accuracy [44736/60000 = 74.56%] train loss = 0.6786
evaluate test set 100%|===================>[10000/10000] test accuracy [7418/10000 = 74.18%] test loss = 0.6964
------------------------------------------------------------------------------------------
train epoch 04/10 100%|===================>[60000/60000] Time: 3.20s->13.64s/33.98s | Loss: 0.64453046875
evaluate train set 100%|===================>[60000/60000] train accuracy [45980/60000 = 76.63%] train loss = 0.6093
evaluate test set 100%|===================>[10000/10000] test accuracy [7596/10000 = 75.96%] test loss = 0.6313
------------------------------------------------------------------------------------------
train epoch 05/10 100%|===================>[60000/60000] Time: 3.18s->16.81s/33.57s | Loss: 0.5954639973958333
evaluate train set 100%|===================>[60000/60000] train accuracy [46672/60000 = 77.79%] train loss = 0.5849
evaluate test set 100%|===================>[10000/10000] test accuracy [7670/10000 = 76.70%] test loss = 0.6114
------------------------------------------------------------------------------------------
train epoch 06/10 100%|===================>[60000/60000] Time: 3.09s->19.91s/33.08s | Loss: 0.5582283854166666
evaluate train set 100%|===================>[60000/60000] train accuracy [47951/60000 = 79.92%] train loss = 0.5363
evaluate test set 100%|===================>[10000/10000] test accuracy [7890/10000 = 78.90%] test loss = 0.5639
------------------------------------------------------------------------------------------
train epoch 07/10 100%|===================>[60000/60000] Time: 3.00s->22.91s/32.48s | Loss: 0.5252293294270833
evaluate train set 100%|===================>[60000/60000] train accuracy [48539/60000 = 80.90%] train loss = 0.5163
evaluate test set 100%|===================>[10000/10000] test accuracy [7983/10000 = 79.83%] test loss = 0.5430
------------------------------------------------------------------------------------------
train epoch 08/10 100%|===================>[60000/60000] Time: 3.15s->26.06s/32.58s | Loss: 0.502067578125
evaluate train set 100%|===================>[60000/60000] train accuracy [49386/60000 = 82.31%] train loss = 0.4847
evaluate test set 100%|===================>[10000/10000] test accuracy [8102/10000 = 81.02%] test loss = 0.5132
------------------------------------------------------------------------------------------
train epoch 09/10 100%|===================>[60000/60000] Time: 3.17s->29.23s/32.48s | Loss: 0.4797150716145833
evaluate train set 100%|===================>[60000/60000] train accuracy [49650/60000 = 82.75%] train loss = 0.4728
evaluate test set 100%|===================>[10000/10000] test accuracy [8168/10000 = 81.68%] test loss = 0.5002
------------------------------------------------------------------------------------------
train epoch 10/10 100%|===================>[60000/60000] Time: 3.23s->32.46s/32.46s | Loss: 0.46196787109375
evaluate train set 100%|===================>[60000/60000] train accuracy [50077/60000 = 83.46%] train loss = 0.4516
evaluate test set 100%|===================>[10000/10000] test accuracy [8207/10000 = 82.07%] test loss = 0.4798
------------------------------------------------------------------------------------------
training done!
evaluate train set 100%|===================>[60000/60000] train accuracy [50077/60000 = 83.46%] train loss = 0.4516
evaluate test set 100%|===================>[10000/10000] test accuracy [8207/10000 = 82.07%] test loss = 0.4798
model saved to pretrained_model/LeNet.pt
"""
