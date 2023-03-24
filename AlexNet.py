import os
import argparse
import torch
from torch import nn

from standard_image_classification import download_FashionMNIST_dataloader, standard_image_classification
from utils import Config, count_linear_conv

NET_NAME = os.path.splitext(os.path.basename(__file__))[0]


def main():
    parser = argparse.ArgumentParser(description='AlexNet Example')
    parser.add_argument('--config', type=str, default="default.yaml", metavar='CONFIG',
                        help='config file path')
    parser.add_argument('--test', action='store_true', default=False,
                        help='For testing the saved Model')
    args = parser.parse_args()
    config = Config(args.config).set_sub_config('AlexNet')

    devices = [index for index in config['gpu'] if index < torch.cuda.device_count()]
    print(f"use gpu: {devices}")
    train_dataloader, test_dataloader = \
        download_FashionMNIST_dataloader(batch_size=config['batch_size'] * len(devices),
                                         test_batch_size=config['test_batch_size'] * len(devices),
                                         resize=config['resize'],
                                         num_workers=config['num_workers'])

    model = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
        nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
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
input test shape torch.Size([2560, 1, 224, 224]) torch.Size([2560])
AlexNet-8 Depth=8 Linear=3 Conv=5
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [256, 10]                 --
├─Conv2d: 1-1                            [256, 96, 54, 54]         11,712
├─ReLU: 1-2                              [256, 96, 54, 54]         --
├─MaxPool2d: 1-3                         [256, 96, 26, 26]         --
├─Conv2d: 1-4                            [256, 256, 26, 26]        614,656
├─ReLU: 1-5                              [256, 256, 26, 26]        --
├─MaxPool2d: 1-6                         [256, 256, 12, 12]        --
├─Conv2d: 1-7                            [256, 384, 12, 12]        885,120
├─ReLU: 1-8                              [256, 384, 12, 12]        --
├─Conv2d: 1-9                            [256, 384, 12, 12]        1,327,488
├─ReLU: 1-10                             [256, 384, 12, 12]        --
├─Conv2d: 1-11                           [256, 256, 12, 12]        884,992
├─ReLU: 1-12                             [256, 256, 12, 12]        --
├─MaxPool2d: 1-13                        [256, 256, 5, 5]          --
├─Flatten: 1-14                          [256, 6400]               --
├─Linear: 1-15                           [256, 4096]               26,218,496
├─ReLU: 1-16                             [256, 4096]               --
├─Dropout: 1-17                          [256, 4096]               --
├─Linear: 1-18                           [256, 4096]               16,781,312
├─ReLU: 1-19                             [256, 4096]               --
├─Dropout: 1-20                          [256, 4096]               --
├─Linear: 1-21                           [256, 10]                 40,970
==========================================================================================
Total params: 46,764,746
Trainable params: 46,764,746
Non-trainable params: 0
Total mult-adds (G): 240.32
==========================================================================================
Input size (MB): 51.38
Forward/backward pass size (MB): 1246.52
Params size (MB): 187.06
Estimated Total Size (MB): 1484.95
==========================================================================================
train epoch 01/10 100%|===================>[60000/60000] Time: 27.60s->27.60s/4m36.01s | Loss: 0.5812106119791667
evaluate train set 100%|===================>[60000/60000] train accuracy [53198/60000 = 88.66%] train loss = 0.3036
evaluate test set 100%|===================>[10000/10000] test accuracy [8743/10000 = 87.43%] test loss = 0.3310
------------------------------------------------------------------------------------------
train epoch 02/10 100%|===================>[60000/60000] Time: 25.94s->53.54s/4m21.25s | Loss: 0.29795250651041666
evaluate train set 100%|===================>[60000/60000] train accuracy [53810/60000 = 89.68%] train loss = 0.2833
evaluate test set 100%|===================>[10000/10000] test accuracy [8852/10000 = 88.52%] test loss = 0.3182
------------------------------------------------------------------------------------------
train epoch 03/10 100%|===================>[60000/60000] Time: 26.35s->1m19.89s/4m25.55s | Loss: 0.2532158854166667
evaluate train set 100%|===================>[60000/60000] train accuracy [55186/60000 = 91.98%] train loss = 0.2142
evaluate test set 100%|===================>[10000/10000] test accuracy [9023/10000 = 90.23%] test loss = 0.2660
------------------------------------------------------------------------------------------
train epoch 04/10 100%|===================>[60000/60000] Time: 26.64s->1m46.53s/4m26.33s | Loss: 0.21780323893229167
evaluate train set 100%|===================>[60000/60000] train accuracy [55704/60000 = 92.84%] train loss = 0.1928
evaluate test set 100%|===================>[10000/10000] test accuracy [9052/10000 = 90.52%] test loss = 0.2550
------------------------------------------------------------------------------------------
train epoch 05/10 100%|===================>[60000/60000] Time: 26.72s->2m13.25s/4m26.51s | Loss: 0.18920016276041668
evaluate train set 100%|===================>[60000/60000] train accuracy [56338/60000 = 93.90%] train loss = 0.1618
evaluate test set 100%|===================>[10000/10000] test accuracy [9146/10000 = 91.46%] test loss = 0.2408
------------------------------------------------------------------------------------------
train epoch 06/10 100%|===================>[60000/60000] Time: 27.10s->2m40.35s/4m28.19s | Loss: 0.167001123046875
evaluate train set 100%|===================>[60000/60000] train accuracy [57316/60000 = 95.53%] train loss = 0.1248
evaluate test set 100%|===================>[10000/10000] test accuracy [9249/10000 = 92.49%] test loss = 0.2155
------------------------------------------------------------------------------------------
train epoch 07/10 100%|===================>[60000/60000] Time: 27.22s->3m7.58s/4m28.87s | Loss: 0.149636328125
evaluate train set 100%|===================>[60000/60000] train accuracy [57343/60000 = 95.57%] train loss = 0.1247
evaluate test set 100%|===================>[10000/10000] test accuracy [9200/10000 = 92.00%] test loss = 0.2263
------------------------------------------------------------------------------------------
train epoch 08/10 100%|===================>[60000/60000] Time: 26.92s->3m34.50s/4m28.12s | Loss: 0.13042865397135417
evaluate train set 100%|===================>[60000/60000] train accuracy [58033/60000 = 96.72%] train loss = 0.0904
evaluate test set 100%|===================>[10000/10000] test accuracy [9248/10000 = 92.48%] test loss = 0.2218
------------------------------------------------------------------------------------------
train epoch 09/10 100%|===================>[60000/60000] Time: 26.56s->4m1.06s/4m27.82s | Loss: 0.11542186686197917
evaluate train set 100%|===================>[60000/60000] train accuracy [58058/60000 = 96.76%] train loss = 0.0884
evaluate test set 100%|===================>[10000/10000] test accuracy [9220/10000 = 92.20%] test loss = 0.2366
------------------------------------------------------------------------------------------
train epoch 10/10 100%|===================>[60000/60000] Time: 26.37s->4m27.43s/4m27.43s | Loss: 0.09769990234375
evaluate train set 100%|===================>[60000/60000] train accuracy [58192/60000 = 96.99%] train loss = 0.0815
evaluate test set 100%|===================>[10000/10000] test accuracy [9200/10000 = 92.00%] test loss = 0.2566
------------------------------------------------------------------------------------------
training done!
evaluate train set 100%|===================>[60000/60000] train accuracy [58192/60000 = 96.99%] train loss = 0.0815
evaluate test set 100%|===================>[10000/10000] test accuracy [9200/10000 = 92.00%] test loss = 0.2566
model saved to pretrained_model/AlexNet.pt
"""
