import os
import sys
import time

import torch.utils.data
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision import transforms, datasets

from utils import fix_random, count_linear_conv, xavier_init, ProgressBar, epoch_message, get_depth, count_linear, \
    count_conv


def train_one_epoch(model, optimizer, device, train_dataloader, criterion, custom_print=None, prefix=""):
    model.train()
    epoch_loss = torch.zeros(1, device=device)
    start_time = time.time()
    progress_bar = ProgressBar(len(train_dataloader.dataset))
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)

        y_hat = model(X)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.update(len(X))
        epoch_loss = epoch_loss + loss * len(X)
        if custom_print:
            custom_print("\r" + prefix + progress_bar.bar, end=" ")
    time_cost = time.time() - start_time

    return time_cost, epoch_loss, progress_bar.current


@torch.no_grad()
def _evaluate(model, device, test_dataloader, criterion, custom_print=None, prefix=""):
    model.eval()
    n_right = 0
    total_loss = 0
    progress_bar = ProgressBar(len(test_dataloader.dataset))
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)

        y_hat = model(X)

        n_right += torch.eq(y_hat.argmax(dim=1), y).sum()
        total_loss += criterion(y_hat, y).item() * len(X)
        progress_bar.update(len(X))
        if custom_print:
            custom_print("\r" + prefix + progress_bar.bar, end=" ")
    return n_right, total_loss, progress_bar.current


def evaluate(model, device, dataloader, criterion, set_name):
    torch.cuda.empty_cache()
    n_right, total_loss, total_eval_samples = _evaluate(model, device, dataloader, criterion,
                                                        custom_print=print,
                                                        prefix=f"evaluate {set_name} set ")
    acc = n_right / total_eval_samples
    loss = total_loss / total_eval_samples
    print(f"{set_name} accuracy [{n_right}/{total_eval_samples} = {acc * 100:.2f}%] "
          f"{set_name} loss = {loss:.4f}")
    return acc, loss


def standard_image_classification(model, train_dataloader: torch.utils.data.DataLoader,
                                  test_dataloader: torch.utils.data.DataLoader, dataset_name, net_name, devices,
                                  learning_rate,
                                  num_epochs,
                                  eval_interval,
                                  random_seed=None, image_size=(1, 224, 224), train=True):
    if random_seed:
        fix_random(random_seed)
    torch.set_float32_matmul_precision('high')
    os.makedirs('log', exist_ok=True)
    writer = SummaryWriter(os.path.join('log', net_name))

    devices = [index for index in devices if index < torch.cuda.device_count()]
    base_device = torch.device(f"cuda:{devices[0]}")

    data = next(iter(train_dataloader))
    print(f"input train shape {data[0].shape} {data[1].shape}")
    data = next(iter(test_dataloader))
    print(f"input test shape {data[0].shape} {data[1].shape}")
    model.depth = get_depth(model)
    print(f"{net_name}-{model.depth} Depth={model.depth} Linear={count_linear(model)} Conv={count_conv(model)}")
    summary(model, (train_dataloader.batch_size, *image_size), device=base_device, depth=10)
    if train:
        model.apply(xavier_init)
    else:
        model.load_state_dict(torch.load(os.path.join('pretrained_model', net_name + '.pt'), map_location=base_device))
    if torch.__version__[0] == 2:
        print("pytorch2.x feature: compile model")
        torch.compile(model)
    model = nn.DataParallel(model, device_ids=devices)
    criterion = nn.CrossEntropyLoss()

    if train:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        total_time_cost = 0

        for epoch in range(1, num_epochs + 1):
            time_cost, epoch_loss, epoch_n_samples = train_one_epoch(model, optimizer, base_device, train_dataloader,
                                                                     criterion,
                                                                     custom_print=print,
                                                                     prefix=f"train epoch "
                                                                            f"{epoch:0{len(str(num_epochs))}}"
                                                                            f"/{num_epochs} ")
            total_time_cost += time_cost
            print(epoch_message(epoch, num_epochs, time_cost, total_time_cost, epoch_loss.item(), epoch_n_samples))
            writer.add_scalars(net_name + '/' + dataset_name, {'lr': optimizer.param_groups[0]['lr']}, epoch)

            if epoch % eval_interval == 0:
                acc, train_loss = evaluate(model, base_device, train_dataloader, criterion, "train")
                writer.add_scalars(net_name + '/' + dataset_name, {'train_loss': train_loss,
                                                                   'train_accuracy': acc
                                                                   }, epoch)
                acc, test_loss = evaluate(model, base_device, test_dataloader, criterion, "test")
                writer.add_scalars(net_name + '/' + dataset_name, {'test_loss': test_loss,
                                                                   'test_accuracy': acc
                                                                   }, epoch)
            print('-' * 90)

    print("training done!" if train else "load and test!")
    evaluate(model, base_device, train_dataloader, criterion, "train")
    evaluate(model, base_device, test_dataloader, criterion, "test")
    if train:
        os.makedirs("pretrained_model", exist_ok=True)
        ckpt_path = os.path.join("pretrained_model", f"{net_name}.pt")
        torch.save(model.cpu().state_dict(), ckpt_path)
        print(f"model saved to {ckpt_path}")


def download_FashionMNIST_dataloader(batch_size, test_batch_size, num_workers, resize=224):
    # 加载MNIST数据集
    normal_augs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize, antialias=False)
    ])
    train_dataset = datasets.FashionMNIST(root='data/', train=True, transform=normal_augs, download=True)
    test_dataset = datasets.FashionMNIST(root='data/', train=False, transform=normal_augs, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
