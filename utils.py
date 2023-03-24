import datetime
import logging
import math
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import interpolate
from torch import nn


def fix_random(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)

    try:
        import dgl
        dgl.random.seed(seed)
    except ImportError as e:
        # print(e, ". Can't fix random in dgl")
        pass


def xavier_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def interpolate_to_window_size(raw_data: pd.DataFrame, window_size):
    interpolated_data = []
    x1 = np.linspace(0, window_size - 1, len(raw_data))
    x_new = np.linspace(0, window_size - 1, window_size)
    for yi in range(raw_data.shape[1]):
        tck = interpolate.splrep(x1, raw_data.values[:, yi])
        a = interpolate.splev(x_new, tck)
        interpolated_data.append(a)
    interpolated_data = pd.DataFrame(np.array(interpolated_data).T)
    interpolated_data.columns = raw_data.columns
    return interpolated_data


class Logger:
    def __init__(self, log_dir=None, log_filename=None, name_suffix="", append=True):
        self.log_dir = log_dir if log_dir else os.path.join("log")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_filename = log_filename if log_filename else \
            datetime.datetime.now().strftime('%Y-%m-%d') + str(name_suffix) + ".log"
        self.log_path = os.path.join(self.log_dir, self.log_filename)
        mode = "a" if append else "w"
        self.fh = logging.FileHandler(self.log_path, encoding="utf-8", mode=mode)
        self.fh.setLevel(logging.INFO)
        fmt = "%(asctime)s %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt)
        self.fh.setFormatter(formatter)

        self.logger = logging.getLogger(self.log_filename)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.fh)

    def __call__(self, *args, sep=' ', end='\n', print_out=True):
        message = ''
        for idx, s in enumerate(args):
            message += str(s)
            if idx != len(args) - 1:
                message += sep
        if print_out:
            print(message, end=end)
        self.logger.info(message)


def load_pickle_cache(pkl_path, logger, build_func, *args, **kwargs):
    obj = None
    if os.path.isfile(pkl_path):
        if logger is not None:
            logger(f"use pickle file {pkl_path} to speed up")
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
    else:
        if build_func is not None:
            if logger is not None:
                logger(f"pickle file {pkl_path} doesn't exist! Create cache file to speed up")
            obj = build_func(*args, **kwargs)
        else:
            raise NotImplementedError(f"{pkl_path} has no build func")
        with open(pkl_path, "wb") as f:
            pickle.dump(obj, f)
    return obj


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y))


def set_requires_grad(model: nn.Module, requires_grad: bool):
    for param in model.parameters():
        param.requires_grad = requires_grad


def loop_iterable(iterable):
    while True:
        yield from iterable


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_layer(model, layer_set):
    return sum(isinstance(p, layer_set) for p in model.modules())


def count_linear_conv(model):
    return count_layer(model, (nn.Linear, nn.Conv2d))


def count_linear(model):
    return count_layer(model, nn.Linear)


def count_conv(model):
    return count_layer(model, nn.Conv2d)


def get_depth(model):
    if hasattr(model, "depth"):
        return model.depth
    children = list(model.children())
    if len(children) == 0:
        return isinstance(model, (nn.Linear, nn.Conv2d))
    else:
        return sum(get_depth(child) for child in children)


class Config:
    def __init__(self, path):
        self.path = path
        self.config = {}
        self.sub_config = {}
        self.read_config()

    def set_sub_config(self, sub_config_name):
        self.sub_config = self.config.get(sub_config_name, {})
        return self

    def __getitem__(self, key):
        if key in self.sub_config:
            return self.sub_config[key]
        elif key in self.config:
            return self.config[key]
        else:
            raise KeyError(f"key {key} not in config!")

    def __setitem__(self, key, value):
        if key in self.sub_config:
            self.sub_config[key] = value
        elif key in self.config:
            self.config[key] = value
        else:
            raise KeyError(f"key {key} not in config!")

    def get(self, key, value=None):
        if key in self.sub_config:
            return self.sub_config[key]
        if key in self.config:
            return self.config[key]
        if value is not None:
            return value
        else:
            raise KeyError(f"key {key} not in config!")

    def read_config(self):
        """"读取配置"""
        with open(self.path, "r", encoding="utf-8") as yaml_file:
            self.config = yaml.load(yaml_file.read(), Loader=yaml.CLoader)
        return self

    def update_config(self):
        """"更新配置"""
        with open(self.path, 'w', encoding="utf-8") as yaml_file:
            yaml.dump(self.config, yaml_file, indent=4, default_flow_style=False)
        return self


def seconds2format_time_string(seconds):
    d = h = m = s = 0
    if seconds > 60:
        m, s = divmod(seconds, 60)
        m = int(m)
    else:
        s = seconds
    if m > 60:
        h, m = divmod(m, 60)
    if h > 24:
        d, h = divmod(h, 24)

    results = ""
    if d != 0:
        results = results + f"{d}d"
    if h != 0:
        results = results + f"{h}h"
    if m != 0:
        results = results + f"{m}m"
    results = results + f"{s:.2f}s"
    return results


def s2hms(seconds):
    m, s = divmod(seconds, 60)
    m = int(m)
    h, m = divmod(m, 60)
    return h, m, s


def cal_speed_alpha(x, threshold):
    if x < threshold:
        return math.exp((x - threshold) * 19) * 0.5
    else:
        return 1. / (1 + math.exp((threshold - x) * 7))


def epoch_message(epoch, num_epochs, time_cost, total_time_cost, epoch_loss, epoch_n_samples):
    time_cost_mean = total_time_cost / epoch
    speed_alpha = cal_speed_alpha(abs(time_cost - time_cost_mean), 0.3)
    remain_time_cost = (num_epochs - epoch) * (
            time_cost * speed_alpha + time_cost_mean * (1 - speed_alpha))

    message = f"Time: {seconds2format_time_string(time_cost)}->{seconds2format_time_string(total_time_cost)}/" \
              f"{seconds2format_time_string(total_time_cost + remain_time_cost)} | " \
              f"Loss: {epoch_loss / epoch_n_samples}"
    return message


class ProgressBar:
    def __init__(self, total, n_col=20):
        self.total = total
        self.n_col = n_col
        self.current = 0
        self.progress = 0
        self.n_sharp = 0
        self.l_bar = f"{math.ceil(self.progress * 100):>3d}%|"
        self.m_bar = " " * n_col
        self.r_bar = f"[{self.current:>5d}/{self.total:>5d}]"

    def update(self, n):
        self.current += n
        self.progress = self.current / self.total
        self.n_sharp = math.ceil(self.progress / (1. / self.n_col))
        self.l_bar = f"{math.ceil(self.progress * 100):>3d}%|"
        self.m_bar = "=" * (self.n_sharp - 1) + ">" + " " * (self.n_col - self.n_sharp)
        self.r_bar = f"[{self.current:>5d}/{self.total:>5d}]"
        return self.bar

    @property
    def bar(self):
        return self.l_bar + self.m_bar + self.r_bar
