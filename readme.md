# try_ml

PyTorch实现经典模型，帮助入门深度学习


### 已实现经典CV模型

参数量受版本（深度）影响，计算复杂度受输入数据集影响，并非固定。
| 模型-版本   | 参数量 | 计算量  | 核心思想                                                     |
| ----------- | ------ | ------- | ------------------------------------------------------------ |
| LeNet-5     | 61K    | 108.30M | 卷积核，平均池化与Sigmoid                                    |
| AlexNet-8   | 46M    | 240.32G | 卷积层、最大池化层、ReLU激活函数、Dropout                    |
| VGG-11      | 128M   | 1.93T   | 以块来搭建网络                                               |
| NiN-12      | 2M     | 195.54G | 通过1*1卷积来调整channel数，最后使用全局平均池化让宽高归1（降采样到1） |
| GooLeNet-22 | 6M     | 385.59G | Inception块（4种路径合并channel），整个模型分成5个stage，和NiN一样使用1*1卷积调整channel，全局平均池化 |
| ResNet-18   | 11M    | 444.77G | 残差块（让块拟合残差而不是原始变换），1*1卷积+设置stride调整channel和高宽，5个stage，最后的全局平均池化 |

Note：LeNet的输入是（1,28,28）其他模型的输入是（1,224,224)，因此LeNet的计算量会偏小。

### 如何使用？

训练
```shell
python xxx.py
```

测试
```shell
python xxx.py --test
```