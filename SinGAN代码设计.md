# SinGAN代码设计

参考该文章的设计：https://zhuanlan.zhihu.com/p/409662511

```shell
torch_base 
├── checkpoints # 存放模型的地方 
├── data        # 定义各种用于训练测试的dataset 
├── eval.py     # 测试代码 
├── loss.py     # 定义各种花里胡哨的loss 
├── metrics.py  # 定义各种约定俗成的评估指标 
├── model       # 定义各种实验中的模型 
├── options.py  # 定义各种实验参数，以命令行形式传入 
├── README.md   # 介绍一下自己的repo 
├── scripts     # 各种训练，测试脚本 
├── train.py    # 训练代码 
└── utils       # 各种工具代码
```

# options.py

使用命令行参数，用argparse库实现。可分成`parse_common_args`、`parse_train_args`、`parse_test_args`三个函数，处理不同情况下的parser参数。我们可以把其实不需要在命令行中自定义，但全局需要用到的参数都放进里面，以防修改。

参考源码及规范，可以给出如下定义

- parse_common_args
  - 模型、数据载入与保存
    - netG
    - netD
    - manual_seed 随机种子
    - input 输入图像路径
    - output 保存输出结果的目录
  - 模型的卷积参数
    - num_channel 图像通道数
    - num_conv_channel_init 最底层卷积层通道数
    - ker_size 卷积核大小
    - num_layer 网络层数
    - stride 卷积步长
    - padd_size 这个应该计算得出
  - 模型的金字塔结构参数
    - scale_factor 默认缩小倍数
    - noise_amp 初始噪声叠加系数
    - min_size 图像最小尺寸
    - max_size 图像最大尺寸
- parse_train_args
  - num_epoch 训练epoch数
  - gamma    scheduler gamma
  - lr_g
  - lr_d
  - beta1
  - num_steps G和D网络各自单独更新次数
  - lambda_grad  梯度惩罚项权重
  - alpha  重建损失权重

## data文件夹

对于SinGan来说，没有数据集，因此这个文件夹下直接存放图片即可。

## model文件夹

存放模型结构

- GenerationModel.py 包含生成器的类
- WDiscriminatorModel.py 包含判别器的类

## utils文件夹

存放各种可复用的util函数或者类

- logger.py  需要参考原文章。用于数据统计、保存曲线、保存模型参数

## train.py

核心训练代码

- trainer类

  - init  初始化命令行参数、日志工具、载入数据、参数优化器、模型

  - train 训练，调用logger存储曲线和图像

    

