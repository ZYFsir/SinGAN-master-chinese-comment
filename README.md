# SinGAN

原仓库：https://github.com/tamarott/SinGAN

代码基于pytoch 1.4实现。本仓库添加中文注释方便学习。

## 文件结构

- Downloads      SinGAN在BSD100数据集上的SR结果，以及研究者使用的数据
- imgs
- Input
- SIFID      一种图像评分，衡量生成图像与真实图像的相似度
  - inception.py
  - sifid_score.py
- SinGAN
  - functions.py
  - imresize.py     来自于另一个库，模仿matlab的imresize函数
  - manipulate.py
  - models.py
  - training.py
- animation.py    从单幅图像生成动画
- config.py     命令行参数处理，opt的来源
- editing.py     图像剪切
- harmonization.py    将剪贴的图像融合进去
- main_train.py     训练代码
- paint2image.py   手绘图转为图像
- random_samples.py    随机产生一些图像样例
- SR.py   超分辨率

## 难点解读

`functions.adjust_scales2image(real_,opt)`

该函数用于求模型的缩放次数、缩放因子，并返回调整尺寸后的输入图像。在opt中同时定义了`scale_factor`缩放因子、`min_size`最小尺寸、`max_size`最大尺寸。其中最小尺寸意味着最粗糙的尺度下，图像长宽不能小于该值；最大尺寸意味着，最精细的尺度下，图像长宽不能大于该值。

该代码的做法是，根据已知条件先求缩放次数，取整，再根据最小尺寸最大尺寸求真正的缩放因子。

> 举例：输入图像为1980 x 1480，缩放因子为0.5，最小尺寸为100，最大尺寸为250
>
> 设缩放次数为s，则对于最小尺寸这一约束条件有
>
> $1480 * 0.5^s>=100$， 即$s = ceil\{log_{0.5}\frac{100}{1480}\}$
>
> 由于$s$进行了取整操作
