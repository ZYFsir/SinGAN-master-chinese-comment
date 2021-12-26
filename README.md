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
  - imresize.py
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

