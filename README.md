# mocogan-chainer

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/piyo56/mocogan-chainer/blob/master/LICENSE)
[![arXiv Tag](https://img.shields.io/badge/arXiv-1610.07584-brightgreen.svg)](https://arxiv.org/abs/1707.04993)


## Chainer implementation of MoCoGAN

This repository contains an chainer implementation of MoCoGAN.

Paper: [MoCoGAN: Decomposing Motion and Content for Video Generation by Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz](https://arxiv.org/abs/1707.04993).

### Result

preparing...

### Requirements

- Python 3.6
- chainer
- opencv
- pillow

### Dataset

I trained the model using **[MUG Facial Expression Database](https://mug.ee.auth.gr/fed/)**. 

### Usage

#### Preparing Dataset

Firstly, yownload dataset from [MUG Facial Expression Database](https://mug.ee.auth.gr/fed/), and then perform preprocessing:

```
python preprocess.py <downloaded datset path> <save path>
```

#### Training

```
python train.py --dataset <save path> --batchsize <batchsize>
```

#### Generation

```
python generate_samples.py -m <model_file> -d <save_dir> -n <num samples>
```
