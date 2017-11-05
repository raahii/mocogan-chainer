# mocogan-chainer

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/piyo56/3dgan-chainer/blob/master/LICENSE)
[![arXiv Tag](https://img.shields.io/badge/arXiv-1610.07584-brightgreen.svg)](https://arxiv.org/abs/1610.07584)


## Chainer implementation of MoCoGAN

This repository contains an chainer implementation of MoCoGAN: Decomposing Motion and Content for Video Generation by Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz.

### Result
Some **good** samples generated videos.

<!-- <img width='33%' src='result/generated_samples/png/7.png'><img width='33%' src='result/generated_samples/png/13.png'><img width='33%' src='result/generated_samples/png/17.png'> -->
<!-- <img width='33%' src='result/generated_samples/png/21.png'><img width='33%' src='result/generated_samples/png/30.png'><img width='33%' src='result/generated_samples/png/31.png'> -->
<!-- <img width='33%' src='result/generated_samples/png/40.png'><img width='33%' src='result/generated_samples/png/97.png'> -->

<!-- ``` -->
<!-- python generate_samples.py result/trained_models/Generator_50epoch.npz <save direcotry> <num to be generated> -->
<!-- ``` -->

### Requirements

- chainer(2.0.1)
- opencv

### Dataset

I train model only using **MUG Facial Expression Database**. 

### Usage

#### Generation

```
python generate_samples.py <model_file> <save_dir> <num samples>
```

#### Preparing Dataset

Firstly, yownload dataset from [MUG Facial Expression Database](https://mug.ee.auth.gr/fed/), and then perform preprocessing:

```
python preprocess.py <downloaded datset path> <save path>
```

#### Training

```
python train.py --dataset=<save path>
```
