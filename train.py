import argparse
import os, sys
from pathlib import Path
import pickle

from pytz import timezone
from datetime import datetime

import chainer
from chainer import training
from chainer.training import extensions

from model.net import ImageGenerator
from model.net import ImageDiscriminator
from model.net import VideoDiscriminator
from model.updater import Updater

from datasets import MugDataset, MovingMnistDataset

from visualize import log_tensorboard
from tb_chainer import utils, SummaryWriter

def main():
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset_type', choices=['mug', 'mnist'], default='mug', help="dataset type")
    parser.add_argument('--dataset', default='data/dataset/train', help="dataset root path")
    parser.add_argument('--batchsize', type=int, default=100, help="batchsize")
    parser.add_argument('--max_epoch', type=int, default=1000, help="num learning epochs")
    parser.add_argument('--model', type=str, choices=['normal', 'cgan', 'infogan'], default="normal", help="MoCoGAN model")
    parser.add_argument('--save_name', default=datetime.now(timezone('Asia/Tokyo')).strftime("%Y_%m%d_%H%M"), \
                                          help="save path for log, snapshot etc")
    parser.add_argument('--display_interval', type=int, default=1, help='interval of displaying log to console')
    parser.add_argument('--snapshot_interval', type=int, default=10, help='interval of snapshot')
    parser.add_argument('--log_tensorboard_interval', type=int, default=10, help='interval of log to tensorboard')
    parser.add_argument('--num_gen_samples', type=int, default=36, help='num generate samples')
    parser.add_argument('--dim_zc', type=int, default=50, help='number of dimensions of z content')
    parser.add_argument('--dim_zm', type=int, default=10, help='number of dimensions of z motion')
    parser.add_argument('--n_filters_gen', type=int, default=64, help='number of channelsof image generator')
    parser.add_argument('--n_filters_idis', type=int, default=64, help='number of channel of image discriminator')
    parser.add_argument('--n_filters_vdis', type=int, default=64, help='number of channel of video discriminator')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    # parameters
    size         = 64 # image size
    channel      = 3  # num channels
    video_length = 16 # video length
    dim_zc       = args.dim_zc
    dim_zm       = args.dim_zm
    n_filters_gen  = args.n_filters_gen
    n_filters_idis = args.n_filters_idis
    n_filters_vdis = args.n_filters_vdis
    use_noise = True
    noise_sigma = 0.2

    # Set up dataset
    if args.dataset_type == "mug":
        num_labels = 6
        train_dataset = MugDataset(args.dataset, video_length)
    elif args.dataset_type == "mnist":
        num_labels = 0
        train_dataset = MovingMnistDataset(args.dataset, video_length)
    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize)

    # Set up models
    if args.model == "normal":
        use_label = False
        image_gen = ImageGenerator(dim_zc, dim_zm, num_labels, channel, n_filters_gen, video_length)
        image_dis = ImageDiscriminator(channel, 1, n_filters_gen, use_noise, noise_sigma)
        video_dis = VideoDiscriminator(channel, 1, n_filters_gen, use_noise, noise_sigma)
    elif args.model == "cgan":
        if num_labels == 0: raise ValueError("Called cgan model, but dataset has no label.")
        use_label = True
        image_gen = ImageGenerator(dim_zc, dim_zm, num_labels, channel, n_filters_gen, video_length)
        image_dis = ImageDiscriminator(channel+num_labels, 1, n_filters_gen, use_noise, noise_sigma)
        video_dis = VideoDiscriminator(channel+num_labels, 1, n_filters_gen, use_noise, noise_sigma)
    elif args.model == "infogan":
        if num_labels == 0: raise ValueError("Called cgan model, but dataset has no label.")
        use_label = True
        image_gen = ImageGenerator(dim_zc, dim_zm, num_labels, channel, n_filters_gen, video_length)
        image_dis = ImageDiscriminator(channel, 1+num_labels, n_filters_gen, use_noise, noise_sigma)
        video_dis = VideoDiscriminator(channel, 1+num_labels, n_filters_gen, use_noise, noise_sigma)
    
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        image_gen.to_gpu()
        image_dis.to_gpu()
        video_dis.to_gpu()

    def make_optimizer(model, alpha=1e-3, beta1=0.9, beta2=0.999):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_dec')
        return optimizer

    opt_image_gen = make_optimizer(image_gen, 2e-4, 5e-5, 0.999)
    opt_image_dis = make_optimizer(image_dis, 2e-4, 5e-5, 0.999)
    opt_video_dis = make_optimizer(video_dis, 2e-4, 5e-5, 0.999)

    # tensorboard writer
    writer = SummaryWriter(Path('runs') / args.save_name)

    # updater args
    updater_args = {
        "model":              args.model,
        "models":             (image_gen, image_dis, video_dis),
        "video_length":       video_length,
        "img_size":           size,
        "channel":            channel,
        "dim_zl":             num_labels,
        "iterator":           train_iter,
        "tensorboard_writer": writer,
        "optimizer":          {
            'image_gen':      opt_image_gen,
            'image_dis':      opt_image_dis,
            'video_dis':      opt_video_dis,
        },
        "device":             args.gpu
    }

    # Setup updater
    updater = Updater(**updater_args)

    # Setup logging
    save_path = Path('result') / args.save_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # trainer
    trainer = training.Trainer(updater, (args.max_epoch, 'epoch'), out=save_path)

    # snapshot setting
    snapshot_interval = (args.snapshot_interval, 'epoch')
    trainer.extend(
        extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        image_gen, 'image_gen_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        image_dis, 'image_dis_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        video_dis, 'video_dis_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)

    # loss setting
    display_interval = (args.display_interval, 'epoch')
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'image_gen/loss', 'image_dis/loss', 'video_dis/loss'
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=1))

    # tensorboard-chainer
    log_tensorboard_interval = (args.log_tensorboard_interval, 'epoch')
    if np.sqrt(args.num_gen_samples) % 1.0 != 0:
        raise ValueError('--num_gen_samples must be n^2 (n: natural number).')
    trainer.extend(
        log_tensorboard(image_gen, args.num_gen_samples, video_length, writer),
        trigger=log_tensorboard_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    
    print('[ Training configuration ]')
    print('# gpu: {}'.format(args.gpu))
    print('# minibatch size: {}'.format(args.batchsize))
    print('# max epoch: {}'.format(args.max_epoch))
    print('# num batches: {}'.format(len(train_dataset) // args.batchsize))
    print('# data size: {}'.format(len(train_dataset)))
    print('# data shape: {}'.format(train_dataset[0][0].shape))
    print('# num filters igen: {}'.format(n_filters_gen))
    print('# num filters idis: {}'.format(n_filters_idis))
    print('# num filters vdis: {}'.format(n_filters_vdis))
    print('# use noise: {}(sigma={})'.format(use_noise, noise_sigma))
    print('# use label: {}'.format(use_label))
    print('# snapshot interval: {}'.format(args.snapshot_interval))
    print('# log tensorboard interval: {}'.format(args.log_tensorboard_interval))
    print('# num generate samples: {}'.format(args.num_gen_samples))
    print('')
    
    # start training
    trainer.run()

    if args.gpu >= 0:
        image_gen.to_cpu()
        image_dis.to_cpu()
        video_dis.to_cpu()

    chainer.serializers.save_npz(save_path / 'image_gen_epoch_fianl.npz', image_gen)
    chainer.serializers.save_npz(save_path / 'image_dis_epoch_fianl.npz', image_dis)
    chainer.serializers.save_npz(save_path / 'video_dis_epoch_fianl.npz', video_dis)

if __name__ == '__main__':
    main()
