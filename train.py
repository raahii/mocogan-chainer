import argparse
import os, sys
from pytz import timezone
from datetime import datetime

import chainer
from chainer import training
from chainer.training import extensions

from datasets import MugDataset, MovingMnistDataset

# normal gan
from model.net import ImageGenerator
from model.net import ImageDiscriminator
from model.net import VideoDiscriminator
from model.updater import NormalUpdater
# cgan
from model.net import ConditionalImageGenerator
from model.net import ConditionalImageDiscriminator
from model.net import ConditionalVideoDiscriminator
from model.updater import ConditionalGANUpdater
# infogan
from model.net import InfoImageGenerator
from model.net import InfoImageDiscriminator
from model.net import InfoVideoDiscriminator
from model.updater import InfoGANUpdater

from visualize import log_tensorboard
from tb_chainer import utils, SummaryWriter

def main():
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    # parser.add_argument('--save_path', default=None)
    parser.add_argument('--dataset', default='data/dataset')
    parser.add_argument('--batchsize', type=int, default=200)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--use_label', action='store_true')
    parser.add_argument('--categorical_model', '-c',  type=str, default="cGAN")
    parser.add_argument('--save_dirname', default=None)
    parser.add_argument('--display_interval', type=int, default=1, help='Interval of displaying log to console')
    parser.add_argument('--snapshot_interval', type=int, default=10, help='Interval of snapshot')
    parser.add_argument('--gen_samples_interval', type=int, default=5)
    parser.add_argument('--gen_samples_num', type=int, default=36)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    # parameters
    size         = 64
    channel      = 3
    video_length = 16 # num frames
    dim_zc       = 50 # the dimension of the content vector
    dim_zm       = 10 # the dimension of the  motion vector
    n_hidden     = dim_zc + dim_zm
    num_labels   = 6

    n_filters_gen  = 64
    n_filters_idis = 64
    n_filters_vdis = 16

    use_noise = True
    noise_sigma = 0.1

    # print('*** updater_svc ***')
    print('*** updater ***')
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.max_epoch))

    # Set up dataset
    train_dataset = MugDataset(args.dataset, video_length)
    # train_dataset = MovingMnistDataset(args.dataset, T)
    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize)
    print('# data size: {}'.format(len(train_dataset)))
    print('# data shape: {}'.format(train_dataset[0][0].shape))
    print('# num batches: {}'.format(len(train_dataset) // args.batchsize))
    print('# num filters gen: {}'.format(n_filters_gen))
    print('# num filters idis: {}'.format(n_filters_idis))
    print('# num filters vdis: {}'.format(n_filters_vdis))
    print('# use noise: {}'.format(use_noise))
    print('# use label: {}'.format(args.use_label))
    print('# categorical model: {}'.format(args.categorical_model))
    print('\nTraining configuration is above. Start training? [enter]')
    sys.stdin.read(1)
    print('')

    # logging configurations
    if args.save_dirname is None:
        save_dirname = datetime.now(timezone('Asia/Tokyo')).strftime("%Y_%m%d_%H%M")
    else:
        save_dirname = args.save_dirname
    save_path = 'result/' + save_dirname + '/'
    os.makedirs(os.path.join(save_path, 'samples'), exist_ok=True)

    def make_optimizer(model, alpha=1e-3, beta1=0.9, beta2=0.999):
        # optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer = chainer.optimizers.RMSprop(lr=alpha, alpha=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_dec')
        return optimizer

    # Set up models
    if args.use_label:
        if args.categorical_model == "cGAN":
            image_gen = ConditionalImageGenerator(channel, n_filters_gen, \
                                                  video_length, dim_zc, dim_zm, num_labels)
            image_dis = ConditionalImageDiscriminator(channel, 1, n_filters_idis, \
                                                      use_noise, noise_sigma)
            video_dis = ConditionalVideoDiscriminator(channel+num_labels, 1, n_filters_vdis, \
                                                      use_noise, noise_sigma)
        elif args.categorical_model == "infoGAN":
            image_gen = InfoImageGenerator(channel, n_filters_gen, \
                                           video_length, dim_zc, dim_zm, num_labels)
            image_dis = InfoImageDiscriminator(channel, 1, n_filters_idis, \
                                               use_noise, noise_sigma)
            video_dis = InfoVideoDiscriminator(channel, num_labels+1, n_filters_vdis, \
                                                      use_noise, noise_sigma)
        else:
            raise NotImplementedError

    else:
        image_gen = ImageGenerator(channel, n_filters_gen, T=video_length, dim_zc = dim_zc, dim_zm = dim_zm)
        image_dis = ImageDiscriminator(channel, n_filters_gen, use_noise, noise_sigma)
        video_dis = VideoDiscriminator(channel, n_filters_gen, use_noise, noise_sigma)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        image_gen.to_gpu()
        image_dis.to_gpu()
        video_dis.to_gpu()

    opt_image_gen = make_optimizer(image_gen, 5e-5, 0.5)
    opt_image_dis = make_optimizer(image_dis, 5e-5, 0.5)
    opt_video_dis = make_optimizer(video_dis, 5e-5, 0.5)

    # init tensorboard writer
    writer = SummaryWriter(os.path.join('runs', save_dirname))

    # Setup updater
    if args.use_label:
        if args.categorical_model == "cGAN":
            updater = ConditionalGANUpdater(
                models=(image_gen, image_dis, video_dis),
                video_length=video_length,
                img_size=size,
                channel=channel,
                iterator=train_iter,
                tensorboard_writer=writer,
                optimizer={
                    'image_gen': opt_image_gen,
                    'image_dis': opt_image_dis,
                    'video_dis': opt_video_dis,
                },
                device=args.gpu)
        elif args.categorical_model == "infoGAN":
            updater = InfoGANUpdater(
                models=(image_gen, image_dis, video_dis),
                video_length=video_length,
                img_size=size,
                channel=channel,
                iterator=train_iter,
                tensorboard_writer=writer,
                optimizer={
                    'image_gen': opt_image_gen,
                    'image_dis': opt_image_dis,
                    'video_dis': opt_video_dis,
                },
                device=args.gpu)
        else:
            raise NotImplementedError
    else:
        updater = NormalUpdater(
            models=(image_gen, image_dis, video_dis),
            video_length=video_length,
            img_size=size,
            channel=channel,
            iterator=train_iter,
            tensorboard_writer=writer,
            optimizer={
                'image_gen': opt_image_gen,
                'image_dis': opt_image_dis,
                'video_dis': opt_video_dis,
            },
            device=args.gpu)

    # Setup logging
    trainer = training.Trainer(updater, (args.max_epoch, 'epoch'), out=save_path)
    snapshot_interval = (args.snapshot_interval, 'epoch')
    display_interval = (args.display_interval, 'iteration')
    gen_samples_interval = (args.gen_samples_interval, 'epoch')
    trainer.extend(
        extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        image_gen, 'image_gen_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        image_dis, 'image_dis_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        video_dis, 'video_dis_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'image_gen/loss', 'image_dis/loss', 'video_dis/loss'
    ]), trigger=display_interval)

    # logging with tensorboard-chainer
    trainer.extend(
        log_tensorboard(image_gen, args.gen_samples_num, args.use_label, args.seed, writer, save_path),
        trigger=gen_samples_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    
    trainer.run()

    if args.gpu >= 0:
        image_gen.to_cpu()
        image_dis.to_cpu()
        video_dis.to_cpu()

    chainer.serializers.save_npz(os.path.join(save_path, 'image_gen_epoch_fianl.npz'), image_gen)
    chainer.serializers.save_npz(os.path.join(save_path, 'image_dis_epoch_fianl.npz'), image_dis)
    chainer.serializers.save_npz(os.path.join(save_path, 'video_dis_epoch_fianl.npz'), video_dis)

if __name__ == '__main__':
    main()
