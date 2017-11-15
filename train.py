import matplotlib
matplotlib.use('Agg')

import argparse
import os, sys
from pytz import timezone
from datetime import datetime

import chainer
from chainer import training
from chainer.training import extensions

from datasets import MugDataset, MovingMnistDataset

from model.net import ImageGenerator
from model.net import ImageDiscriminator
from model.net import VideoDiscriminator
from model.updater import Updater

from visualize import save_video_samples

sys.path.append(os.path.dirname(__file__))

def main():
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', default='data/dataset')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--display_interval', type=int, default=1, help='Interval of displaying log to console')
    parser.add_argument('--snapshot_interval', type=int, default=10, help='Interval of snapshot')
    parser.add_argument('--gen_samples_interval', type=int, default=5)
    parser.add_argument('--num_gen_samples', type=int, default=5)
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

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.max_epoch))

    # Set up dataset
    train_dataset = MugDataset(args.dataset, video_length)
    # train_dataset = MovingMnistDataset(args.dataset, T)
    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize)
    print('# data-size: {}'.format(len(train_dataset)))
    print('# data-shape: {}'.format(train_dataset[0].shape))
    print('# num-batches: {}'.format(len(train_dataset) // args.batchsize))
    print('')

    # logging configurations
    now = datetime.now(timezone('Asia/Tokyo')).strftime("%Y_%m%d_%H%M")
    save_path = 'result/' + now + '/'
    os.makedirs(os.path.join(save_path, 'samples'), exist_ok=True)
    generate_num = min(args.num_gen_samples, len(train_dataset))
    ext = 'gif'

    def make_optimizer(model, alpha=1e-3, beta1=0.9, beta2=0.999):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_dec')
        return optimizer

    # Set up models
    image_gen = ImageGenerator(channel, T=video_length, dim_zc = dim_zc, dim_zm = dim_zm)
    image_dis = ImageDiscriminator(channel, use_noise=True, noise_sigma=0.2)
    video_dis = VideoDiscriminator(channel, use_noise=True, noise_sigma=0.1)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        image_gen.to_gpu()
        image_dis.to_gpu()
        video_dis.to_gpu()

    opt_image_gen = make_optimizer(image_gen, 2e-4, 0.5)
    opt_image_dis = make_optimizer(image_dis, 2e-4, 0.5)
    opt_video_dis = make_optimizer(video_dis, 2e-4, 0.5)

    # Setup updater
    updater = Updater(
        models=(image_gen, image_dis, video_dis),
        video_length=video_length,
        img_size=size,
        channel=channel,
        iterator=train_iter,
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
    trainer.extend(
        save_video_samples(image_gen, generate_num, size, channel, video_length, args.seed, save_path, ext),
        trigger=gen_samples_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    if args.gpu >= 0:
        gen.to_cpu()

    chainer.serializers.save_npz(os.path.join(savepath, 'image_gen_last.npz'), image_gen)
    chainer.serializers.save_npz(os.path.join(savepath, 'image_dis_last.npz'), image_dis)
    chainer.serializers.save_npz(os.path.join(savepath, 'video_dis_last.npz'), video_dis)

if __name__ == '__main__':
    main()
