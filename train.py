import argparse
import os, sys
from datetime import datetime

import chainer
from chainer import training
from chainer.training import extensions

from dataset import VideoDataset

from model.net import GRU
from model.net import Generator
from model.net import ImageDiscriminator
from model.net import VideoDiscriminator
from model.updater import Updater
from visualize import save_video_samples


sys.path.append(os.path.dirname(__file__))

def make_optimizer(model, alpha=1e-3, beta1=0.9, beta2=0.999):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_dec')
    return optimizer

def main():
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--dataset', default='data/dataset')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--snapshot_interval', type=int, default=5, help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=1, help='Interval of displaying log to console')
    parser.add_argument('--gen_samples_interval', type=int, default=1)
    parser.add_argument('--num_gen_samples', type=int, default=5)

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.max_epoch))

    # Set up dataset
    train_dataset = VideoDataset(args.dataset)
    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize)
    print('# data-size: {}'.format(len(train_dataset)))
    print('# data-shape: {}'.format(train_dataset[0].shape))
    print('# num-batches: {}'.format(len(train_dataset) // args.batchsize))
    print('')

    # config
    T = 16
    generate_num = min(args.num_gen_samples, len(train_dataset))
    seed = 3
    ext = 'gif'
    now = datetime.now().strftime("%Y_%m%d_%H%M")
    save_path = 'result/' + now + '/'

    # Set up models
    gru = GRU()
    gen = Generator()
    image_dis = ImageDiscriminator()
    video_dis = VideoDiscriminator()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        image_dis.to_gpu()
        video_dis.to_gpu()

    opt_gru       = make_optimizer(gru, 2e-4, 0.5)
    opt_gen       = make_optimizer(gen, 2e-4, 0.5)
    opt_image_dis = make_optimizer(image_dis, 2e-4, 0.5)
    opt_video_dis = make_optimizer(video_dis, 2e-4, 0.5)

    # Setup updater
    updater = Updater(
        models=(gru, gen, image_dis, video_dis),
        video_length=16,
        img_size=96,
        iterator=train_iter,
        optimizer={
            'gru': opt_gru,
            'gen': opt_gen,
            'image_dis': opt_image_dis,
            'video_dis': opt_video_dis,
        },
        device=args.gpu)

    # Setup logging
    trainer = training.Trainer(updater, (args.max_epoch, 'epoch'), out=save_path)
    snapshot_interval = (args.snapshot_interval, 'epoch')
    display_interval = (args.display_interval, 'iteration')
    gen_samples_interval = (args.gen_samples_interval, 'epoch')
    trainer.extend(extensions.snapshot_object(
        gru, 'gru_iter_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        image_dis, 'image_dis_iter_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        video_dis, 'vide_dis_iter_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'gru/loss', 'gen/loss', 'image_dis/loss', 'video_dis/loss'
    ]), trigger=display_interval)
    trainer.extend(
        save_video_samples(gru, gen, generate_num, T, seed, save_path, ext),
        trigger=gen_samples_interval)

    trainer.run()

    if args.gpu >= 0:
        gen.to_cpu()
    chainer.serializers.save_npz(os.path.join(savepath, 'gen.npz'), gen)

if __name__ == '__main__':
    main()
