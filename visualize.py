import os

import chainer
import chainer.cuda
from chainer import Variable
import chainer.functions as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def write_video(x, filepath, ext, fps=25.0):
    ch, frames, height, width = x.shape
    
    imgs = []
    fig = plt.figure()
    x = chainer.cuda.to_cpu(x).transpose(1,2,3,0) / 2. + 0.5
    for i in range(frames):
        img = plt.imshow(x[i], animated=True)
        imgs.append([img])

    ani = animation.ArtistAnimation(fig, imgs, interval=200)
    if ext == 'mp4':
        ani.save(filepath, writer="ffmpeg")
    elif ext == 'gif':
        ani.save(filepath, writer="imagemagick")
    plt.close()

def write_grid_videos(x, filepath, ext):
    fig = plt.figure()
    x = chainer.cuda.to_cpu(x)
    x = x.transpose(0, 2, 3, 4, 1) / 2. + 0.5
    bs, t, h, w, c = x.shape

    if c == 1:
        x = x.reshape(bs, t, h, w)

    sq = int(np.sqrt(bs))
    for i in range(sq):
        row = x[sq*i]

        for j in range(1, sq):
            row = np.concatenate([row, x[sq*i+j]], axis=2)

        if i == 0:
            videos = row
        else:
            videos = np.concatenate([videos, row], axis=1)

    imgs = []
    for i in range(t):
        img = plt.imshow(videos[i], animated=True)
        if c == 1:
            plt.gray()
        imgs.append([img])

    ani = animation.ArtistAnimation(fig, imgs, interval=200)
    if ext == 'mp4':
        ani.save(filepath, writer="ffmpeg")
    elif ext == 'gif':
        ani.save(filepath, writer="imagemagick")
    plt.close()

def save_video_samples(gen, num, size, ch, T, seed, save_path, ext):
    @chainer.training.make_extension()
    def make_video(trainer):
        np.random.seed(seed)
        updater = trainer.updater

        zc = Variable(updater.converter(gru.make_zc(num*num), updater.device))
        xp = chainer.cuda.get_array_module(zc.data)
        
        videos = xp.empty((T, num*num, ch, size, size), dtype=xp.float32)
        ht = Variable(xp.asarray(gru.make_h0(num*num)))
        with chainer.using_config('train', False):
            for i in range(T):
                e = Variable(xp.asarray(gru.make_zm(num*num)))
                zm = gru(ht, e)
                ht = zm
                z = F.concat([zc, zm], axis=1)
                
                videos[i] = gen(z).data
        
        videos = videos.transpose(1,2,0,3,4)

        output_path = os.path.join(save_path, 'samples', 'epoch_{}.{}'.format(updater.epoch, ext))
        write_grid_videos(videos, output_path, ext)
        
    return make_video
