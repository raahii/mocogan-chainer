import os

import chainer
import chainer.cuda
from chainer import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# from notify_slack import post_image

def write_video(x, filepath, ext, fps=25.0):
    ch, frames, height, width = x.shape
    
    imgs = []
    fig = plt.figure()
    x = x.transpose(1,3,2,0) / 2. + 0.5
    for i in range(frames):
        img = plt.imshow(x[i], animated=True)
        imgs.append([img])

    ani = animation.ArtistAnimation(fig, imgs, interval=50)
    if ext == 'mp4':
        ani.save(filepath, writer="ffmpeg")
    elif ext == 'gif':
        ani.save(filepath, writer="imagemagick")

def save_video_samples(gru, gen, num, T, seed, save_path, ext):
    @chainer.training.make_extension()
    def make_video(trainer):
        np.random.seed(seed)
        updater = trainer.updater
        zc = Variable(updater.converter(gru.make_zc(num), updater.device))
        xp = chainer.cuda.get_array_module(zc.data)

        videos = xp.empty((T, num, 3, 96, 96), dtype=xp.float32)
        with chainer.using_config('train', False):
            for i in range(T):
                eps = Variable(xp.asarray(gru.make_zm(num)))
                zm = gru(eps)
                z = xp.concatenate((zc.data, zm.data), axis=1)
                videos[i] = gen(z).data
        
        videos = videos.transpose(1, 2, 0, 3, 4)
        
        output_dir = os.path.join(save_path, 'samples', 'epoch_{}'.format(updater.epoch))
        os.makedirs(output_dir, exist_ok=True)
        for i in range(videos.shape[0]):
            output_path = os.path.join(output_dir, '{}.{}'.format(i, ext))
            write_video(videos[i], output_path, ext)
            # # notify slack
            # post_image(preview_path, str(trainer.updater.epoch)+'epoch')
        
    return make_video
