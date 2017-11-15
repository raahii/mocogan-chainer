import os

import numpy as np
from PIL import Image, GifImagePlugin

import chainer
import chainer.cuda
from chainer import Variable
import chainer.functions as F

def write_grid_videos(x, save_dir):
    """
    create grid video
     * save all frames
     * create an gif image with imageio

    x : (batchsize, video_length, channel, height, width)
    """

    x = chainer.cuda.to_cpu(x)
    x = x.transpose(0, 1, 3, 4, 2)
    x = ((x / 2. + 0.5) * 255).astype(np.uint8)
    bs, t, h, w, c = x.shape

    # if c == 1:
    #     x = x.reshape(bs, t, h, w)
    
    # make (sq x sq) grid
    sq = int(np.sqrt(bs))
    for i in range(sq):
        row = x[sq*i]

        for j in range(1, sq):
            row = np.concatenate([row, x[sq*i+j]], axis=2)

        if i == 0:
            videos = row
        else:
            videos = np.concatenate([videos, row], axis=1)
    
    # save frames
    gif_imgs = []
    for i in range(t):
        grid_frame = Image.fromarray(videos[i])
        fname = os.path.join(save_dir, "{:02d}.jpg".format(i+1))
        grid_frame.save(fname, 'JPEG', quality=100, optimize=True)
        gif_imgs.append(grid_frame)

    fname = os.path.join(save_dir+".gif")
    gif_imgs[0].save(fname, save_all=True, append_images=gif_imgs[1:], loop=True)

def save_video_samples(image_gen, num, size, ch, T, seed, save_path):
    @chainer.training.make_extension()
    def make_video(trainer):
        np.random.seed(seed)
        updater = trainer.updater

        h0 = Variable(updater.converter(image_gen.make_h0(num*num), updater.device))
        videos = image_gen(h0)

        output_dir = os.path.join(save_path, 'samples', 'epoch_{}'.format(updater.epoch))
        os.makedirs(output_dir, exist_ok=True)
        write_grid_videos(videos.data, output_dir)
        
    return make_video
