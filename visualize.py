import sys, os, glob, subprocess, shutil
import re
from multiprocessing import Process

import numpy as np
from PIL import Image, GifImagePlugin

import chainer
import chainer.cuda
from chainer import Variable
import chainer.functions as F

def to_grid(videos, size):
    """
    arrange images in a (size, size) grid

    videos : target videos array
             shape:(batchsize, video_length, channel, height, width)
    size: this function create size x size grid video
    """

    bs, t, h, w, c = videos.shape
    
    # make (size x size) grid
    if bs < size*size:
        bnum = size*size - bs
        blank_videos = np.zeros((bnum, t, h, w, c))
        videos = np.concatenate((videos, blank_videos))

    for i in range(size):
        row = videos[size*i]

        for j in range(1, size):
            row = np.concatenate((row, videos[size*i+j]), axis=2)

        if i == 0:
            grid_videos = row
        else:
            grid_videos = np.concatenate((grid_videos, row), axis=1)
    
    return grid_videos.astype(np.uint8)

def save_grid_video(video, save_dir, basename, sub_save_dir):
    """
    save video under save_dir.

    input
        video: np.array, shape is (t, y, x, ch)
        save_dir: path to save video
        sub_save_dir: path to save intermidiate files 
    """

    t, y, x, ch = video.shape
    
    image_path = os.path.join(sub_save_dir, 'image')
    os.makedirs(image_path, exist_ok=True)
    for i in range(t):
        image_file = os.path.join(image_path, "{:02d}.jpg".format(i))
        Image.fromarray(video[i,:,:,0:3]).save(image_file)
        
    cmd = 'ffmpeg -y -r 19 -i {} -vcodec libx264 -pix_fmt yuv420p -vf setpts=PTS/0.5 {}'.format(os.path.join(image_path, "%02d.jpg"), os.path.join(save_dir, basename+".mp4"))
    subprocess.call(cmd, shell=True)


def save_video_samples(image_gen, num, size, ch, T, seed, writer):
    @chainer.training.make_extension()
    def make_video(trainer):
        with chainer.using_config('train', False):
            np.random.seed(seed)
            updater = trainer.updater

            save_dir = os.path.join(save_path, 'samples', 'epoch_{:04d}'.format(updater.epoch))
            os.makedirs(save_dir, exist_ok=True)

            # generate samples
            h0 = Variable(updater.converter(image_gen.make_h0(num), updater.device))
            videos = image_gen(h0)
            videos = chainer.cuda.to_cpu(videos.data)
            # (t, bs, ch, x, y)
            videos = (videos + 1.0) / 2. * 255.
            videos = videos.astype(np.uint8)
            videos = videos.transpose(1, 0, 3, 4, 2)

            for i in range(num): # batch
                video_path = os.path.join(save_dir, "{:02d}".format(i+1))
                os.makedirs(video_path, exist_ok=True)
                write_video(videos[i], video_path)
                # np.save(video_path, video[:, i, 0])

    return make_video

def log_tensorboard(image_gen, num, use_labels, seed, writer, save_path):
    @chainer.training.make_extension()
    def log(trainer):
        with chainer.using_config('train', False):
            np.random.seed(seed)
            updater = trainer.updater
            
            # generate samples
            h0 = Variable(updater.converter(image_gen.make_h0(num), updater.device))
            if use_labels:
                if num == 36:
                    labels = np.repeat(np.arange(6), 6)
                    videos, labels = image_gen(h0, labels=labels)
                else:
                    videos, labels = image_gen(h0)
            else:
                videos = image_gen(h0)
            videos = chainer.cuda.to_cpu(videos.data)

            videos = (videos + 1.0) / 2. * 255.
            videos = videos.astype(np.uint8)
            # final videos shape: (bs, t, y, x, ch)
            videos = videos.transpose(1, 0, 3, 4, 2)
            
            # log to tensorboard
            grid_videos = to_grid(videos, int(np.sqrt(num)))
            grid_videos = grid_videos.transpose(0, 3, 1, 2)
            for t in range(len(grid_videos)):
                writer.add_image('{:02d}'.format(t), grid_videos[t,0:3] / 255., updater.epoch)
            
            # save video to directory
            grid_videos = grid_videos.transpose(0, 2, 3, 1)
            save_dir = os.path.join(save_path, 'samples')
            basename = 'epoch_{:04d}'.format(updater.epoch)
            sub_save_dir = os.path.join(save_dir, basename)
            os.makedirs(sub_save_dir, exist_ok=True)
            save_grid_video(grid_videos, save_dir, basename, sub_save_dir)
    return log
