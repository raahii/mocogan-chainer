import sys, os, glob, shutil
import re
from multiprocessing import Process

import numpy as np
from PIL import Image, GifImagePlugin

import chainer
import chainer.cuda
from chainer import Variable
import chainer.functions as F

import subprocess as sp

def to_sequence(video, horizontally=True):
    """Convert a video to an image with frames horizontally aligned

    :param np.ndarray video: video (dim=5, axis=(num, channel, height, width))
    :param bool horizontally: whether concatenate horizontally or not (vertically)
    """
    # write video
    N, C, H, W = video.shape
    
    axis = horizontally and 2 or 1
    seq_image = video[0]
    for i in range(1, N):
        seq_image = np.concatenate((seq_image, video[i]), axis=axis)

    return seq_image

def to_grid(videos, size):
    """
    Convert videos to a size x size grid video

    :param np.ndarray video: video (dim=5, axis=(video_len, batchsize, channel, height, width))
    :param int size: size of one side of the grid
    """

    t, bs, c, h, w = videos.shape
    
    # make (size x size) grid
    if bs < size*size:
        bnum = size*size - bs
        blank_videos = np.zeros((t, bnum, c, h, w), dtype=np.uint8)
        videos = np.concatenate((videos, blank_videos), axis=1)

    grid_video = np.empty((t, c, size*h, size*w), dtype=videos.dtype)
    for i in range(size):
        for j in range(size):
            grid_video[:, :, i*h:i*h+h, j*w:j*w+w] = videos[:, i*size+j]

    return grid_video

def save_frames(video, dirname):
    for i, v in enumerate(video):
        filename = os.path.join(dirname, "{:02d}.jpg".format(i))
        Image.fromarray(v).save(filename)

def save_video(video, filename):
    t, h, w, c  = video.shape
    command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', '{}x{}'.format(h,w),
               '-pix_fmt', 'rgb24',
               '-r', '8',
               '-i', '-',
               '-c:v', 'mjpeg',
               '-q:v', '3',
               '-an',
               filename]
    pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
    pipe.stdin.write(video.tostring())

def log_tensorboard(image_gen, num, video_length, writer):
    @chainer.training.make_extension()
    def log(trainer):
        with chainer.using_config('train', False):
            updater = trainer.updater
            
            # generate samples
            h0 = Variable(updater.converter(image_gen.make_h0(num), updater.device))
            videos, _ = image_gen(h0)
            videos = chainer.cuda.to_cpu(videos.data) # (T, N, C, H, W)
            videos = videos / 2. + 0.5
            
            # make grid video, log only part of video frames.
            grid_video = to_grid(videos, int(np.sqrt(num))) # (T, C, H, W)
            
            ## image shape: (C, H, W), value range: [0, 1.0]
            for i in np.linspace(0, video_length, 4, endpoint=False, dtype=np.int):
                writer.add_image('{:02d}th frame'.format(i), grid_video[i], updater.epoch)
            
            # write videos as image
            for i in range(10):
                video = videos[:, i]
                video = to_sequence(video)
                writer.add_image('video_{:02d}'.format(i), video, updater.epoch)
            
    return log
