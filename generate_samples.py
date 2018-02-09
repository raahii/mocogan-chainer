import argparse
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import pickle

import chainer
from chainer import serializers
from chainer import Variable
import chainer.functions as F

from model.net import ImageGenerator
from util import to_grid, save_frames, save_video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_weight')
    parser.add_argument('save_path')
    parser.add_argument('--num', '-n', type=int, default=36)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    args = parser.parse_args()
    
    # check num
    if np.sqrt(args.num) % 1.0 != 0:
        raise ValueError('--num must be n^2 (n: natural number).')
    n = int(np.sqrt(args.num))

    # gpu or cpu
    xp = np if args.gpu == -1 else chainer.cuda.cupy
    
    gen = ImageGenerator()
    serializers.load_npz(args.model_weight, gen)

    print(">>> generating...")
    videos = gen(args.num, xp) # (t, bs, c, w, h)
    videos = videos[0].data
    videos = ((videos / 2. + 0.5) * 255).astype(np.uint8)
    
    print(">>> saving...")
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # save grid video
    grid_video = to_grid(videos, n)
    grid_video = grid_video.transpose(0, 2, 3, 1)
    save_video(grid_video, save_path/'grid.mp4', \
               True, save_path/'grid')
    
    # save each video
    videos = videos.transpose(1, 0, 3, 4, 2)
    for i, video in enumerate(videos):
        save_video(video, save_path/'{:03d}.mp4'.format(i),\
                   True, save_path/'{:03d}'.format(i))

if __name__=="__main__":
    main()
