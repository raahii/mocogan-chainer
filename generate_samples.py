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
from visualize import to_grid, save_frames, save_video

def generate(image_gen, num):
    h0 = image_gen.make_h0(num)
    x = image_gen(h0)

    return x

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
    
    gen = ImageGenerator()
    serializers.load_npz(args.model_weight, gen)

    print(">>> generating...")
    videos = generate(gen, args.num) # (t, bs, c, w, h)
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
    
    videos = videos.transpose(1, 0, 3, 4, 2)
    # save each video
    for i, video in enumerate(videos):
        save_video(video, save_path/'{:03d}.mp4'.format(i),\
                   True, save_path/'{:03d}'.format(i))

if __name__=="__main__":
    main()
