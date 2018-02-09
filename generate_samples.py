import argparse
from tqdm import tqdm
import os
import numpy as np
from PIL import Image

import chainer
from chainer import serializers
from chainer import Variable
import chainer.functions as F

from model.net import ImageGenerator
from model.net import ConditionalImageGenerator
from model.net import InfoImageGenerator
from model.net import PSInfoImageGenerator
from visualize import to_grid, save_frames, save_video

def generate(image_gen, num, labels):
    h0 = image_gen.make_h0(num)
    x = image_gen(h0, labels=labels)

    return x

def main():
    parser = argparse.ArgumentParser(description='sample generation of videogan')
    parser.add_argument('model_snapshot')
    parser.add_argument('model_name')
    parser.add_argument('save_dir')
    parser.add_argument('--num', '-n', type=int, default=6)

    args = parser.parse_args()
    
    if args.model_name == "cgan" or args.model_name == "cwgan":
        gen = ConditionalImageGenerator()
    elif args.model_name == "infogan":
        gen = InfoImageGenerator()
    else:
        raise NotImplementedError
    serializers.load_npz(args.model_snapshot, gen)

    print(">>> generating...")
    labels = np.repeat(np.arange(6), args.num)
    videos = generate(gen, args.num**2, labels) # (t, bs, c, w, h)
    videos = videos[0].data
    videos = ((videos / 2. + 0.5) * 255).astype(np.uint8)
    
    print(">>> saving...")
    grid_video = to_grid(videos, args.num)
    os.makedirs(args.save_dir, exist_ok=True)
    save_frames(grid_video, args.save_dir)
    video_path = os.path.join(args.save_dir, 'video.mp4')
    save_video(grid_video, video_path)

if __name__=="__main__":
    main()
