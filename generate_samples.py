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
from visualize import write_grid_videos

def generate(image_gen, num):
    h0 = image_gen.make_h0(num)
    x = image_gen(h0)

    return x

def main():
    parser = argparse.ArgumentParser(description='sample generation of videogan')
    parser.add_argument('gen_model')
    parser.add_argument('save_dir')
    parser.add_argument('--gen_num', '-n', type=int, default=100)

    args = parser.parse_args()
    
    gen = ImageGenerator()
    serializers.load_npz(args.gen_model, gen)

    print("generating...")
    videos = generate(gen, args.gen_num)

    print("saving...")
    os.makedirs(args.save_dir, exist_ok=True)
    write_grid_videos(videos.data, args.save_dir)

if __name__=="__main__":
    main()
