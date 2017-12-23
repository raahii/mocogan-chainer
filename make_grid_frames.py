import argparse
from tqdm import tqdm
import os, sys, glob
import numpy as np
from PIL import Image

import chainer
from chainer import serializers
from chainer import Variable
import chainer.functions as F

from model.net import Generator
from model.net import GRU

T = 16
img_size = 64

def generate(gru, gen, num):
    num = num*num
    zc = Variable(np.asarray(gru.make_zc(num)))
    h0 = Variable(np.asarray(gru.make_h0(num)))

    x_fake = Variable()
    for i in range(T):
        e = Variable(np.asarray(gru.make_zm(num)))
        zm = gru(h0, e)

        z = F.concat([zc, zm], axis=1)

        x_fake_t = gen(z)
        x_fake_t = x_fake_t.reshape((1, num, 3, img_size, img_size))

        if x_fake.data is None:
            x_fake = x_fake_t
        else:
            x_fake = F.concat([x_fake, x_fake_t], axis=0)
        
    x_fake = x_fake.transpose(1, 0, 3, 4, 2)
    x_fake = ((x_fake / 2. + 0.5) * 255).data.astype(np.uint8)

    return x_fake

def main():
    parser = argparse.ArgumentParser(description='sample generation of videogan')
    parser.add_argument('--gru_model', required=True)
    parser.add_argument('--gen_model', required=True)
    parser.add_argument('--save_dir', '-p', required=True)
    parser.add_argument('--num', '-n', type=int, default=5)

    args = parser.parse_args()
    
    gen = Generator()
    gru = GRU()
    serializers.load_npz(args.gen_model, gen)
    serializers.load_npz(args.gru_model, gru)
    
    print("generating...")
    videos = generate(gru, gen, args.num)
    print("saving...")

    # グリッドビデオを書き出す
    os.makedirs(args.save_dir, exist_ok=True)
    sq = args.num
    for k in tqdm(range(T)):
        for i in range(sq):
            row = videos[sq*i, k]

            for j in range(1, sq):
                row = np.concatenate([row, videos[sq*i+j, k]], axis=1)

            if i == 0:
                grid = row
            else:
                grid = np.concatenate([grid, row], axis=0)
        
        grid_img = Image.fromarray(grid)
        fname = os.path.join(args.save_dir, "{:02d}.jpg".format(k+1))
        grid_img.save(fname, 'JPEG', quality=100, optimize=True)

if __name__=="__main__":
    main()
