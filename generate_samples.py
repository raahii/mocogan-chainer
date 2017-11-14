import argparse
import os, sys, glob
import chainer
from chainer import serializers

from model.net import Generator
from model.net import GRU
from visualize import write_grid_videos

def generate(gen, num):
    z = gen.make_hidden(num)
    x = gen(z)
    x = x.transpose(0,2,3,4,1)
    x = x / 2. + 0.5

    return x

def main():
    parser = argparse.ArgumentParser(description='sample generation of videogan')
    parser.add_argument('--gru_model', '-gru', required=True)
    parser.add_argument('--gen_model', '-gen', required=True)
    parser.add_argument('--save_path', '-p', required=True)
    parser.add_argument('--gen_num', '-n', type=int, default=25)
    parser.add_argument('--ext', '-e', default='mp4')

    args = parser.parse_args()
    
    gru, gen = GRU(), Generator()
    serializers.load_npz(args.gru_model, gru)
    serializers.load_npz(args.gen_model, gen)

    print("generating...")
    zc = Variable(xp.asarray(gru.make_zc(args.gen_num)))
    h0 = Variable(xp.asarray(gru.make_h0(args.gen_num)))

    for i in range(self.T):
        e = Variable(np.asarray(gru.make_zm(args.gen_num)))
        zm = gru(h0, e)

        z = F.concat([zc, zm], axis=1)

        x_fake_t = gen(z)
        x_fake_t = x_frame_t.reshape((1, batchsize, 3, self.img_size, self.img_size))

        if x_fake.data is None:
            x_fake = x_fake_t
        else:
            x_fake = F.concat([x_fake, x_fake_t], axis=0)
        
    x_fake = x_fake.transpose(1, 2, 0, 3, 4)
    
    print("saving...")
    write_grid_videos(x_fake.data, args.save_path, 'gif')

if __name__=="__main__":
    main()
