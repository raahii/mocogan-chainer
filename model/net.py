import os
import sys

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import GlorotNormal

class ImageDiscriminator(chainer.Chain):
    def __init__(self):
        super(ImageDiscriminator, self).__init__()

        with self.init_scope():
            w = GlorotNormal()
            ch = 64

            self.dc1 = L.ConvolutionND(2,    3,   ch, 4, stride=2, pad=1, initialW=w)
            self.dc2 = L.ConvolutionND(2,   ch, ch*2, 4, stride=2, pad=1, initialW=w)
            self.dc3 = L.ConvolutionND(2, ch*2, ch*4, 4, stride=2, pad=1, initialW=w)
            self.dc4 = L.ConvolutionND(2, ch*4, ch*8, 4, stride=2, pad=1, initialW=w)
            self.dc5 = L.ConvolutionND(2, ch*8,    1, 6, stride=1, pad=0, initialW=w)

            self.bn2 = L.BatchNormalization(ch*2)
            self.bn3 = L.BatchNormalization(ch*4)
            self.bn4 = L.BatchNormalization(ch*8)

    def __call__(self, x):
        y = F.leaky_relu(self.dc1(x), slope=0.2)
        y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)
        y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
        y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
        y = F.sigmoid(self.dc5(y))

        return y

class VideoDiscriminator(chainer.Chain):
    def __init__(self):
        super(VideoDiscriminator, self).__init__()

        with self.init_scope():
            w = GlorotNormal()
            ch = 64

            self.dc1 = L.ConvolutionND(3,    3,   ch, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc2 = L.ConvolutionND(3,   ch, ch*2, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc3 = L.ConvolutionND(3, ch*2, ch*4, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc4 = L.ConvolutionND(3, ch*4, ch*8, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc5 = L.ConvolutionND(3, ch*8,    1, 4, stride=(1,3,3), pad=(0,0,0), initialW=w)

            self.bn2 = L.BatchNormalization(ch*2)
            self.bn3 = L.BatchNormalization(ch*4)
            self.bn4 = L.BatchNormalization(ch*8)

    def __call__(self, x):
        y = F.leaky_relu(self.dc1(x), slope=0.2)
        y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)
        y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
        y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
        y = self.dc5(y)

        return y

class Generator(chainer.Chain):
    def __init__(self, n_hidden=60):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden

        with self.init_scope():
            w = GlorotNormal()
            ch = 64
            n_hidden = self.n_hidden

            self.dc1 = L.DeconvolutionND(2, n_hidden, ch*8, 6, stride=1, pad=0, initialW=w)
            self.dc2 = L.DeconvolutionND(2,     ch*8, ch*4, 4, stride=2, pad=1, initialW=w)
            self.dc3 = L.DeconvolutionND(2,     ch*4, ch*2, 4, stride=2, pad=1, initialW=w)
            self.dc4 = L.DeconvolutionND(2,     ch*2,   ch, 4, stride=2, pad=1, initialW=w)
            self.dc5 = L.DeconvolutionND(2,       ch,    3, 4, stride=2, pad=1, initialW=w)

            self.bn1 = L.BatchNormalization(ch*8)
            self.bn2 = L.BatchNormalization(ch*4)
            self.bn3 = L.BatchNormalization(ch*2)
            self.bn4 = L.BatchNormalization(ch)

    def __call__(self, z):
        x = F.relu(self.bn1(self.dc1(z)))
        x = F.relu(self.bn2(self.dc2(x)))
        x = F.relu(self.bn3(self.dc3(x)))
        x = F.relu(self.bn4(self.dc4(x)))
        x = F.tanh(self.dc5(x))

        return x

class GRU(chainer.Chain):
    def __init__(self, T=16, n_zc=50, n_zm=10):
        super(GRU, self).__init__()
        
        self.T = T
        self.n_zc = n_zc
        self.n_zm = n_zm

        with self.init_scope():
            self.g1 = L.StatelessGRU(self.n_zm, self.n_zm, 0.2)

    def __call__(self, h0, e):
        z = self.g1(h0, e)

        return z

    def make_zc(self, batchsize):
        return np.random.normal(0, 0.33, size=[batchsize, self.n_zc]).astype(np.float32)

    def make_h0(self, batchsize):
        return np.random.normal(0, 0.33, size=[batchsize, self.n_zm]).astype(np.float32)

    def make_zm(self, batchsize):
        return np.random.normal(0, 0.33, size=[batchsize, self.n_zm]).astype(np.float32)

def count_model_params(m):
    return sum(p.data.size for p in m.params())

if __name__ ==  "__main__":
    print(
        count_model_params(ImageDiscriminator()),
        count_model_params(VideoDiscriminator()),
        count_model_params(Generator()),
        count_model_params(GRU()),
    )
