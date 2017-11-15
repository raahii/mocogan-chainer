import os
import sys

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable

def add_noise(x, sigma):
    xp = chainer.cuda.get_array_module(x.data)
    if chainer.config.train:
        return x + sigma * xp.random.randn(*x.shape)
    else:
        return x

class ImageDiscriminator(chainer.Chain):
    def __init__(self, channel=3, use_noise=False, noise_sigma=0.2):
        super(ImageDiscriminator, self).__init__()

        self.use_noise   = use_noise
        self.noise_sigma = noise_sigma
        ch = channel
        ndf = 64

        with self.init_scope():
            w = chainer.initializers.GlorotNormal()

            self.dc1 = L.ConvolutionND(2,    ch,   ndf, 4, stride=2, pad=1, initialW=w)
            self.dc2 = L.ConvolutionND(2,   ndf, ndf*2, 4, stride=2, pad=1, initialW=w)
            self.dc3 = L.ConvolutionND(2, ndf*2, ndf*4, 4, stride=2, pad=1, initialW=w)
            self.dc4 = L.ConvolutionND(2, ndf*4, ndf*8, 4, stride=2, pad=1, initialW=w)
            self.dc5 = L.ConvolutionND(2, ndf*8,     1, 4, stride=1, pad=0, initialW=w)

            self.bn2 = L.BatchNormalization(ndf*2)
            self.bn3 = L.BatchNormalization(ndf*4)
            self.bn4 = L.BatchNormalization(ndf*8)

    def __call__(self, x):
        if self.use_noise:
            x = add_noise(x, self.noise_sigma)
        
        y = F.leaky_relu(self.dc1(x), slope=0.2)
        y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)
        y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
        y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
        y = F.sigmoid(self.dc5(y))

        return y

class VideoDiscriminator(chainer.Chain):
    def __init__(self, channel=3, use_noise=False, noise_sigma=0.2):
        super(VideoDiscriminator, self).__init__()

        self.use_noise   = use_noise
        self.noise_sigma = noise_sigma
        ch = channel
        ndf = 64

        with self.init_scope():
            w = chainer.initializers.GlorotNormal()

            self.dc1 = L.ConvolutionND(3,    ch,   ndf, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc2 = L.ConvolutionND(3,   ndf, ndf*2, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc3 = L.ConvolutionND(3, ndf*2, ndf*4, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc4 = L.ConvolutionND(3, ndf*4, ndf*8, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc5 = L.ConvolutionND(3, ndf*8,     1, 4, stride=(1,3,3), pad=(0,0,0), initialW=w)

            self.bn2 = L.BatchNormalization(ndf*2)
            self.bn3 = L.BatchNormalization(ndf*4)
            self.bn4 = L.BatchNormalization(ndf*8)

    def __call__(self, x):
        if self.use_noise:
            x = add_noise(x, self.noise_sigma)

        y = F.leaky_relu(self.dc1(x), slope=0.2)
        y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)
        y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
        y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
        y = self.dc5(y)

        return y

class ImageGenerator(chainer.Chain):
    def __init__(self, channel=3, T=16, dim_zc=50, dim_zm=10):
        super(ImageGenerator, self).__init__()
        
        self.ch = channel
        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.T = T
        self.n_hidden = dim_zc + dim_zm
        ndf = 64

        with self.init_scope():
            n_hidden = self.n_hidden
            ch = self.ch

            w = chainer.initializers.GlorotNormal()
            
            # Rm
            self.g0 = L.StatelessGRU(self.dim_zm, self.dim_zm, 0.2)
            
            # G
            self.dc1 = L.DeconvolutionND(2, n_hidden, ndf*8, 4, stride=1, pad=0, initialW=w)
            self.dc2 = L.DeconvolutionND(2,    ndf*8, ndf*4, 4, stride=2, pad=1, initialW=w)
            self.dc3 = L.DeconvolutionND(2,    ndf*4, ndf*2, 4, stride=2, pad=1, initialW=w)
            self.dc4 = L.DeconvolutionND(2,    ndf*2,   ndf, 4, stride=2, pad=1, initialW=w)
            self.dc5 = L.DeconvolutionND(2,      ndf,    ch, 4, stride=2, pad=1, initialW=w)

            self.bn1 = L.BatchNormalization(ndf*8)
            self.bn2 = L.BatchNormalization(ndf*4)
            self.bn3 = L.BatchNormalization(ndf*2)
            self.bn4 = L.BatchNormalization(ndf)

    def make_hidden(self, batchsize, size):
        return np.random.normal(0, 0.33, size=[batchsize, size]).astype(np.float32)

    def make_zm(self, h0, batchsize):
        """ make zm vectors """
        xp = chainer.cuda.get_array_module(h0)

        ht = h0
        zm = Variable()
        for i in range(self.T):
            e = Variable(xp.asarray(self.make_hidden(batchsize, self.dim_zm)))
            zm_t = self.g0(ht, e)
            ht = zm_t # use zm_t as next hidden vector

            if i == 0:
                zm = F.reshape(zm_t, (1, batchsize, self.dim_zm))
            else:
                zm = F.concat([zm, F.reshape(zm_t, (1, batchsize, self.dim_zm))], axis=0)

        return zm

    def __call__(self, h0, zc=None):
        batchsize = h0.shape[0]
        
        # make [zc, zm]
        if zc is None:
            zc = self.make_hidden(batchsize, self.dim_zc)
        zc = F.tile(zc, (self.T, 1, 1))
        zm = self.make_zm(h0, batchsize)
        z = F.concat([zc, zm], axis=2)
        z = z.reshape((batchsize*self.T, self.n_hidden, 1, 1))
        
        # G([zc, zm])
        x = F.relu(self.bn1(self.dc1(z)))
        x = F.relu(self.bn2(self.dc2(x)))
        x = F.relu(self.bn3(self.dc3(x)))
        x = F.relu(self.bn4(self.dc4(x)))
        x = F.tanh(self.dc5(x))
        x = x.reshape((batchsize, self.T, self.ch, 64, 64))

        return x

def count_model_params(m):
    return sum(p.data.size for p in m.params())

if __name__ ==  "__main__":
    print(
        count_model_params(ImageDiscriminator()),
        count_model_params(VideoDiscriminator()),
        count_model_params(ImageGenerator()),
    )
