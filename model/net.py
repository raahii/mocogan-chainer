import os
import sys

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable

def add_noise(x, use_noise, sigma):
    xp = chainer.cuda.get_array_module(x.data)
    if chainer.config.train and use_noise:
        return x + sigma * xp.random.randn(*x.data.shape)
    else:
        return x

class ImageGenerator(chainer.Chain):
    def __init__(self, dim_zc=50, dim_zm=10, dim_zl=0, out_channels=3, \
                       n_filters=64, video_len=16):
        super(ImageGenerator, self).__init__()
        
        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.dim_zl = dim_zl
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.video_len = video_len

        n_hidden = dim_zc + dim_zm + dim_zl
        self.n_hidden = n_hidden
        self.use_label = dim_zl != 0

        with self.init_scope():
            w = chainer.initializers.GlorotNormal()
            
            # Rm
            self.g0 = L.StatelessGRU(self.dim_zm+self.dim_zl, self.dim_zm+self.dim_zl)
            
            # G
            self.dc1 = L.DeconvolutionND(2,    n_hidden,  n_filters*8, 4, stride=1, pad=0, initialW=w)
            self.dc2 = L.DeconvolutionND(2, n_filters*8,  n_filters*4, 4, stride=2, pad=1, initialW=w)
            self.dc3 = L.DeconvolutionND(2, n_filters*4,  n_filters*2, 4, stride=2, pad=1, initialW=w)
            self.dc4 = L.DeconvolutionND(2, n_filters*2,  n_filters,   4, stride=2, pad=1, initialW=w)
            self.dc5 = L.DeconvolutionND(2, n_filters  , out_channels, 4, stride=2, pad=1, initialW=w)

            self.bn1 = L.BatchNormalization(n_filters*8)
            self.bn2 = L.BatchNormalization(n_filters*4)
            self.bn3 = L.BatchNormalization(n_filters*2)
            self.bn4 = L.BatchNormalization(n_filters)

    def make_hidden(self, batchsize, size):
        return np.random.normal(0, 0.33, size=[batchsize, size]).astype(np.float32)

    def make_h0(self, batchsize):
        return self.make_hidden(batchsize, self.dim_zm+self.dim_zl)

    def to_one_hot(self, zl, xp):
        return xp.eye(self.dim_zl, dtype=np.float32)[zl]

    def make_zm(self, h0, zl):
        """ make zm vectors """

        batchsize = h0.shape[0]
        xp = chainer.cuda.get_array_module(h0)

        assert self.use_label == (zl is not None)

        ht = [h0]
        for t in range(self.video_len):
            et = Variable(xp.asarray(self.make_hidden(batchsize, self.dim_zm)))
            
            if self.use_label:
                et = F.concat((zl, et))

            ht.append(self.g0(ht[-1], et))
        
        zmt = [F.reshape(hk, (1, batchsize, self.dim_zm+self.dim_zl)) for hk in ht[1:]]
        zm = F.concat(zmt, axis=0)

        return zm

    def __call__(self, h0):
        """
        input h0 shape:  (batchsize, dim_zm)
        input zc shape:  (batchsize, dim_zc)
        output shape: (video_length, batchsize, channel, x, y)
        """
        batchsize = h0.shape[0]
        xp = chainer.cuda.get_array_module(h0)
        
        # make zl
        if self.use_label:
            labels = xp.random.randint(self.dim_zl, size=batchsize)
            zl = Variable(self.to_one_hot(labels, xp))
        else:
            labels = None
            zl = None

        # make zm
        zm = self.make_zm(h0, zl)
        
        # make zc
        zc = Variable(xp.asarray(self.make_hidden(batchsize, self.dim_zc)))
        zc = F.tile(zc, (self.video_len, 1, 1))
        
        # [zc, zm]
        z = F.concat((zc, zm), axis=2)
        z = F.reshape(z, (self.video_len*batchsize, self.n_hidden, 1, 1))
        
        # G(z)
        x = F.relu(self.bn1(self.dc1(z)))
        x = F.relu(self.bn2(self.dc2(x)))
        x = F.relu(self.bn3(self.dc3(x)))
        x = F.relu(self.bn4(self.dc4(x)))
        x = F.tanh(self.dc5(x))
        x = F.reshape(x, (self.video_len, batchsize, self.out_channels, 64, 64))

        return x, labels

class ImageDiscriminator(chainer.Chain):
    def __init__(self, in_channels=3, out_channels=1, n_filters=64, use_noise=False, noise_sigma=0.2):
        super(ImageDiscriminator, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_filters    = n_filters
        self.use_noise    = use_noise
        self.noise_sigma  = noise_sigma

        with self.init_scope():
            w = chainer.initializers.GlorotNormal()

            self.dc1 = L.Convolution2D(in_channels,  n_filters  , 4, stride=2, pad=1, initialW=w)
            self.dc2 = L.Convolution2D(n_filters  ,  n_filters*2, 4, stride=2, pad=1, initialW=w)
            self.dc3 = L.Convolution2D(n_filters*2,  n_filters*4, 4, stride=2, pad=1, initialW=w)
            self.dc4 = L.Convolution2D(n_filters*4,  n_filters*8, 4, stride=2, pad=1, initialW=w)
            self.dc5 = L.Convolution2D(n_filters*8, out_channels, 4, stride=1, pad=0, initialW=w)

            self.bn2 = L.BatchNormalization(n_filters*2)
            self.bn3 = L.BatchNormalization(n_filters*4)
            self.bn4 = L.BatchNormalization(n_filters*8)

    def __call__(self, x):
        """
        input shape:  (batchsize, 3, 64, 64)
        output shape: (batchsize, 1)
        """
        y = add_noise(x, self.use_noise, self.noise_sigma)
        y = F.leaky_relu(self.dc1(y), slope=0.2)
        y = add_noise(y, self.use_noise, self.noise_sigma)
        y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)
        y = add_noise(y, self.use_noise, self.noise_sigma)
        y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
        y = add_noise(y, self.use_noise, self.noise_sigma)
        y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
        y = self.dc5(y)

        return y

class VideoDiscriminator(chainer.Chain):
    def __init__(self, in_channels=3, out_channels=1, n_filters=64, use_noise=False, noise_sigma=0.2):
        super(VideoDiscriminator, self).__init__()
        
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_filters    = n_filters
        self.use_noise    = use_noise
        self.noise_sigma  = noise_sigma

        with self.init_scope():
            w = chainer.initializers.GlorotNormal()

            self.dc1 = L.ConvolutionND(3, in_channels,  n_filters  , 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc2 = L.ConvolutionND(3, n_filters  ,  n_filters*2, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc3 = L.ConvolutionND(3, n_filters*2,  n_filters*4, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc4 = L.ConvolutionND(3, n_filters*4,  n_filters*8, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc5 = L.ConvolutionND(3, n_filters*8, out_channels, 4, stride=(1,3,3), pad=(0,0,0), initialW=w)

            self.bn2 = L.BatchNormalization(n_filters*2)
            self.bn3 = L.BatchNormalization(n_filters*4)
            self.bn4 = L.BatchNormalization(n_filters*8)

    def __call__(self, x):
        """
        input shape:  (batchsize, 1, 16, 64, 64)
        output shape: (batchsize, 1)
        """
        y = add_noise(x, self.use_noise, self.noise_sigma)
        y = F.leaky_relu(self.dc1(y), slope=0.2)
        y = add_noise(y, self.use_noise, self.noise_sigma)
        y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)
        y = add_noise(y, self.use_noise, self.noise_sigma)
        y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
        y = add_noise(y, self.use_noise, self.noise_sigma)
        y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
        y = self.dc5(y)

        return y

if __name__ ==  "__main__":
    main()
