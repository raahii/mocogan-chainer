import os
import sys

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from tb_chainer import name_scope, within_name_scope

def add_noise(x, sigma):
    xp = chainer.cuda.get_array_module(x.data)
    if chainer.config.train:
        return x + sigma * xp.random.randn(*x.shape)
    else:
        return x

# Normal model set
#{{{
class ImageGenerator(chainer.Chain):
    def __init__(self, channel=3, T=16, dim_zc=50, dim_zm=10):
        super(ImageGenerator, self).__init__()
        
        self.ch = channel
        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.T = T
        self.n_hidden = dim_zc + dim_zm

        with self.init_scope():
            n_hidden = self.n_hidden
            ndf = 64

            w = chainer.initializers.HeNormal()
            
            # Rm
            self.g0 = L.StatelessGRU(self.dim_zm, self.dim_zm, 0.2)
            
            # G
            self.dc1 = L.DeconvolutionND(2, n_hidden,   ndf*8, 4, stride=1, pad=0, initialW=w)
            self.dc2 = L.DeconvolutionND(2,    ndf*8,   ndf*4, 4, stride=2, pad=1, initialW=w)
            self.dc3 = L.DeconvolutionND(2,    ndf*4,   ndf*2, 4, stride=2, pad=1, initialW=w)
            self.dc4 = L.DeconvolutionND(2,    ndf*2,     ndf, 4, stride=2, pad=1, initialW=w)
            self.dc5 = L.DeconvolutionND(2,      ndf, channel, 4, stride=2, pad=1, initialW=w)

            self.bn1 = L.BatchNormalization(ndf*8)
            self.bn2 = L.BatchNormalization(ndf*4)
            self.bn3 = L.BatchNormalization(ndf*2)
            self.bn4 = L.BatchNormalization(ndf)

    def make_hidden(self, batchsize, size):
        return np.random.normal(0, 0.33, size=[batchsize, size]).astype(np.float32)

    def make_h0(self, batchsize):
        return self.make_hidden(batchsize, self.dim_zm)

    def make_zm(self, h0, batchsize):
        """ make zm vectors """
        xp = chainer.cuda.get_array_module(h0)

        ht = h0
        zm = Variable()
        for t in range(self.T):
            e = Variable(xp.asarray(self.make_hidden(batchsize, self.dim_zm)))
            zm_t = self.g0(ht, e)
            ht = zm_t # use zm_t as next hidden vector

            if t == 0:
                zm = F.reshape(zm_t, (1, batchsize, self.dim_zm))
            else:
                zm = F.concat((zm, F.reshape(zm_t, (1, batchsize, self.dim_zm))), axis=0)

        return zm

    @within_name_scope('image_gen')
    def __call__(self, h0, zc=None):
        """
        input h0 shape:  (batchsize, dim_zm)
        input zc shape:  (batchsize, dim_zc)
        output shape: (video_length, batchsize, channel, x, y, z)
        """
        batchsize = h0.shape[0]
        xp = chainer.cuda.get_array_module(h0)
        
        # make [zc, zm]
        # z shape: (video_length, batchsize, channel)
        if zc is None:
            zc = Variable(xp.asarray(self.make_hidden(batchsize, self.dim_zc)))
        zc = F.tile(zc, (self.T, 1, 1))
        zm = self.make_zm(h0, batchsize)
        z = F.concat((zc, zm), axis=2)
        z = F.reshape(z, (self.T*batchsize, self.n_hidden, 1, 1))
        
        # G([zc, zm])
        with name_scope('conv1', self.dc1.params()):
            x = F.relu(self.bn1(self.dc1(z)))
        with name_scope('conv2', self.dc2.params()):
            x = F.relu(self.bn2(self.dc2(x)))
        with name_scope('conv3', self.dc3.params()):
            x = F.relu(self.bn3(self.dc3(x)))
        with name_scope('conv4', self.dc4.params()):
            x = F.relu(self.bn4(self.dc4(x)))
        with name_scope('conv5', self.dc5.params()):
            x = F.tanh(self.dc5(x))
        x = F.reshape(x, (self.T, batchsize, self.ch, 64, 64))

        return x

class ImageDiscriminator(chainer.Chain):
    def __init__(self, channel=3, use_noise=False, noise_sigma=0.2):
        super(ImageDiscriminator, self).__init__()

        self.use_noise   = use_noise
        self.noise_sigma = noise_sigma
        self.ch          = channel

        with self.init_scope():
            w = chainer.initializers.HeNormal()

            ndf = 96

            self.dc1 = L.Convolution2D(channel,   ndf, 4, stride=2, pad=1, initialW=w)
            self.dc2 = L.Convolution2D(    ndf, ndf*2, 4, stride=2, pad=1, initialW=w)
            self.dc3 = L.Convolution2D(  ndf*2, ndf*4, 4, stride=2, pad=1, initialW=w)
            self.dc4 = L.Convolution2D(  ndf*4, ndf*8, 4, stride=2, pad=1, initialW=w)
            self.dc5 = L.Convolution2D(  ndf*8,     1, 4, stride=1, pad=0, initialW=w)

            self.bn2 = L.BatchNormalization(ndf*2)
            self.bn3 = L.BatchNormalization(ndf*4)
            self.bn4 = L.BatchNormalization(ndf*8)

    @within_name_scope('image_dis')
    def __call__(self, x):
        """
        input shape:  (batchsize, 3, 64, 64)
        output shape: (batchsize, 1)
        """
        if self.use_noise:
            x = add_noise(x, self.noise_sigma)

        with name_scope('conv1', self.dc1.params()):
            y = F.leaky_relu(self.dc1(x), slope=0.2)
        with name_scope('conv2', self.dc2.params()):
            y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)
        with name_scope('conv3', self.dc3.params()):
            y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
        with name_scope('conv4', self.dc4.params()):
            y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
        with name_scope('conv5', self.dc5.params()):
            y = self.dc5(y)

        return y

class VideoDiscriminator(chainer.Chain):
    def __init__(self, channel=3, use_noise=False, noise_sigma=0.2):
        super(VideoDiscriminator, self).__init__()

        self.use_noise   = use_noise
        self.noise_sigma = noise_sigma

        with self.init_scope():
            w = chainer.initializers.HeNormal()

            ndf = 64
            
            self.dc1 = L.ConvolutionND(3, channel,   ndf, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc2 = L.ConvolutionND(3,     ndf, ndf*2, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc3 = L.ConvolutionND(3,   ndf*2, ndf*4, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc4 = L.ConvolutionND(3,   ndf*4, ndf*8, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc5 = L.ConvolutionND(3,   ndf*8,     1, 4, stride=(1,3,3), pad=(0,0,0), initialW=w)

            self.bn2 = L.BatchNormalization(ndf*2)
            self.bn3 = L.BatchNormalization(ndf*4)
            self.bn4 = L.BatchNormalization(ndf*8)

    @within_name_scope('image_gen')
    def __call__(self, x):
        """
        input shape:  (batchsize, 1, 16, 64, 64)
        output shape: (batchsize, 1)
        """
        if self.use_noise:
            x = add_noise(x, self.noise_sigma)

        with name_scope('conv1', self.dc1.params()):
            y = F.leaky_relu(self.dc1(x), slope=0.2)
        with name_scope('conv2', self.dc2.params()):
            y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)
        with name_scope('conv3', self.dc3.params()):
            y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
        with name_scope('conv4', self.dc4.params()):
            y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
        with name_scope('conv5', self.dc5.params()):
            y = self.dc5(y)

        return y
#}}}

# Categorical model set
# {{{
class CategoricalImageGenerator(chainer.Chain):
    def __init__(self, channel=3, T=16, dim_zc=50, dim_zm=10, num_labels=6):
        super(CategoricalImageGenerator, self).__init__()
        
        self.ch = channel
        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.num_labels = num_labels
        self.T = T
        self.n_hidden = dim_zc + dim_zm + num_labels

        with self.init_scope():
            ndf = 64
            n_hidden = self.n_hidden

            w = chainer.initializers.HeNormal()
            
            # Rm
            self.g0 = L.StatelessGRU(self.dim_zm, self.dim_zm, 0.2)
            
            # G
            self.dc1 = L.DeconvolutionND(2, n_hidden,   ndf*8, 4, stride=1, pad=0, initialW=w)
            self.dc2 = L.DeconvolutionND(2,    ndf*8,   ndf*4, 4, stride=2, pad=1, initialW=w)
            self.dc3 = L.DeconvolutionND(2,    ndf*4,   ndf*2, 4, stride=2, pad=1, initialW=w)
            self.dc4 = L.DeconvolutionND(2,    ndf*2,     ndf, 4, stride=2, pad=1, initialW=w)
            self.dc5 = L.DeconvolutionND(2,      ndf, channel, 4, stride=2, pad=1, initialW=w)

            self.bn1 = L.BatchNormalization(ndf*8)
            self.bn2 = L.BatchNormalization(ndf*4)
            self.bn3 = L.BatchNormalization(ndf*2)
            self.bn4 = L.BatchNormalization(ndf)

    def make_hidden(self, batchsize, size):
        return np.random.normal(0, 0.33, size=[batchsize, size]).astype(np.float32)

    def make_h0(self, batchsize):
        return self.make_hidden(batchsize, self.dim_zm)

    def make_zc(self, batchsize):
        zc = self.make_hidden(batchsize, self.dim_zc)
        # extend video frame axis
        zc = np.tile(zc, (self.T, 1, 1))
        
        return zc

    def make_zl(self, batchsize, labels=None):
        """ make z_label """
        if labels is None:
            labels = np.random.randint(self.num_labels, size=batchsize)
        one_hot_labels = np.eye(self.num_labels)[labels].astype(np.float32)
        # extend video frame axis
        z_label = np.tile(one_hot_labels, (self.T, 1, 1))

        return z_label, labels

    def make_zm(self, h0, batchsize):
        """ make zm vectors """
        xp = chainer.cuda.get_array_module(h0)

        ht = h0
        zm = Variable()
        for t in range(self.T):
            e = Variable(xp.asarray(self.make_hidden(batchsize, self.dim_zm)))
            zm_t = self.g0(ht, e)
            ht = zm_t # use zm_t as next hidden vector

            if t == 0:
                zm = F.reshape(zm_t, (1, batchsize, self.dim_zm))
            else:
                zm = F.concat((zm, F.reshape(zm_t, (1, batchsize, self.dim_zm))), axis=0)

        return zm

    @within_name_scope('categ_image_gen')
    def __call__(self, h0, zc=None, labels=None):
        """
        input h0 shape:  (batchsize, dim_zm)
        input zc shape:  (batchsize, dim_zc)
        output shape: (video_length, batchsize, channel, x, y, z)
        """
        batchsize = h0.shape[0]
        xp = chainer.cuda.get_array_module(h0)
        
        # make [zc, zm, zl]
        # z shape: (video_length, batchsize, channel)
    
        ## z_content
        if zc is None:
            zc = Variable(xp.asarray(self.make_zc(batchsize)))

        ## z_motion
        zm = self.make_zm(h0, batchsize)

        ## z_label
        zl, labels = self.make_zl(batchsize, labels)
        zl = Variable(xp.asarray(zl))

        z = F.concat((zc, zl, zm), axis=2)
        z = F.reshape(z, (self.T*batchsize, self.n_hidden, 1, 1))
        
        # G([zc, zm, zl])
        x = F.relu(self.bn1(self.dc1(z)))
        x = F.relu(self.bn2(self.dc2(x)))
        x = F.relu(self.bn3(self.dc3(x)))
        x = F.relu(self.bn4(self.dc4(x)))
        x = F.tanh(self.dc5(x))
        x = F.reshape(x, (self.T, batchsize, self.ch, 64, 64))
        
        # concat label as additional feature map for discriminator
        label_video = -1.0 * xp.ones((self.T, batchsize, self.num_labels, 64, 64), dtype=np.float32)
        label_video[:,np.arange(batchsize), labels] = 1.
        x = F.concat((x, label_video), axis=2)

        return x, labels

class CategoricalImageDiscriminator(chainer.Chain):
    def __init__(self, channel=3, use_noise=False, noise_sigma=0.2):
        super(CategoricalImageDiscriminator, self).__init__()

        self.use_noise   = use_noise
        self.noise_sigma = noise_sigma
        self.ch          = channel

        with self.init_scope():
            w = chainer.initializers.HeNormal()

            ndf = 96

            self.dc1 = L.Convolution2D(channel,   ndf, 4, stride=2, pad=1, initialW=w)
            self.dc2 = L.Convolution2D(    ndf, ndf*2, 4, stride=2, pad=1, initialW=w)
            self.dc3 = L.Convolution2D(  ndf*2, ndf*4, 4, stride=2, pad=1, initialW=w)
            self.dc4 = L.Convolution2D(  ndf*4, ndf*8, 4, stride=2, pad=1, initialW=w)
            self.dc5 = L.Convolution2D(  ndf*8,     1, 4, stride=1, pad=0, initialW=w)

            self.bn2 = L.BatchNormalization(ndf*2)
            self.bn3 = L.BatchNormalization(ndf*4)
            self.bn4 = L.BatchNormalization(ndf*8)

    @within_name_scope('categ_image_dis')
    def __call__(self, x):
        """
        input shape:  (batchsize, 3, 64, 64)
        output shape: (batchsize, 1)
        """
        if self.use_noise:
            x = add_noise(x, self.noise_sigma)

        with name_scope('conv1', self.dc1.params()):
            y = F.leaky_relu(self.dc1(x), slope=0.2)
        with name_scope('conv2', self.dc2.params()):
            y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)
        with name_scope('conv3', self.dc3.params()):
            y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
        with name_scope('conv4', self.dc4.params()):
            y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
        with name_scope('conv5', self.dc5.params()):
            y = self.dc5(y)

        return y

class CategoricalVideoDiscriminator(chainer.Chain):
    def __init__(self, channel=3, num_labels=6, use_noise=False, noise_sigma=0.2):
        super(CategoricalVideoDiscriminator, self).__init__()

        self.use_noise   = use_noise
        self.noise_sigma = noise_sigma

        with self.init_scope():
            w = chainer.initializers.HeNormal()

            ndf = 64
            
            self.dc1 = L.ConvolutionND(3, channel+num_labels,   ndf, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc2 = L.ConvolutionND(3,     ndf, ndf*2, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc3 = L.ConvolutionND(3,   ndf*2, ndf*4, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc4 = L.ConvolutionND(3,   ndf*4, ndf*8, 4, stride=(1,2,2), pad=(0,1,1), initialW=w)
            self.dc5 = L.ConvolutionND(3,   ndf*8, num_labels+1, 4, stride=(1,3,3), pad=(0,0,0), initialW=w)

            self.bn2 = L.BatchNormalization(ndf*2)
            self.bn3 = L.BatchNormalization(ndf*4)
            self.bn4 = L.BatchNormalization(ndf*8)

    @within_name_scope('categ_video_dis')
    def __call__(self, x):
        """
        input shape:  (batchsize, ch, video_length, y, x)
        output shape: (batchsize, )
        """
        if self.use_noise:
            x = add_noise(x, self.noise_sigma)

        y = F.leaky_relu(self.dc1(x), slope=0.2)
        y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)
        y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
        y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
        y = self.dc5(y)

        return y
# }}}

if __name__ ==  "__main__":
    main()
