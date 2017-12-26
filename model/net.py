import os
import sys

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from tb_chainer import name_scope, within_name_scope

def add_noise(x, use_noise, sigma):
    xp = chainer.cuda.get_array_module(x.data)
    if chainer.config.train and use_noise:
        return x + sigma * xp.random.randn(*x.data.shape)
    else:
        return x

class pixel_shuffler():
    def __init__(self, r):
        self.r = r

    def __call__(self, x):
        r = self.r
        batchsize = x.shape[0]
        in_channels = x.shape[1]
        out_channels = in_channels // (r ** 2)
        in_height = x.shape[2]
        in_width = x.shape[3]
        out_height = in_height * r
        out_width = in_width * r
        x = F.reshape(x, (batchsize, r, r, out_channels, in_height, in_width))
        x = F.transpose(x, (0, 3, 4, 1, 5, 2))
        x = F.reshape(x, (batchsize, out_channels, out_height, out_width))

        return x

# Normal model set
#{{{
class ImageGenerator(chainer.Chain):
    def __init__(self, num_channels=3, video_len=16, dim_zc=50, dim_zm=10):
        super(ImageGenerator, self).__init__()
        
        self.ch = num_channels
        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.video_len = video_len
        self.n_hidden = dim_zc + dim_zm

        with self.init_scope():
            n_hidden = self.n_hidden
            ndf = 64

            w = chainer.initializers.GlorotNormal()
            
            # Rm
            self.g0 = L.StatelessGRU(self.dim_zm, self.dim_zm)
            
            # G
            self.dc1 = L.DeconvolutionND(2, n_hidden,   ndf*8, 4, stride=1, pad=0, initialW=w)
            self.dc2 = L.DeconvolutionND(2,    ndf*8,   ndf*4, 4, stride=2, pad=1, initialW=w)
            self.dc3 = L.DeconvolutionND(2,    ndf*4,   ndf*2, 4, stride=2, pad=1, initialW=w)
            self.dc4 = L.DeconvolutionND(2,    ndf*2,     ndf, 4, stride=2, pad=1, initialW=w)
            self.dc5 = L.DeconvolutionND(2,      ndf, num_channels, 4, stride=2, pad=1, initialW=w)

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

        ht = [h0]
        for t in range(self.video_len):
            et = Variable(xp.asarray(self.make_hidden(batchsize, self.dim_zm)))
            ht.append(self.g0(ht[-1], et))
        
        zmt = [hk.reshape(1, batchsize, self.dim_zm) for hk in ht]
        zm = F.concat(zmt[1:], axis=0)

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
        zc = F.tile(zc, (self.video_len, 1, 1))
        zm = self.make_zm(h0, batchsize)
        z = F.concat((zc, zm), axis=2)
        z = F.reshape(z, (self.video_len*batchsize, self.n_hidden, 1, 1))
        
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
        x = F.reshape(x, (self.video_len, batchsize, self.ch, 64, 64))

        return x

class ImageDiscriminator(chainer.Chain):
    def __init__(self, channel=3, use_noise=False, noise_sigma=0.2):
        super(ImageDiscriminator, self).__init__()

        self.use_noise   = use_noise
        self.noise_sigma = noise_sigma
        self.ch          = channel

        with self.init_scope():
            w = chainer.initializers.GlortNormal()

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
            w = chainer.initializers.GlorotNormal()

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

# Base Categorical model set
# {{{
class CategoricalImageGenerator(chainer.Chain):
    def __init__(self, out_channels=3, n_filters=64, \
                 video_len=16, dim_zc=50, dim_zm=10, dim_zl=6):
        super(CategoricalImageGenerator, self).__init__()
        
        self.out_ch = out_channels
        self.video_len = video_len
        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.dim_zl = dim_zl
        self.n_hidden = dim_zc + dim_zm + dim_zl

        with self.init_scope():
            n_hidden = self.n_hidden

            # w = chainer.initializers.GlorotNormal()
            
            # Rm
            self.g0 = L.StatelessGRU(self.dim_zm, self.dim_zm)
            
            # G
            k = 4

            w = chainer.initializers.Uniform(1./(n_hidden*k**2))
            self.dc1 = L.Deconvolution2D(   n_hidden,  n_filters*8, k, stride=1, pad=0, nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*8*k**2))
            self.dc2 = L.Deconvolution2D(n_filters*8,  n_filters*4, k, stride=2, pad=1, nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*4*k**2))
            self.dc3 = L.Deconvolution2D(n_filters*4,  n_filters*2, k, stride=2, pad=1, nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*2*k**2))
            self.dc4 = L.Deconvolution2D(n_filters*2,    n_filters, k, stride=2, pad=1, nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*k**2))
            self.dc5 = L.Deconvolution2D(  n_filters, out_channels, k, stride=2, pad=1, nobias=True, initialW=w)

            self.bn1 = L.BatchNormalization(n_filters*8)
            self.bn2 = L.BatchNormalization(n_filters*4)
            self.bn3 = L.BatchNormalization(n_filters*2)
            self.bn4 = L.BatchNormalization(  n_filters)

    def make_hidden(self, batchsize, size):
        return np.random.normal(0, 0.33, size=[batchsize, size]).astype(np.float32)

    def make_h0(self, batchsize):
        return self.make_hidden(batchsize, self.dim_zm)

    def make_zc(self, batchsize):
        zc = self.make_hidden(batchsize, self.dim_zc)
        # extend video frame axis
        zc = np.tile(zc, (self.video_len, 1, 1))

        return zc

    def make_zl(self, batchsize, labels=None):
        """ make z_label """
        if labels is None:
            labels = np.random.randint(self.dim_zl, size=batchsize)
        one_hot_labels = np.eye(self.dim_zl)[labels].astype(np.float32)
        # extend video frame axis
        z_label = np.tile(one_hot_labels, (self.video_len, 1, 1))

        return z_label, labels

    def make_zm(self, h0, batchsize):
        """ make zm vectors """
        xp = chainer.cuda.get_array_module(h0)

        ht = [h0]
        for t in range(self.video_len):
            et = Variable(xp.asarray(self.make_hidden(batchsize, self.dim_zm)))
            ht.append(self.g0(ht[-1], et))
        
        zmt = [hk.reshape(1, batchsize, self.dim_zm) for hk in ht]
        zm = F.concat(zmt[1:], axis=0)

        return zm

class CategoricalImageDiscriminator(chainer.Chain):
    def __init__(self, in_channels=3, out_channels=1, n_filters=64, \
                 use_noise=False, noise_sigma=0.1):
        super(CategoricalImageDiscriminator, self).__init__()
        
        self.use_noise   = use_noise
        self.noise_sigma = noise_sigma

        with self.init_scope():
            k = 4

            w = chainer.initializers.Uniform(1./(in_channels*k**2))
            self.dc1 = L.Convolution2D(in_channels,    n_filters, k, stride=2, pad=1, nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*k**2))
            self.dc2 = L.Convolution2D(  n_filters,  n_filters*2, k, stride=2, pad=1, nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*2*k**2))
            self.dc3 = L.Convolution2D(n_filters*2,  n_filters*4, k, stride=2, pad=1, nobias=True, initialW=w)
            # w = chainer.initializers.Uniform(1./(n_filters*4*k**2))
            # self.dc4 = L.Convolution2D(n_filters*4,            1, k, stride=2, pad=1, nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*4*k**2))
            self.dc4 = L.Convolution2D(n_filters*4,  n_filters*8, 4, stride=2, pad=1, nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*8*k**2))
            self.dc5 = L.Convolution2D(n_filters*8, out_channels, 4, stride=1, pad=0, nobias=True, initialW=w)

            self.bn2 = L.BatchNormalization(n_filters*2)
            self.bn3 = L.BatchNormalization(n_filters*4)
            self.bn4 = L.BatchNormalization(n_filters*8)

class CategoricalVideoDiscriminator(chainer.Chain):
    def __init__(self, in_channels=3, out_channels=1, n_filters=64, \
                 use_noise=False, noise_sigma=0.1):
        super(CategoricalVideoDiscriminator, self).__init__()


        self.use_noise   = use_noise
        self.noise_sigma = noise_sigma

        with self.init_scope():
            k = 4

            w = chainer.initializers.Uniform(1./(in_channels*k**3))
            self.dc1 = L.ConvolutionND(3, in_channels,    n_filters, k, stride=(1,2,2), pad=(0,1,1), nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*k**3))
            self.dc2 = L.ConvolutionND(3,   n_filters,  n_filters*2, k, stride=(1,2,2), pad=(0,1,1), nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*2*k**3))
            self.dc3 = L.ConvolutionND(3, n_filters*2,  n_filters*4, k, stride=(1,2,2), pad=(0,1,1), nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*4*k**3))
            self.dc4 = L.ConvolutionND(3, n_filters*4,  n_filters*8, k, stride=(1,2,2), pad=(0,1,1), nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*8*k**3))
            self.dc5 = L.ConvolutionND(3, n_filters*8, out_channels, k, stride=(1,1,1), pad=(0,0,0), nobias=True, initialW=w)

            self.bn2 = L.BatchNormalization(n_filters*2)
            self.bn3 = L.BatchNormalization(n_filters*4)
            self.bn4 = L.BatchNormalization(n_filters*8)
# }}}

# cGAN model set
#{{{
class ConditionalImageGenerator(CategoricalImageGenerator):
    def __init__(self, *args, **kwargs):
        super(ConditionalImageGenerator, self).__init__(*args, **kwargs)

    @within_name_scope('conditional_igen')
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

        z = F.concat((zc, zm, zl), axis=2)
        z = F.reshape(z, (self.video_len*batchsize, self.n_hidden, 1, 1))
        
        # G([zc, zm, zl])
        x = F.relu(self.bn1(self.dc1(z)))
        x = F.relu(self.bn2(self.dc2(x)))
        x = F.relu(self.bn3(self.dc3(x)))
        x = F.relu(self.bn4(self.dc4(x)))
        x = F.tanh(self.dc5(x))
        x = F.reshape(x, (self.video_len, batchsize, self.out_ch, 64, 64))
        
        # concat label info as additional feature maps
        label_video = -1.0 * xp.ones((self.video_len, batchsize, self.dim_zl, 64, 64), dtype=np.float32)
        label_video[:,np.arange(batchsize), labels] = 1.
        x = F.concat((x, label_video), axis=2)

        return x, labels

class ConditionalImageDiscriminator(CategoricalImageDiscriminator):
    def __init__(self, *args, **kwargs):
        super(ConditionalImageDiscriminator, self).__init__(*args, **kwargs)

    @within_name_scope('conditional_idis')
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
        y = self.dc4(y)

        # y = add_noise(y, self.use_noise, self.noise_sigma)
        # with name_scope('idis_dc4', self.dc4.params()):
        #     y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
        #
        # y = add_noise(y, self.use_noise, self.noise_sigma)
        # with name_scope('idis_dc5', self.dc5.params()):
        #     y = self.dc5(y)

        return y

class ConditionalVideoDiscriminator(CategoricalVideoDiscriminator):
    def __init__(self, *args, **kwargs):
        super(ConditionalVideoDiscriminator, self).__init__(*args, **kwargs)

    @within_name_scope('conditional_vdis')
    def __call__(self, x):
        """
        input shape:  (batchsize, ch, video_length, y, x)
        output shape: (batchsize, )
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
#}}}

# infoGAN model set
# {{{
class InfoImageGenerator(CategoricalImageGenerator):
    def __init__(self, *args, **kwargs):
        super(InfoImageGenerator, self).__init__(*args, **kwargs)

    @within_name_scope('info_igen')
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

        z = F.concat((zc, zm, zl), axis=2)
        z = F.reshape(z, (self.video_len*batchsize, self.n_hidden, 1, 1))
        
        # G([zc, zm, zl])
        with name_scope('gen_dc2', self.dc2.params()):
            x = F.relu(self.bn1(self.dc1(z)))
        with name_scope('gen_dc2', self.dc2.params()):
            x = F.relu(self.bn2(self.dc2(x)))
        with name_scope('gen_dc3', self.dc3.params()):
            x = F.relu(self.bn3(self.dc3(x)))
        with name_scope('gen_dc4', self.dc4.params()):
            x = F.relu(self.bn4(self.dc4(x)))
        with name_scope('gen_dc5', self.dc5.params()):
            x = F.tanh(self.dc5(x))
        x = F.reshape(x, (self.video_len, batchsize, self.out_ch, 64, 64))

        return x, labels

class InfoImageDiscriminator(CategoricalImageDiscriminator):
    def __init__(self, *args, **kwargs):
        super(InfoImageDiscriminator, self).__init__(*args, **kwargs)

    @within_name_scope('info_idis')
    def __call__(self, x):
        """
        input shape:  (batchsize, 3, 64, 64)
        output shape: (batchsize, 1)
        """
        y = add_noise(x, self.use_noise, self.noise_sigma)
        with name_scope('idis_dc1', self.dc1.params()):
            y = F.leaky_relu(self.dc1(y), slope=0.2)

        y = add_noise(y, self.use_noise, self.noise_sigma)
        with name_scope('idis_dc2', self.dc2.params()):
            y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)

        y = add_noise(y, self.use_noise, self.noise_sigma)
        with name_scope('idis_dc3', self.dc3.params()):
            y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)

        # y = add_noise(y, self.use_noise, self.noise_sigma)
        # with name_scope('idis_dc4', self.dc4.params()):
        #     y = self.dc4(y)

        y = add_noise(y, self.use_noise, self.noise_sigma)
        with name_scope('idis_dc4', self.dc4.params()):
            y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)

        y = add_noise(y, self.use_noise, self.noise_sigma)
        with name_scope('idis_dc5', self.dc5.params()):
            y = self.dc5(y)

        return y

class InfoVideoDiscriminator(CategoricalVideoDiscriminator):
    def __init__(self, *args, **kwargs):
        super(InfoVideoDiscriminator, self).__init__(*args, **kwargs)

    @within_name_scope('info_vdis')
    def __call__(self, x):
        """
        input shape:  (batchsize, ch, video_length, y, x)
        output shape: (batchsize, )
        """

        y = add_noise(x, self.use_noise, self.noise_sigma)
        with name_scope('vdis_dc1', self.dc1.params()):
            y = F.leaky_relu(self.dc1(y), slope=0.2)

        y = add_noise(y, self.use_noise, self.noise_sigma)
        with name_scope('vdis_dc2', self.dc2.params()):
            y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)

        y = add_noise(y, self.use_noise, self.noise_sigma)
        with name_scope('vdis_dc3', self.dc3.params()):
            y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
        
        y = add_noise(y, self.use_noise, self.noise_sigma)
        with name_scope('vdis_dc4', self.dc4.params()):
            y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)

        with name_scope('vdis_dc5', self.dc5.params()):
            y = self.dc5(y)

        return y

class PSInfoImageGenerator(CategoricalImageGenerator):
    def __init__(self, out_channels=3, n_filters=64, \
                 video_len=16, dim_zc=50, dim_zm=10, dim_zl=6):
        super(CategoricalImageGenerator, self).__init__()
        
        self.out_ch = out_channels
        self.video_len = video_len
        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.dim_zl = dim_zl
        self.n_hidden = dim_zc + dim_zm + dim_zl

        with self.init_scope():
            n_hidden = self.n_hidden

            # w = chainer.initializers.GlorotNormal()
            
            # Rm
            self.g0 = L.StatelessGRU(self.dim_zm, self.dim_zm)
            
            # G
            k = 3 # kernel size of convolution layers
            sk = 1 # kernel size of sub convolution layers
            r = 2 # expantion rate of feature map

            w = chainer.initializers.Uniform(1./(n_hidden*k**2))
            self.cn1 = L.Convolution2D(n_hidden,  n_filters*4, k, stride=1, pad=1, nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*4*sk**2))
            self.scn1 = L.Convolution2D(n_filters*4,  n_filters*4*r**2, sk, stride=1, pad=0, nobias=True, initialW=w)
            self.ps1 = pixel_shuffler(r)

            w = chainer.initializers.Uniform(1./(n_filters*4*k**2))
            self.cn2 = L.Convolution2D(n_filters*4,  n_filters*4, k, stride=1, pad=1, nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*4*sk**2))
            self.scn2 = L.Convolution2D(n_filters*4,  n_filters*4*r**2, sk, stride=1, pad=0, nobias=True, initialW=w)
            self.ps2 = pixel_shuffler(r)
            
            w = chainer.initializers.Uniform(1./(n_filters*4*k**2))
            self.cn3 = L.Convolution2D(n_filters*4,  n_filters*2, k, stride=1, pad=1, nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*2*sk**2))
            self.scn3 = L.Convolution2D(n_filters*2,  n_filters*2*r**2, sk, stride=1, pad=0, nobias=True, initialW=w)
            self.ps3 = pixel_shuffler(r)

            w = chainer.initializers.Uniform(1./(n_filters*2*k**2))
            self.cn4 = L.Convolution2D(n_filters*2,  n_filters*2, k, stride=1, pad=1, nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*2*sk**2))
            self.scn4 = L.Convolution2D(n_filters*2,  n_filters*2*r**2, sk, stride=1, pad=0, nobias=True, initialW=w)
            self.ps4 = pixel_shuffler(r)

            w = chainer.initializers.Uniform(1./(n_filters*2*k**2))
            self.cn5 = L.Convolution2D(n_filters*2,  n_filters, k, stride=1, pad=1, nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(n_filters*sk**2))
            self.scn5 = L.Convolution2D(n_filters,  n_filters*r**2, sk, stride=1, pad=0, nobias=True, initialW=w)
            self.ps5 = pixel_shuffler(r)

            w = chainer.initializers.Uniform(1./(n_filters*k**2))
            self.cn6 = L.Convolution2D(n_filters,  out_channels, k, stride=1, pad=1, nobias=True, initialW=w)
            w = chainer.initializers.Uniform(1./(out_channels*sk**2))
            self.scn6 = L.Convolution2D(out_channels, out_channels*r**2, sk, stride=1, pad=0, nobias=True, initialW=w)
            self.ps6 = pixel_shuffler(r)

            self.bn1 = L.BatchNormalization(n_filters*4)
            self.bn2 = L.BatchNormalization(n_filters*4)
            self.bn3 = L.BatchNormalization(n_filters*2)
            self.bn4 = L.BatchNormalization(n_filters*2)
            self.bn5 = L.BatchNormalization(  n_filters)

    @within_name_scope('ps_info_igen')
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

        z = F.concat((zc, zm, zl), axis=2)
        z = F.reshape(z, (self.video_len*batchsize, self.n_hidden, 1, 1))

        # G([zc, zm, zl])
        x = self.ps1(self.scn1(self.cn1(z)))
        x = F.relu(self.bn1(x))

        x = self.ps2(self.scn2(self.cn2(x)))
        x = F.relu(self.bn2(x))

        x = self.ps3(self.scn3(self.cn3(x)))
        x = F.relu(self.bn3(x))

        x = self.ps4(self.scn4(self.cn4(x)))
        x = F.relu(self.bn4(x))

        x = self.ps5(self.scn5(self.cn5(x)))
        x = F.relu(self.bn5(x))

        x = self.ps6(self.scn6(self.cn6(x)))
        x = F.tanh(x)

        x = F.reshape(x, (self.video_len, batchsize, self.out_ch, 64, 64))

        return x, labels
#}}}


if __name__ ==  "__main__":
    main()
