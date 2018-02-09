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

# # cGAN model set
# #{{{
# class ConditionalImageGenerator(CategoricalImageGenerator):
#     def __init__(self, *args, **kwargs):
#         super(ConditionalImageGenerator, self).__init__(*args, **kwargs)
#
#     def __call__(self, h0, zc=None, labels=None):
#         """
#         input h0 shape:  (batchsize, dim_zm)
#         input zc shape:  (batchsize, dim_zc)
#         output shape: (video_length, batchsize, channel, x, y, z)
#         """
#         batchsize = h0.shape[0]
#         xp = chainer.cuda.get_array_module(h0)
#         
#         # make [zc, zm, zl]
#         # z shape: (video_length, batchsize, channel)
#     
#         ## z_content
#         if zc is None:
#             zc = Variable(xp.asarray(self.make_zc(batchsize)))
#
#         ## z_motion
#         zm = self.make_zm(h0, batchsize)
#
#         ## z_label
#         zl, labels = self.make_zl(batchsize, labels)
#         zl = Variable(xp.asarray(zl))
#
#         z = F.concat((zc, zm, zl), axis=2)
#         z = F.reshape(z, (self.video_len*batchsize, self.n_hidden, 1, 1))
#         
#         # G([zc, zm, zl])
#         x = F.relu(self.bn1(self.dc1(z)))
#         x = F.relu(self.bn2(self.dc2(x)))
#         x = F.relu(self.bn3(self.dc3(x)))
#         x = F.relu(self.bn4(self.dc4(x)))
#         x = F.tanh(self.dc5(x))
#         x = F.reshape(x, (self.video_len, batchsize, self.out_ch, 64, 64))
#         
#         return x, labels
#
# class ConditionalImageDiscriminator(CategoricalImageDiscriminator):
#     def __init__(self, *args, **kwargs):
#         super(ConditionalImageDiscriminator, self).__init__(*args, **kwargs)
#
#     def __call__(self, x):
#         """
#         input shape:  (batchsize, 3, 64, 64)
#         output shape: (batchsize, 1)
#         """
#         y = add_noise(x, self.use_noise, self.noise_sigma)
#         y = F.leaky_relu(self.dc1(y), slope=0.2)
#
#         y = add_noise(y, self.use_noise, self.noise_sigma)
#         y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)
#
#         y = add_noise(y, self.use_noise, self.noise_sigma)
#         y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
#
#         # y = add_noise(y, self.use_noise, self.noise_sigma)
#         # y = self.dc4(y)
#
#         y = add_noise(y, self.use_noise, self.noise_sigma)
#         with name_scope('idis_dc4', self.dc4.params()):
#             y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
#
#         y = add_noise(y, self.use_noise, self.noise_sigma)
#         with name_scope('idis_dc5', self.dc5.params()):
#             y = self.dc5(y)
#
#         return y
#
# class ConditionalVideoDiscriminator(CategoricalVideoDiscriminator):
#     def __init__(self, *args, **kwargs):
#         super(ConditionalVideoDiscriminator, self).__init__(*args, **kwargs)
#
#     def __call__(self, x):
#         """
#         input shape:  (batchsize, ch, video_length, y, x)
#         output shape: (batchsize, )
#         """
#
#         y = add_noise(x, self.use_noise, self.noise_sigma)
#         y = F.leaky_relu(self.dc1(y), slope=0.2)
#
#         y = add_noise(y, self.use_noise, self.noise_sigma)
#         y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)
#
#         y = add_noise(y, self.use_noise, self.noise_sigma)
#         y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
#         
#         y = add_noise(y, self.use_noise, self.noise_sigma)
#         y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
#
#         y = self.dc5(y)
#
#         return y
# #}}}
#
# # infoGAN model set
# # {{{
# class InfoImageGenerator(CategoricalImageGenerator):
#     def __init__(self, *args, **kwargs):
#         super(InfoImageGenerator, self).__init__(*args, **kwargs)
#
#     def __call__(self, h0, zc=None, labels=None):
#         """
#         input h0 shape:  (batchsize, dim_zm)
#         input zc shape:  (batchsize, dim_zc)
#         output shape: (video_length, batchsize, channel, x, y, z)
#         """
#         batchsize = h0.shape[0]
#         xp = chainer.cuda.get_array_module(h0)
#         
#         # make [zc, zm, zl]
#         # z shape: (video_length, batchsize, channel)
#     
#         ## z_content
#         if zc is None:
#             zc = Variable(xp.asarray(self.make_zc(batchsize)))
#
#         ## z_motion
#         zm = self.make_zm(h0, batchsize)
#
#         ## z_label
#         zl, labels = self.make_zl(batchsize, labels)
#         zl = Variable(xp.asarray(zl))
#
#         z = F.concat((zc, zm, zl), axis=2)
#         z = F.reshape(z, (self.video_len*batchsize, self.n_hidden, 1, 1))
#         
#         # G([zc, zm, zl])
#         with name_scope('gen_dc1', self.dc1.params()):
#             x = F.relu(self.bn1(self.dc1(z)))
#         with name_scope('gen_dc2', self.dc2.params()):
#             x = F.relu(self.bn2(self.dc2(x)))
#         with name_scope('gen_dc3', self.dc3.params()):
#             x = F.relu(self.bn3(self.dc3(x)))
#         with name_scope('gen_dc4', self.dc4.params()):
#             x = F.relu(self.bn4(self.dc4(x)))
#         with name_scope('gen_dc5', self.dc5.params()):
#             x = F.tanh(self.dc5(x))
#         x = F.reshape(x, (self.video_len, batchsize, self.out_ch, 64, 64))
#
#         return x, labels
#
# class InfoImageDiscriminator(CategoricalImageDiscriminator):
#     def __init__(self, *args, **kwargs):
#         super(InfoImageDiscriminator, self).__init__(*args, **kwargs)
#
#     def __call__(self, x):
#         """
#         input shape:  (batchsize, ch, video_length, y, x)
#         output shape: (batchsize, )
#         """
#
#         y = add_noise(x, self.use_noise, self.noise_sigma)
#         with name_scope('vdis_dc1', self.dc1.params()):
#             y = F.leaky_relu(self.dc1(y), slope=0.2)
#
#         y = add_noise(y, self.use_noise, self.noise_sigma)
#         with name_scope('vdis_dc2', self.dc2.params()):
#             y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)
#
#         y = add_noise(y, self.use_noise, self.noise_sigma)
#         with name_scope('vdis_dc3', self.dc3.params()):
#             y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
#         
#         y = add_noise(y, self.use_noise, self.noise_sigma)
#         with name_scope('vdis_dc4', self.dc4.params()):
#             y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
#
#         with name_scope('vdis_dc5', self.dc5.params()):
#             y = self.dc5(y)
#
#         return y
#
# class PSInfoImageGenerator(CategoricalImageGenerator):
#     def __init__(self, out_channels=3, n_filters=64, \
#                  video_len=16, dim_zc=50, dim_zm=10, dim_zl=6):
#         super(CategoricalImageGenerator, self).__init__()
#         
#         self.out_ch = out_channels
#         self.video_len = video_len
#         self.dim_zc = dim_zc
#         self.dim_zm = dim_zm
#         self.dim_zl = dim_zl
#         self.n_hidden = dim_zc + dim_zm + dim_zl
#
#         with self.init_scope():
#             n_hidden = self.n_hidden
#
#             # w = chainer.initializers.GlorotNormal()
#             
#             # Rm
#             self.g0 = L.StatelessGRU(self.dim_zm, self.dim_zm)
#             
#             # G
#             k = 3 # kernel size of convolution layers
#             sk = 1 # kernel size of sub convolution layers
#             r = 2 # expantion rate of feature map
#
#             oc = out_channels
#             r2 = r ** 2
#
#             if n_filters % r**2 != 0:
#                 print("n_filters is invalid")
#                 raise ValueError
#
#             w = chainer.initializers.Uniform(1./(oc*r2**6*k**2))
#             self.cn1 = L.Convolution2D(n_hidden, oc*r2**6, k, stride=1, pad=1, nobias=True, initialW=w)
#             self.ps1 = PixelShuffler(r)
#
#             w = chainer.initializers.Uniform(1./(oc*r2**5*k**2))
#             self.cn2 = L.Convolution2D(oc*r2**5, oc*r2**5, k, stride=1, pad=1, nobias=True, initialW=w)
#             self.ps2 = PixelShuffler(r)
#             
#             w = chainer.initializers.Uniform(1./(oc*r2**4*k**2))
#             self.cn3 = L.Convolution2D(oc*r2**4, oc*r2**4, k, stride=1, pad=1, nobias=True, initialW=w)
#             self.ps3 = PixelShuffler(r)
#
#             w = chainer.initializers.Uniform(1./(oc*r2**3*k**2))
#             self.cn4 = L.Convolution2D(oc*r2**3, oc*r2**3, k, stride=1, pad=1, nobias=True, initialW=w)
#             self.ps4 = PixelShuffler(r)
#
#             w = chainer.initializers.Uniform(1./(oc*r2**2*k**2))
#             self.cn5 = L.Convolution2D(oc*r2**2, oc*r2**2, k, stride=1, pad=1, nobias=True, initialW=w)
#             self.ps5 = PixelShuffler(r)
#
#             w = chainer.initializers.Uniform(1./(oc*r2*k**2))
#             self.cn6 = L.Convolution2D(oc*r2, oc*r2, k, stride=1, pad=1, nobias=True, initialW=w)
#             self.ps6 = PixelShuffler(r)
#
#             self.bn1 = L.BatchNormalization(oc*r2**5)
#             self.bn2 = L.BatchNormalization(oc*r2**4)
#             self.bn3 = L.BatchNormalization(oc*r2**3)
#             self.bn4 = L.BatchNormalization(oc*r2**2)
#             self.bn5 = L.BatchNormalization(oc*r2**1)
#
#     def __call__(self, h0, zc=None, labels=None):
#         """
#         input h0 shape:  (batchsize, dim_zm)
#         input zc shape:  (batchsize, dim_zc)
#         output shape: (video_length, batchsize, channel, x, y, z)
#         """
#         batchsize = h0.shape[0]
#         xp = chainer.cuda.get_array_module(h0)
#         
#         # make [zc, zm, zl]
#         # z shape: (video_length, batchsize, channel)
#     
#         ## z_content
#         if zc is None:
#             zc = Variable(xp.asarray(self.make_zc(batchsize)))
#
#         ## z_motion
#         zm = self.make_zm(h0, batchsize)
#
#         ## z_label
#         zl, labels = self.make_zl(batchsize, labels)
#         zl = Variable(xp.asarray(zl))
#
#         z = F.concat((zc, zm, zl), axis=2)
#         z = F.reshape(z, (self.video_len*batchsize, self.n_hidden, 1, 1))
#
#         # G([zc, zm, zl])
#         x = F.relu(self.bn1(self.ps1(self.cn1(z))))
#         x = F.relu(self.bn2(self.ps2(self.cn2(x))))
#         x = F.relu(self.bn3(self.ps3(self.cn3(x))))
#         x = F.relu(self.bn4(self.ps4(self.cn4(x))))
#         x = F.relu(self.bn5(self.ps5(self.cn5(x))))
#         x = F.tanh(self.ps6(self.cn6(x)))
#
#         x = F.reshape(x, (self.video_len, batchsize, self.out_ch, 64, 64))
#
#         return x, labels
# #}}}

if __name__ ==  "__main__":
    main()
