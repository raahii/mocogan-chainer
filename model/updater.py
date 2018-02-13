import chainer
import chainer.functions as F
from chainer import Variable
from chainer.dataset import concat_examples
from scipy import linalg
import numpy as np
import re

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop('model')
        self.image_gen, self.image_dis, self.video_dis = kwargs.pop('models')
        self.video_length = kwargs.pop('video_length')
        self.img_size = kwargs.pop('img_size')
        self.channel  = kwargs.pop('channel')
        self.dim_zl  = kwargs.pop('dim_zl')
        self.tf_writer = kwargs.pop('tensorboard_writer')

        super(Updater, self).__init__(*args, **kwargs)
    
    def loss_dis(self, dis, y_real, y_fake, t_real, t_fake):
        batchsize = len(y_fake)

        # gan criterion
        loss = F.sum(F.softplus(-y_real[:1])) / batchsize
        loss += F.sum(F.softplus(y_fake)[:1]) / batchsize
        
        if self.model == 'infogan' and dis.name == "VideoDiscriminator":
            # eliminate shape difference
            N = y_real.shape[0]
            C = y_real.shape[1]
            y_real = F.reshape(y_real, (N, C))
            y_fake = F.reshape(y_fake, (N, C))

            # categorical criterion
            loss += F.softmax_cross_entropy(y_real[:, 1:], t_real)
            loss += F.softmax_cross_entropy(y_fake[:, 1:], t_fake)

        if self.is_new_epoch:
            chainer.report({'loss': loss}, dis)
            self.tf_writer.add_scalar('loss:{}'.format(dis.name), \
                                       loss.data, self.epoch)

        return loss

    def loss_gen(self, gen, y_fake_i, y_fake_v, t_fake):
        batchsize = len(y_fake_i)

        # gan criterion
        loss = F.sum(F.softplus(-y_fake_i[:, 0])) / batchsize
        loss += F.sum(F.softplus(-y_fake_v[:, 0])) / batchsize
        
        if self.model == 'infogan':
            # categorical criterion
            loss += F.softmax_cross_entropy(y_fake_i[:, 1:, 0, 0], t_fake)
            loss += F.softmax_cross_entropy(y_fake_v[:, 1:, 0, 0, 0], t_fake)

        if self.is_new_epoch:
            chainer.report({'loss': loss}, gen)
            self.tf_writer.add_scalar('loss:{}'.format(gen.name), \
                                       loss.data, self.epoch)

        return loss

    def concat_label_video(self, video, label, xp):
        """
        Concatenate video with label

        :param np.ndarray video: shape: (batchsize, channel, video_length, height, width)
        :param class xp, numpy or cupy
        """
        N, C, T, H, W = video.shape
        label_video = -1.0 * xp.ones((N, self.dim_zl, T, H, W), dtype=np.float32)
        label_video[np.arange(N), label] = 1.

        return F.concat((video, label_video), axis=1)

    def update_core(self):
        ## load models
        image_gen_optimizer = self.get_optimizer('image_gen')
        image_dis_optimizer = self.get_optimizer('image_dis')
        video_dis_optimizer = self.get_optimizer('video_dis')
        image_gen            = self.image_gen
        image_dis, video_dis = self.image_dis, self.video_dis
            
        ## real data
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x_real, t_real = concat_examples(batch)
        x_real = Variable(self.converter(x_real, self.device))
        xp = chainer.cuda.get_array_module(x_real.data)
        t_real = Variable(xp.asarray(t_real).astype(np.int))
        if self.model == 'cgan':
            # concat label features
            x_real = self.concat_label_video(x_real, t_real, xp)
        t = xp.random.randint(0, self.video_length)
        y_real_i = image_dis(x_real[:,:,t])
        y_real_v = video_dis(x_real)

        ## fake data
        x_fake, t_fake = image_gen(batchsize, xp)
        x_fake = x_fake.transpose(1, 2, 0, 3, 4) # (T, N, C, H, W) -> (N, C, T, H, W)
        t_fake = Variable(xp.asarray(t_fake).astype(np.int))
        if self.model == 'cgan':
            # concat label features
            x_fake = self.concat_label_video(x_fake, t_fake, xp)
        y_fake_i = image_dis(x_fake[:,:,t])
        y_fake_v = video_dis(x_fake)

        ## update
        image_dis_optimizer.update(self.loss_dis, image_dis, y_real_i, y_fake_i, t_real, t_fake)
        video_dis_optimizer.update(self.loss_dis, video_dis, y_real_v, y_fake_v, t_real, t_fake)
        image_gen_optimizer.update(self.loss_gen, image_gen, y_fake_i, y_fake_v, t_fake)
