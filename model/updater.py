import chainer
import chainer.functions as F
from chainer import Variable
from chainer.dataset import concat_examples
from scipy import linalg
import numpy as np
import re

def total_size(obj, verbose=False):
    import sys
    from itertools import chain
    from collections import deque
    seen = set()

    def sizeof(o):
        if id(o) in seen:
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default=0)
        if verbose:
            print(s, type(o), repr(o))
        if isinstance(o, (tuple, list, set, frozenset, deque)):
            s += sum(map(sizeof, iter(o)))
        elif isinstance(o, dict):
            s += sum(map(sizeof, chain.from_iterable(o.items())))
        elif "__dict__" in dir(o):  # もっと良い方法はあるはず
            s += sum(map(sizeof, chain.from_iterable(o.__dict__.items())))
        return s

    return sizeof(obj) / 1e6

model_name_regex = re.compile(r'model.net.([a-zA-Z]+) object')
def get_model_name(model):
    # model.net.ImageDiscriminator object at 0x7f0f0c3cdf60
    m = re.search(model_name_regex, str(model))
    return m.group(1)

class NormalUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.image_gen, self.image_dis, self.video_dis = kwargs.pop('models')
        self.T = kwargs.pop('video_length')
        self.img_size = kwargs.pop('img_size')
        self.channel  = kwargs.pop('channel')
        self.tf_writer = kwargs.pop('tensorboard_writer')
        super(NormalUpdater, self).__init__(*args, **kwargs)
    
    def loss_vdis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)

        # gan criterion
        loss = F.sum(F.softplus(-y_real)) / batchsize
        loss += F.sum(F.softplus(y_fake)) / batchsize

        chainer.report({'loss': loss}, dis)
        if self.is_new_epoch:
            self.tf_writer.add_scalar('loss:video_discriminator', loss.data, self.epoch)
            self.tf_writer.add_graph([y_fake, y_real])

        return loss

    def loss_idis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)

        loss = F.sum(F.softplus(-y_real)) / batchsize
        loss += F.sum(F.softplus(y_fake)) / batchsize
        
        chainer.report({'loss': loss}, dis)
        if self.is_new_epoch:
            self.tf_writer.add_scalar('loss:image_discriminator', loss.data, self.epoch)
            self.tf_writer.add_graph([y_fake, y_real])

        return loss

    def loss_gen(self, gen, y_fake_i, y_fake_v):
        batchsize = len(y_fake_i)

        # gan criterion
        loss = F.sum(F.softplus(-y_fake_i)) / batchsize
        loss += F.sum(F.softplus(-y_fake_v)) / batchsize

        chainer.report({'loss': loss}, gen)
        if self.is_new_epoch:
            self.tf_writer.add_scalar('loss:image_generator', loss.data, self.epoch)

        return loss

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
        x_real = Variable(self.converter(batch, self.device))
        xp = chainer.cuda.get_array_module(x_real.data)
        y_real_i = image_dis(x_real[:,0:self.channel,xp.random.randint(0, self.T)])
        y_real_v = video_dis(x_real[:,0:self.channel])

        ## fake data
        h0 = Variable(xp.asarray(image_gen.make_h0(batchsize)))
        x_fake = image_gen(h0)
        # (t, bs, c, y, x) -> (bs, c, t, y, x)
        x_fake = x_fake.transpose(1, 2, 0, 3, 4)
        y_fake_i = image_dis(x_fake[:,0:self.channel,xp.random.randint(0, self.T)])
        y_fake_v = video_dis(x_fake)
        
        ## update
        image_dis_optimizer.update(self.loss_idis, image_dis, y_fake_i, y_real_i)
        video_dis_optimizer.update(self.loss_vdis, video_dis, y_fake_v, y_real_v)
        image_gen_optimizer.update(self.loss_gen, image_gen, y_fake_i, y_fake_v)

class ConditionalGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.image_gen, self.image_dis, self.video_dis = kwargs.pop('models')
        self.T = kwargs.pop('video_length')
        self.img_size = kwargs.pop('img_size')
        self.channel  = kwargs.pop('channel')
        self.tf_writer = kwargs.pop('tensorboard_writer')
        super(ConditionalGANUpdater, self).__init__(*args, **kwargs)
    
    def loss_vdis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)

        # gan criterion
        loss = F.sum(F.softplus(-y_real)) / batchsize
        loss += F.sum(F.softplus(y_fake)) / batchsize

        chainer.report({'loss': loss}, dis)
        if self.is_new_epoch:
            self.tf_writer.add_scalar('loss:video_discriminator', loss.data, self.epoch)
            self.tf_writer.add_graph([y_fake, y_real])

        return loss

    def loss_idis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)

        loss = F.sum(F.softplus(-y_real)) / batchsize
        loss += F.sum(F.softplus(y_fake)) / batchsize

        chainer.report({'loss': loss}, dis)
        if self.is_new_epoch:
            self.tf_writer.add_scalar('loss:image_discriminator', loss.data, self.epoch)
            self.tf_writer.add_graph([y_fake, y_real])

        return loss

    def loss_gen(self, gen, y_fake_i, y_fake_v):
        batchsize = len(y_fake_i)

        # gan criterion
        loss = F.sum(F.softplus(-y_fake_i)) / batchsize
        loss += F.sum(F.softplus(-y_fake_v)) / batchsize

        chainer.report({'loss': loss}, gen)
        if self.is_new_epoch:
            self.tf_writer.add_scalar('loss:image_generator', loss.data, self.epoch)

        return loss

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
        y_real_i = image_dis(x_real[:,0:self.channel,xp.random.randint(0, self.T)])
        y_real_v = video_dis(x_real)

        ## fake data
        h0 = Variable(xp.asarray(image_gen.make_h0(batchsize)))
        x_fake, t_fake = image_gen(h0)
        # (t, bs, c, y, x) -> (bs, c, t, y, x)
        x_fake = x_fake.transpose(1, 2, 0, 3, 4)
        y_fake_i = image_dis(x_fake[:,0:self.channel,xp.random.randint(0, self.T)])
        y_fake_v = video_dis(x_fake)
        
        ## update
        image_dis_optimizer.update(self.loss_idis, image_dis, y_fake_i, y_real_i)
        video_dis_optimizer.update(self.loss_vdis, video_dis, y_fake_v, y_real_v)
        image_gen_optimizer.update(self.loss_gen, image_gen, y_fake_i, y_fake_v)

class InfoGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.image_gen, self.image_dis, self.video_dis = kwargs.pop('models')
        self.T = kwargs.pop('video_length')
        self.img_size = kwargs.pop('img_size')
        self.channel  = kwargs.pop('channel')
        self.tf_writer = kwargs.pop('tensorboard_writer')
        super(InfoGANUpdater, self).__init__(*args, **kwargs)
    
    def loss_vdis(self, dis, y_fake, t_fake, y_real, t_real):
        batchsize = len(y_fake)
        
        # gan criterion
        loss = F.sum(F.softplus(-y_real[:,0])) / batchsize
        loss += F.sum(F.softplus(y_fake[:,0])) / batchsize

        # categorical criterion
        loss += F.softmax_cross_entropy(y_real[:, 1:, 0, 0, 0], t_real)
        loss += F.softmax_cross_entropy(y_fake[:, 1:, 0, 0, 0], t_fake)

        chainer.report({'loss': loss}, dis)
        if self.is_new_epoch:
            y_fake = y_fake[:,:,0,0,0]
            self.tf_writer.add_scalar('loss:video_discriminator', loss.data, self.epoch)
            # self.tf_writer.add_graph([y_fake, y_real])
            # self.tf_writer.add_all_variable_images([y_real, y_fake], pattern='.*vdis.*')
            # self.tf_writer.add_all_parameter_histograms([y_real, y_fake], pattern='.*vdis.*')

        return loss

    def loss_idis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        
        # gan criterion
        loss = F.sum(F.softplus(-y_real)) / batchsize
        loss += F.sum(F.softplus(y_fake)) / batchsize

        chainer.report({'loss': loss}, dis)
        if self.is_new_epoch:
            y_fake = y_fake[:,0,0,0]
            y_real = y_real[:,0,0,0]
            self.tf_writer.add_scalar('loss:image_discriminator', loss.data, self.epoch)
            # self.tf_writer.add_graph([y_fake, y_real])
            # self.tf_writer.add_all_variable_images([y_real, y_fake], pattern='.*idis.*')
            # self.tf_writer.add_all_parameter_histograms([y_real, y_fake], pattern='.*idis.*')

        return loss

    def loss_gen(self, gen, y_fake_i, y_fake_v, t_fake_v):
        batchsize = len(y_fake_i)
        
        # gan criterion
        loss = F.sum(F.softplus(-y_fake_i)) / batchsize
        loss += F.sum(F.softplus(-y_fake_v[:,0])) / batchsize

        # categorical criterion
        loss += F.softmax_cross_entropy(y_fake_v[:, 1:, 0, 0, 0], t_fake_v)
        
        chainer.report({'loss': loss}, gen)
        if self.is_new_epoch:
            y_fake_i = y_fake_i[:,0,0,0]
            self.tf_writer.add_scalar('loss:image_generator', loss.data, self.epoch)
            # self.tf_writer.add_graph([y_fake_i, y_fake_v])
            # self.tf_writer.add_all_variable_images([y_fake_i], pattern='.*gen.*')
            # self.tf_writer.add_all_parameter_histograms([y_fake_i], pattern='.*gen.*')
        
        return loss

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
        t_real = xp.asarray(t_real)
        y_real_i = image_dis(x_real[:,0:self.channel,xp.random.randint(0, self.T)])
        y_real_v = video_dis(x_real[:,0:self.channel])
        
        ## fake data
        h0 = Variable(xp.asarray(image_gen.make_h0(batchsize)))
        x_fake, t_fake = image_gen(h0)
        # (t, bs, c, y, x) -> (bs, c, t, y, x)
        x_fake   = x_fake.transpose(1, 2, 0, 3, 4)
        t_fake   = xp.asarray(t_fake)
        y_fake_i = image_dis(x_fake[:,0:self.channel,xp.random.randint(0, self.T)])
        y_fake_v = video_dis(x_fake[:,0:self.channel])
        
        ## update
        image_dis_optimizer.update(self.loss_idis, image_dis, y_fake_i, y_real_i)
        video_dis_optimizer.update(self.loss_vdis, video_dis, y_fake_v, t_fake, y_real_v, t_fake)
        image_gen_optimizer.update(self.loss_gen,  image_gen, y_fake_i, y_fake_v, t_fake)

def _clip_singular_value(A, name=None):
    U, s, Vh = linalg.svd(A, full_matrices=False)
    s[s > 1] = 1
    if name:
        return name, np.dot(np.dot(U, np.diag(s)), Vh)
    else:
        return np.dot(np.dot(U, np.diag(s)), Vh)

class WGANSVCUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.image_gen, self.image_dis, self.video_dis = kwargs.pop('models')
        self.T = kwargs.pop('video_length')
        self.img_size = kwargs.pop('img_size')
        self.channel  = kwargs.pop('channel')
        self.tf_writer = kwargs.pop('tensorboard_writer')
        super(WGANSVCUpdater, self).__init__(*args, **kwargs)
    
    def loss_vdis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)

        # gan criterion
        loss = F.sum(-y_real[:,0]) / batchsize
        loss += F.sum(y_fake[:,0]) / batchsize

        chainer.report({'loss': loss}, dis)
        if self.is_new_epoch:
            self.tf_writer.add_scalar('loss:video_discriminator', loss.data, self.epoch)

        return loss

    def loss_idis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        
        # gan criterion
        loss = F.sum(-y_real) / batchsize
        loss += F.sum(y_fake) / batchsize

        chainer.report({'loss': loss}, dis)
        if self.is_new_epoch:
            self.tf_writer.add_scalar('loss:image_discriminator', loss.data, self.epoch)

        return loss

    def loss_gen(self, gen, y_fake_i, y_fake_v):
        batchsize = len(y_fake_i)

        # gan criterion
        loss = F.sum(-y_fake_i[:,0]) / batchsize
        loss += F.sum(-y_fake_v[:,0]) / batchsize
        
        chainer.report({'loss': loss}, gen)
        if self.is_new_epoch:
            self.tf_writer.add_scalar('loss:image_generator', loss.data, self.epoch)

        return loss

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
        y_real_i = image_dis(x_real[:,0:self.channel,xp.random.randint(0, self.T)])
        y_real_v = video_dis(x_real)

        ## fake data
        h0 = Variable(xp.asarray(image_gen.make_h0(batchsize)))
        x_fake, t_fake = image_gen(h0)
        # (t, bs, c, y, x) -> (bs, c, t, y, x)
        x_fake = x_fake.transpose(1, 2, 0, 3, 4)
        y_fake_i = image_dis(x_fake[:,0:self.channel,xp.random.randint(0, self.T)])
        y_fake_v = video_dis(x_fake)
        
        ## update
        image_dis_optimizer.update(self.loss_idis, image_dis, y_fake_i, y_real_i)
        video_dis_optimizer.update(self.loss_vdis, video_dis, y_fake_v, y_real_v)
        image_gen_optimizer.update(self.loss_gen, image_gen, y_fake_i, y_fake_v)

        ## Singular Value Clipping
        freq = 2
        if self.iteration % freq == 0:
            dis = self.image_dis
        else:
            dis = self.video_dis
        
        for p in dis.params():
            if p.data.ndim >= 4:
                if self.device >= 0:
                    A = p.data.reshape((p.data.shape[0], -1)).get()
                else:
                    A = p.data.reshape((p.data.shape[0], -1))
                A = _clip_singular_value(A)
                p.data = xp.asarray(A.reshape(p.data.shape))

        for n in dis.links():
            if 'bn' in str(n.name):
                if self.device >= 0:
                    gamma = n.gamma.data.get()
                    std = np.sqrt(n.avg_var.get())
                else:
                    gamma = n.gamma.data
                    std = np.sqrt(n.avg_var)
                gamma[gamma > std] = std[gamma > std]
                gamma[gamma < 0.01 * std] = 0.01 * std[gamma < 0.01 * std]
                n.gamma.data = xp.asarray(gamma)
