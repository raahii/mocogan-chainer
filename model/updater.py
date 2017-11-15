import chainer
import chainer.functions as F
from chainer import Variable

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.image_gen, self.image_dis, self.video_dis = kwargs.pop('models')
        self.T = kwargs.pop('video_length')
        self.img_size = kwargs.pop('img_size')
        self.channel  = kwargs.pop('channel')
        super(Updater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        image_gen_optimizer = self.get_optimizer('image_gen')
        image_dis_optimizer = self.get_optimizer('image_dis')
        video_dis_optimizer = self.get_optimizer('video_dis')

        image_gen = self.image_gen
        image_dis, video_dis = self.image_dis, self.video_dis
        
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        
        ## real data
        x_real = Variable(self.converter(batch, self.device))
        xp = chainer.cuda.get_array_module(x_real.data)
        y_real_i = image_dis(x_real[:,:,xp.random.randint(0, self.T)])
        y_real_v = video_dis(x_real)

        ## fake data
        h0 = image_gen.make_hidden(batchsize, image_gen.dim_zm)
        x_fake = image_gen(h0)
        x_fake = x_fake.transpose(0, 2, 1, 3, 4)
        y_fake_i = image_dis(x_fake[:,:,xp.random.randint(0, self.T)])
        y_fake_v = video_dis(x_fake)
        y_fake = y_fake_i + y_fake_v.reshape(batchsize, 1, 1, 1)
        
        ## update
        image_dis_optimizer.update(self.loss_dis, image_dis, y_fake_i, y_real_i)
        video_dis_optimizer.update(self.loss_dis, video_dis, y_fake_v, y_real_v)
        image_gen_optimizer.update(self.loss_gen, image_gen, y_fake)
