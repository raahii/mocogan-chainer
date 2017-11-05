import chainer
import chainer.functions as F
from chainer import Variable

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gru, self.gen, self.image_dis, self.video_dis = kwargs.pop('models')
        self.T = kwargs.pop('video_length')
        self.img_size = kwargs.pop('img_size')
        super(Updater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gru(self, gru, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gru)
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gru_optimizer = self.get_optimizer('gru')
        gen_optimizer = self.get_optimizer('gen')
        image_dis_optimizer = self.get_optimizer('image_dis')
        video_dis_optimizer = self.get_optimizer('video_dis')

        gru, gen = self.gru, self.gen
        image_dis, video_dis = self.image_dis, self.video_dis
        
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        
        ## real data
        x_real = Variable(self.converter(batch, self.device) / 255.)
        xp = chainer.cuda.get_array_module(x_real.data)

        y_real_i = image_dis(x_real[:,:,xp.random.randint(0, self.T)])
        y_real_v = video_dis(x_real)

        ## fake data
        zc = Variable(xp.asarray(gru.make_zc(batchsize)))
        h0 = Variable(xp.asarray(gru.make_h0(batchsize)))
        
        # x_fake = xp.empty((self.T, batchsize, 3, self.img_size, self.img_size), dtype=xp.float32)
        x_fake = Variable()
        for i in range(self.T):
            e = Variable(xp.asarray(gru.make_zm(batchsize)))
            zm = gru(h0, e)
            z = F.concat([zc, zm], axis=1)

            bs, zd = z.shape
            z = F.reshape(z, (bs, zd, 1, 1))
            frame = F.reshape(gen(z), (1, batchsize, 3, self.img_size, self.img_size))

            if x_fake.data is None:
                x_fake = frame
            else:
                x_fake = F.concat([x_fake, frame], axis=0)
        
        x_fake = x_fake.transpose(1, 2, 0, 3, 4)
        y_fake_i = image_dis(x_fake[:,:,xp.random.randint(0, self.T)])
        y_fake_v = video_dis(x_fake)
        y_fake = y_fake_i + y_fake_v.reshape(batchsize, 1, 1, 1)
        # import pdb; pdb.set_trace()
        
        ## update
        image_dis_optimizer.update(self.loss_dis, image_dis, y_fake_i, y_real_i)
        video_dis_optimizer.update(self.loss_dis, video_dis, y_fake_v, y_real_v)
        gru_optimizer.update(self.loss_gru, gru, y_fake)
        gen_optimizer.update(self.loss_gen, gen, y_fake)
