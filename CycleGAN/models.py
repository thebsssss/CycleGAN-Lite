import torch.nn as nn
import torch
import itertools
from collections import OrderedDict
import os
from CycleGAN.utils import ImageBuffer,mkdirs
from torchvision.utils import make_grid,save_image

class CycleGAN():
    def __init__(self,opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True

        self.optimizers = []
        self.image_paths = []
        self.loss_names = ['total_loss_G', 'gan_loss','id_loss','cycle_loss','total_loss_D_A','total_loss_D_B']

        if self.isTrain:
            self.network_names = ['G_AB', 'G_BA', 'D_A', 'D_B']
        else:  # during test time, only load Generators
            self.network_names = ['G_AB', 'G_BA']

        if self.isTrain:
            if opt.lambda_identity >0:
                assert opt.input_nc==opt.output_nc

        self.G_AB = ResnetGenerator(opt.input_nc, opt.output_nc, opt.dropout, opt.n_residual_blocks)
        self.G_BA = ResnetGenerator(opt.output_nc, opt.input_nc, opt.dropout, opt.n_residual_blocks)
        self.G_AB = init_network(self.G_AB, self.gpu_ids)
        self.G_BA = init_network(self.G_BA, self.gpu_ids)

        if self.isTrain:
            self.D_A = Discriminator(opt.input_nc)
            self.D_B = Discriminator(opt.output_nc)
            self.D_A = init_network(self.D_A, self.gpu_ids)
            self.D_B = init_network(self.D_B, self.gpu_ids)
            # Replay buffer
            self.fake_A_buffer = ImageBuffer(opt.buffer_size)
            self.fake_B_buffer = ImageBuffer(opt.buffer_size)
            # losses
            self.criterion_GAN = torch.nn.MSELoss()
            self.criterion_cycle = torch.nn.L1Loss()
            self.criterion_identity = torch.nn.L1Loss()
            # optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_AB.parameters(),self.G_BA.parameters()), lr=opt.lr,betas=(opt.beta1,opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch: 1.0 - max(0, epoch + opt.epoch_to_start - opt.n_epochs) / float(opt.n_epochs_decay + 1)) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train:
            self.load_models(opt.load_epoch)

        if self.opt.phase == 'test':
            self.eval()

    def loss_list(self):
        loss_dict = OrderedDict()
        for name in self.loss_names:
            loss_dict[name] = getattr(self,name)
        return loss_dict

    def update_parameters(self):
        self.fake_B = self.G_AB(self.real_A)
        self.fake_A = self.G_BA(self.real_B)
        self.recov_A = self.G_BA(self.fake_B)
        self.recov_B = self.G_AB(self.fake_A)


        # Train generators
        self.optimizer_G.zero_grad()
        # Identity loss
        if self.opt.lambda_identity>0:
            id_loss_A = self.criterion_identity(self.G_BA(self.real_A), self.real_A)
            id_loss_B = self.criterion_identity(self.G_AB(self.real_B), self.real_B)
            self.id_loss = id_loss_A+id_loss_B
        else:
            self.id_loss = 0

        # Gan loss
        fake_score_B = self.D_B(self.fake_B)
        self.gan_loss_AB = self.criterion_GAN(fake_score_B,torch.ones_like(fake_score_B).to(self.device))
        fake_score_A = self.D_A(self.fake_A)
        self.gan_loss_BA = self.criterion_GAN(fake_score_A, torch.ones_like(fake_score_A).to(self.device))
        self.gan_loss = self.gan_loss_AB+ self.gan_loss_BA
        # Cycle loss
        self.cycle_loss_A = self.criterion_cycle(self.recov_A,self.real_A)
        self.cycle_loss_B = self.criterion_cycle(self.recov_B, self.real_B)
        self.cycle_loss = self.cycle_loss_A + self.cycle_loss_B

        # Total loss
        self.total_loss_G = self.id_loss*self.opt.lambda_identity + self.gan_loss + self.cycle_loss*self.opt.lambda_identity
        self.total_loss_G.backward()
        self.optimizer_G.step()

        # Train discriminator
        self.optimizer_D.zero_grad()
        # D_A
        # Real loss
        real_score_A = self.D_A(self.real_A)
        self.real_loss_A = self.criterion_GAN(real_score_A,torch.ones_like(real_score_A).to(self.device))
        # Fake loss using a history of generated images
        fake_A = self.fake_A_buffer.reselect(self.fake_A)
        fake_score_A = self.D_A(fake_A)
        self.fake_loss_A = self.criterion_GAN(fake_score_A,torch.ones_like(fake_score_A).to(self.device))

        # Total loss
        self.total_loss_D_A = self.real_loss_A + self.fake_loss_A

        self.total_loss_D_A.backward()

        # D_B
        # Real loss
        real_score_B = self.D_A(self.real_B)
        self.real_loss_B = self.criterion_GAN(real_score_B, torch.ones_like(real_score_B).to(self.device))
        # Fake loss using a history of generated images
        fake_B = self.fake_B_buffer.reselect(self.fake_B)
        fake_score_B = self.D_A(fake_B)
        self.fake_loss_B = self.criterion_GAN(fake_score_B, torch.ones_like(fake_score_B).to(self.device))

        # Total loss
        self.total_loss_D_B = self.real_loss_B + self.fake_loss_B

        self.total_loss_D_B.backward()
        self.optimizer_D.step()

    def update_lr(self):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate has changed from % .8f to %.8f'%(old_lr,lr))

    def eval(self):
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        with torch.no_grad:
            self.fake_B = self.G_AB(self.real_A)
            self.fake_A = self.G_BA(self.real_B)




    def save_images(self,batch,epoch = None ):
        real_A = make_grid(self.real_A, nrow=self.opt.batch_size, normalize=True)
        real_B = make_grid(self.real_A, nrow=self.opt.batch_size, normalize=True)
        fake_A = make_grid(self.fake_A, nrow=self.opt.batch_size, normalize=True)
        fake_B = make_grid(self.fake_B, nrow=self.opt.batch_size, normalize=True)

        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)

        path = os.path.join(self.opt.results_dir,self.opt.name,self.opt.phase)
        mkdirs(path)
        if epoch:
            file_name = '%s_batch_%s_epoch.png'%(batch,epoch)
        else:
            file_name = '%s_batch.png'%(batch)
        save_path = os.path.join(path,file_name)
        save_image(image_grid,save_path, False)
        print('Images are saved at: %s' % save_path)


    def input(self,input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

    def save_models(self,epoch):
        for name in self.network_names:
            filename = '%s_net_%s.pth'%(epoch,name)
            path = os.path.join(self.save_dir,filename)
            net = getattr(self,name)
            if len(self.gpu_ids)>0 and torch.cuda.is_available():
                torch.save(net.module.cpu().state_dict(),path)
            else:
                torch.save(net.cpu().state_dict(),path)

    def load_models(self,epoch):
        for name in self.network_names:
            filename = '%s_net_%s.pth'%(epoch,name)
            path = os.path.join(self.save_dir,filename)
            net = getattr(self,name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            net.load_state_dict(torch.load(path, map_location=self.device))
            print('Successfully load model from %s'%path)



class ResidualBlock(nn.Module):
    def __init__(self, in_features,use_dropout):
        super(ResidualBlock, self).__init__()

        block = [nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3),
                nn.InstanceNorm2d(in_features),
                nn.ReLU(inplace=True)]

        if use_dropout:
            block += [nn.Dropout(0.5)]

        block += block

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)

class ResnetGenerator(nn.Module):
    def __init__(self,input_nc, output_nc, use_dropout = False , n_residual_blocks=9):
        super(ResnetGenerator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc,64,7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [nn.Conv2d(in_features,out_features,3,stride=2,padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features,use_dropout)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features,out_features,3,stride=2,padding=1,output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64,output_nc,7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self,x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self,input_nc):
        super(Discriminator, self).__init__()

        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(512,1,4,padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self,x):
        return self.model(x)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)

def init_network(net,gpu_ids = []):
    if len(gpu_ids)>0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = nn.DataParallel(net,gpu_ids)
    net.apply(weights_init_normal)
    return net