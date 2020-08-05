import torch
import random
import argparse



class ImageBuffer():
    def __init__(self,max_size):
        self.buffer_size = max_size
        if self.buffer_size>0:
            self.num_imgs = 0
            self.imgs = []

    def reselect(self, input):
        if not (self.buffer_size > 0):
            return input
        to_return = []
        for image in input:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.buffer_size:
                self.num_imgs += 1
                self.imgs.append(image)
                to_return.append(image)
            else:
                if random.uniform(0,1) > 0.5:
                    img_id = random.randint(0,self.buffer_size-1)
                    tmp = self.imgs[img_id].clone()
                    self.imgs[img_id] = image
                    to_return.append(tmp)
                else:
                    to_return.append(image)
        return torch.cat(to_return)


def Options():
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument('--generated_images_dir', type=str, default='./images', help='saves results here.')
    parser.add_argument('--dataroot', required=True, help='path of datasets')
    parser.add_argument('--name', type=str, default='CycleGAN_homework', help='name of the experiment.')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='save models here')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    # model parameters
    parser.add_argument('--input_nc', type=int, default=3,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='# of residual blocks in generator')
    parser.add_argument('--dropout', action='store_true', help='if use dropout for the generator or not ')
    # dataset parameters
    parser.add_argument('--shuffle', action='store_false',
                        help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--n_cpu', default=8, type=int, help='# of cpu threads to use during batch generation')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--image_size', type=int, default=256, help='image size of input')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                        help='maximum number of samples allowed per dataset.')
    # additional parameters
    parser.add_argument('--load_epoch', type=str, default='latest', help='which epoch to load')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training losses')
    # network saving and loading parameters
    parser.add_argument('--save_epoch_freq', type=int, default=5,
                        help='frequency of saving checkpoints')
    parser.add_argument('--continue_train', action='store_true',
                        help='continue training: load the latest model')
    parser.add_argument('--epoch_to_start', type=int, default=1,
                        help='epoch to start training from')
    # training parameters
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='# of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100,
                        help='# of epochs for learning rate decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--beta2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--buffer_size', type=int, default=50,
                        help='the size of image buffer')
    # CycleGAN options
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='weight for cycle loss ')
    parser.add_argument('--lambda_identity', type=float, default=5.0, help='weight for identity loss ')

    # Test options
    parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
    parser.add_argument('--num_test', type=int, default=100, help='# of test images')
    # get the options
    opt = parser.parse_args()
    # Comment this if you want to specify these options yourself
    if opt.phase == 'test':
        opt.shuffle = False
        opt.dropout = False
        opt.n_cpu = 1
        opt.batch_size = 1
    print(opt)
    return opt

