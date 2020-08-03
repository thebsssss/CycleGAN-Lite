import argparse
import torch

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # basic parameters
        parser.add_argument('--results_dir', type=str, default='./results', help='saves results here.')
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # model parameters
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--n_residual_blocks', type=int, default=9, help='# of residual blocks in generator')
        parser.add_argument('--dropout', action='store_true', help='if use dropout for the generator or not ')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--shuffle', action='store_false', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--n_cpu', default=8, type=int, help='# of cpu threads to use during batch generation')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='crop images to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        # additional parameters
        parser.add_argument('--load_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.initialized = True
        return parser


    def parse(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        # save and return the parser
        self.parser = parser
        # get the options
        opt = parser.parse_args()
        # train or test
        opt.isTrain = self.isTrain

        # comment this section if you want to set the test options yourself
        if not opt.isTrain:
            opt.num_threads = 1
            opt.batch_size = 1
            opt.shuffle = False
            opt.dropout = False

        print(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training losses on console')
        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true',
                            help='continue training: load the latest model')
        parser.add_argument('--epoch_to_start', type=int, default=1,
                            help='the start epoch of training')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100,
                                    help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100,
                            help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
        parser.add_argument('--beta2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--buffer_size', type=int, default=50,
                            help='the size of image buffer that stores previously generated images')
        # CycleGAN options
        parser.add_argument('--lambda_cycle', type=float, default=10.0, help='weight for cycle loss ')
        parser.add_argument('--lambda_identity', type=float, default=5.0, help='weight for identity loss ')

        self.isTrain = True
        return parser

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=100, help='how many test images to run')
        parser.set_defaults(phase='test')
        self.isTrain = False

        return parser
