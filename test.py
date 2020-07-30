import time
from CycleGAN import *

if __name__ == '__main__':
    opt = options.TestOptions().parse()
    dataloader = datasets.CycleLoader(opt)

    model = models.CycleGAN(opt)

    for i,batch in enumerate(dataloader):
        if i*opt.batch_size>=opt.num_test:
            break
        model.input(batch)
        model.test()
