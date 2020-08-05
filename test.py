from CycleGAN import *

if __name__ == '__main__':
    opt = utils.Options()
    dataloader = datasets.CycleLoader(opt)

    model = models.CycleGAN(opt)

    for i,batch in enumerate(dataloader):
        if i*opt.batch_size>=opt.num_test:
            break
        model.input(batch)
        model.test()
        model.save_images(i)
