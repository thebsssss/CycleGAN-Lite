import time
from CycleGAN import *
import sys




if __name__ == '__main__':
    opt = options.TrainOptions().parse()
    dataloader = datasets.CycleLoader(opt)
    model = models.CycleGAN(opt)

    for epoch in range(opt.epoch_to_start,opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        for i,batch in enumerate(dataloader):
            model.input(batch)
            model.update_parameters()
            loss_list = model.loss_list()
            # Log Information
            to_print = 'Epoch: %d/%d, Batch: %d/%d,'%(epoch,opt.n_epochs+opt.n_epochs_decay, i+1, len(dataloader))
            for key,value in loss_list.items():
                to_print += ' %s: %.3f '%(key,value)
            sys.stdout.write(
                '\r Current:' + to_print+'\r'
            )
            if (i+1)%opt.print_freq == 0:
                print(to_print)
                model.save_images(i,epoch)
        model.update_lr()

        if epoch% opt.save_epoch_freq ==0:
            model.save_models('latest')
            model.save_models(epoch)

        print('Time taken of epoch %d / %d: %d sec '%(epoch,opt.n_epochs+opt.n_epochs_decay,time.time()-epoch_start_time))





