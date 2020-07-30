import os
import torch
import random


class ImageBuffer():
    def __init__(self,max_size):
        self.buffer_size = max_size
        if self.buffer_size>0:
            self.num_imgs = 0
            self.imgs = []

    def reselect(self, input):
        if self.buffer_size == 0:
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


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)