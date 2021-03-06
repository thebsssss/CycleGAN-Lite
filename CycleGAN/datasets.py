import random
import glob
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def CycleLoader(opt):
    dataset = CycleDataset(opt)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=opt.shuffle,
        num_workers=opt.n_cpu
    )
    return dataloader


class CycleDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        # create image paths
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        # load images
        self.files_A = sorted(glob.glob(self.dir_A + "/*.*"))
        self.A_size = min(opt.max_dataset_size, len(self.files_A))
        self.files_A = self.files_A[: self.A_size]
        self.files_B = sorted(glob.glob(self.dir_B + "/*.*"))
        self.B_size = min(opt.max_dataset_size, len(self.files_B))
        self.files_B = self.files_B[: self.B_size]

        self.transform_A = transformer_set(self.opt, grayscale=(opt.input_nc == 1))
        self.transform_B = transformer_set(self.opt, grayscale=(opt.output_nc == 1))

    def __getitem__(self, index):
        files_A = self.files_A[index % self.A_size]
        if not self.opt.shuffle:
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        files_B = self.files_B[index_B]
        A_img = Image.open(files_A)
        B_img = Image.open(files_B)

        if (A_img.mode != 'RGB'):
            A_img = A_img.convert('RGB')
        if (B_img.mode != 'RGB'):
            B_img = B_img.convert('RGB')

        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)


        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.A_size, self.B_size)

def transformer_set(opt, grayscale=False, method=Image.BICUBIC):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if opt.phase == 'train':
        re_size = int(1.12*opt.image_size)
        in_size = [re_size, re_size]
        transform_list.append(transforms.Resize(in_size, method))
        transform_list.append(transforms.RandomCrop(opt.image_size))
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor()]
    if grayscale:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)




