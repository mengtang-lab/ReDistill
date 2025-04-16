import os

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import default_collate


class Dataset(object):
    name = ""
    normalize = None
    num_classes = None
    ori_im_size = None
    TorchDataset = None

    def __init__(self, path, im_size=None):
        self.train_transform = None
        self.test_transform = None
        self.train_dataset = None
        self.test_dataset = None

        self.path = path
        if type(im_size) == int:
            im_size = (im_size, im_size)

        if im_size is None:
            self.im_size = self.ori_im_size
        else:
            self.im_size = im_size

        self.set_train_transform()
        self.set_test_transform()
        self.set_train_dataset()
        self.set_test_dataset()

    def set_train_transform(self, custom_transform=None):
        if custom_transform is not None:
            self.train_transform = custom_transform
            return

        # transform = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                 transforms.RandomResizedCrop(size=self.im_size,
        #                                                              scale=(0.08,1.0),
        #                                                              ratio=(3 / 4, 4 / 3)),
        #                                 transforms.RandAugment(num_ops=4, magnitude=5),
        #                                 transforms.ToTensor(),
        #                                 self.normalize])

        if self.im_size == self.ori_im_size:
            transform = transforms.Compose([transforms.RandomResizedCrop(self.im_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            self.normalize])
        else:
            transform = transforms.Compose([transforms.Resize(self.im_size),
                                            transforms.RandomResizedCrop(self.im_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            self.normalize])

        self.train_transform = transform

    def set_test_transform(self, custom_transform=None):
        if custom_transform is not None:
            self.test_transform = custom_transform
            return
        self.test_transform = transforms.Compose([transforms.Resize(self.im_size),
                                                  transforms.CenterCrop(self.im_size),
                                                  transforms.ToTensor(),
                                                  self.normalize])

    def set_train_dataset(self, custom_dataset=None):
        if custom_dataset is not None:
            self.train_dataset = custom_dataset
            return
        self.train_dataset = self.TorchDataset(root=self.path, train=True, download=True,
                                               transform=self.train_transform)

    def set_test_dataset(self, custom_dataset=None):
        if custom_dataset is not None:
            self.test_dataset = custom_dataset
            return
        self.test_dataset = self.TorchDataset(root=self.path, train=False, download=True,
                                              transform=self.test_transform)

    def train_loader(self, batch_size, shuffle=True, data_aug={"cutmix": 0.2, "mixup": 0.2}):
        if data_aug is not None:
            cutmix = v2.CutMix(num_classes=self.num_classes, alpha=data_aug["cutmix"])
            mixup = v2.MixUp(num_classes=self.num_classes, alpha=data_aug["mixup"])
            cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
            collate_fn = lambda batch: cutmix_or_mixup(*default_collate(batch))
            return torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=shuffle)

    def test_loader(self, batch_size=1, shuffle=False):
        return torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=shuffle)


class CIFAR100(Dataset):
    name = "CIFAR100"
    normalize = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    num_classes = 100
    ori_im_size = (32, 32)
    TorchDataset = datasets.CIFAR100

    def __init__(self, path='./data/cifar_data', **kwargs):
        super(CIFAR100, self).__init__(path, **kwargs)


class CIFAR10(Dataset):
    name = "CIFAR10"
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    num_classes = 10
    ori_im_size = (32, 32)
    TorchDataset = datasets.CIFAR10

    def __init__(self, path='./data/cifar_data', **kwargs):
        super(CIFAR10, self).__init__(path, **kwargs)


class STL10(Dataset):
    name = "STL10"
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    num_classes = 10
    ori_im_size = (96, 96)
    TorchDataset = datasets.STL10

    def __init__(self, path='./data/stl10_data', **kwargs):
        super(STL10, self).__init__(path, **kwargs)

    def set_train_dataset(self):
        self.train_dataset = self.TorchDataset(root=self.path, split="train", download=True,
                                               transform=self.train_transform)

    def set_test_dataset(self):
        self.test_dataset = self.TorchDataset(root=self.path, split="test", download=True,
                                              transform=self.test_transform)


def get_dataset(name, **kwargs):
    NAME = name.upper()
    if NAME=='CIFAR100':
        return CIFAR100(**kwargs)
    elif NAME=='CIFAR10':
        return CIFAR10(**kwargs)
    elif NAME=='STL10':
        return STL10(**kwargs)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    dataset = STL10('../')
    print(dataset.train_loader(8))
