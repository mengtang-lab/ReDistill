import os
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from torchvision.utils import make_grid, save_image


class ImbalanceCIFAR10(datasets.CIFAR100):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        print('img_num_list', img_num_list)
        self.num_per_cls_dict = dict()
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class ImbalanceCIFAR100(datasets.CIFAR100):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super(ImbalanceCIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        print('img_num_list', img_num_list)
        self.num_per_cls_dict = dict()
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



class ImageNet(ImageFolder):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    default_train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    # default_target_transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #     ]
    # )
    def __init__(self, root, split="train", transform=default_train_transform):
        super(ImageNet, self).__init__(root=os.path.join(root, split), transform=transform)
        # print(self.class_to_idx, [sum(1 for l in self.targets if l==i) for i in self.class_to_idx.values()])
    def __getitem__(self, index):
        img, t = super().__getitem__(index)
        target = self.targets[index]
        # return img, target, index
        # print('img: ', img.shape, '\ttarget: ', target)
        return img, target


class ImbalanceImageNet(ImageNet):
    def __init__(self, root, split="train", imb_type="exp", imb_factor=1.0, rand_number=0, transform=ImageNet.default_train_transform):
        super(ImbalanceImageNet, self).__init__(root, split=split, transform=transform)
        # === classes, class_to_idx, samples, targets ==== #
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(len(self.classes), imb_type, imb_factor)
        self.num_per_cls_dict = dict()
        self.gen_imbalanced_data(img_num_list)
        print(self.class_to_idx, [sum(1 for l in self.targets if l == i) for i in self.class_to_idx.values()])

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.samples) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_samples = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            select_idx = idx[:the_img_num]
            new_samples.append([self.samples[i] for i in select_idx])
            new_targets.extend([the_class, ] * the_img_num)
        new_samples = np.vstack(new_samples)
        self.samples = new_samples
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(len(self.classes)):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


if __name__ == "__main__":
    import torch
    import numpy as np
    from PIL import Image

    data_path = './.cifar100/'
    img_size = 32
    imb_factor = 0.01
    oup_name = 'medium'
    # selected_categories = [7,8,9]
    idx = np.arange(100)
    selected_categories = [i for i in range(67, 100)]
    # oup_name = '99'
    # selected_categories = [99]

    dataset = ImbalanceCIFAR100(
        root=data_path,  # './data',
        # root='...',
        imb_type='exp',
        imb_factor=imb_factor,
        rand_number=0,
        train=True,
        transform=None,
        download=True,
    )

    X, Y = torch.tensor(dataset.data), torch.tensor(dataset.targets)

    dir_path = os.path.join(data_path, oup_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        print('OUTDIR already exists.')

    Xc = []
    for c in selected_categories:
        Xc.append(X[Y==c])
    Xc = torch.cat(Xc)
    # print(Xc)

    # for i, x in enumerate(Xc):
    #     img = Image.fromarray(x.numpy())
    #     img.save(os.path.join(dir_path, f"{i:08d}.png"))


    # print(Xc.shape)
    # save_image(
    #     Xc.permute(0, 3, 1, 2) / 255.0,
    #     os.path.join(dir_path, f"all.png"),
    #     nrow=16)

    print(f"Prob {c}: {Xc.shape[0]}/{X.shape[0]}={100*(Xc.shape[0]/X.shape[0]):.2f}%")



