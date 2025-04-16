import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader


class CelebAHQ:
    default_im_size = 256
    default_bs = 16
    default_preprocess = transforms.Compose(
            [
                transforms.Resize((default_im_size, default_im_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    default_cache_dir = "~/.cache/huggingface/datasets"

    def __init__(self, dataset_name, split="train", preprocess=None, cache_dir=None, **kwargs):
        self.dataset = load_dataset(dataset_name, split=split, cache_dir=self.default_cache_dir if cache_dir is None else cache_dir, **kwargs)
        self.preprocess = self.default_preprocess if preprocess is None else preprocess
        self.dataset.set_transform(self.transform)

    def transform(self, examples):
        examples["image"] = [self.preprocess(image.convert("RGB")) for image in examples["image"]]
        return examples

    def dataloader(self, batch_size=None, shuffle=True):
        batch_size = self.default_bs if batch_size is None else batch_size
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    img_size = 256
    hf_dataset = CelebAHQ("mattymchen/celeba-hq", split="train[0:5000]", preprocess=transforms.Compose([
                                    transforms.Resize(img_size),
                                    transforms.RandomCrop((img_size, img_size)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]), cache_dir=".cache/dataset/")

    # print(help(hf_dataset))

    dataloader = hf_dataset.dataloader()

    # for data in dataloader:
    #     if 1 in data["label"]:
    #         print(data)

    # print(len(dataloader))
