from pytorch_fid.inception import InceptionV3

import torchvision.transforms as TF
import torch

from fld.features.ImageFeatureExtractor import ImageFeatureExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InceptionFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, save_path=None):
        self.name = "inception"

        super().__init__(save_path)

        self.features_size = 2048
        self.preprocess = TF.ToTensor()

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3(
            [block_idx], resize_input=True, normalize_input=True
        ).to(DEVICE)
        self.model.eval()
        return

    def get_feature_batch(self, img_batch):
        assert img_batch.max() <= 1
        assert img_batch.min() >= 0

        with torch.no_grad():
            features = self.model(img_batch)[0].squeeze()

        return features
