import copy
import json
import os
import warnings
from absl import app, flags
from tqdm import trange

import numpy as np
import torch


import kagglehub
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets import ImageFolder

from torchvision import transforms


from fld.metrics.FID import FID
from fld.metrics.FLD import FLD
from fld.metrics.KID import KID
from fld.metrics.PrecisionRecall import PrecisionRecall

from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from fld.features.CLIPFeatureExtractor import CLIPFeatureExtractor
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor

FLAGS = flags.FLAGS
flags.DEFINE_integer('img_size', 64, help='image size')
flags.DEFINE_string('metrics', 'fid, kid, precision, recall', help='fid | fld | kid | precision | recall | all')
flags.DEFINE_string('extractor', 'inception', help='inception | clip | dinov2')
flags.DEFINE_string('data_type', 'cifar100', help='data type, must be in [cifar10, cifar100, cifar10lt, cifar100lt, imgnetlt, tinyimgnetlt, flowerslt, animalfaceslt]')
flags.DEFINE_string('data_path', '.cache/data', help='data path')
flags.DEFINE_float('imb_factor', 0.01, help='imb_factor for long tail dataset')
flags.DEFINE_string('gen_npy_cache', './', help='generated numpy cache')


def main(argv):
    tran_transform=transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize([FLAGS.img_size, FLAGS.img_size], antialias=True),
        transforms.ToPILImage()
    ])

    if FLAGS.data_type == 'cifar10':
        dataset = CIFAR10(
                root=FLAGS.data_path,
                train=True,
                download=True,
                transform=tran_transform
                )
    elif FLAGS.data_type == 'cifar100':
        dataset = CIFAR100(
                root=FLAGS.data_path,
                train=True,
                download=True,
                transform=tran_transform)
    else:
        FLAGS.data_path = FLAGS.data_path

        dataset = ImageFolder(root=FLAGS.data_path,
                              transform=tran_transform)

    if FLAGS.extractor.lower() == 'inception':
        feature_extractor = InceptionFeatureExtractor()
    elif FLAGS.extractor.lower() == 'clip':
        feature_extractor = CLIPFeatureExtractor()
    elif FLAGS.extractor.lower() == 'dinov2':
        feature_extractor = DINOv2FeatureExtractor()
    else:
        raise NotImplementedError("extractor {} not implemented".format(FLAGS.extractor.lower()))


    train_feat = feature_extractor.get_features(dataset)
    print(train_feat.shape)

    gen_cache = torch.tensor(np.load(FLAGS.gen_npy_cache))
    print(gen_cache.shape)
    gen_feat = feature_extractor.get_tensor_features(gen_cache)
    print(gen_feat.shape)

    metrics = FLAGS.metrics.lower().replace(' ', '').split(',')
    if 'fid' in metrics or 'all' in metrics:
        fid = FID().compute_metric(train_feat, None, gen_feat)
        print('FID: {:.5f}'.format(fid))
    if 'fld' in metrics or 'all' in metrics:
        fld = FLD().compute_metric(train_feat, None, gen_feat)
        print('FLD: {:.5f}'.format(fld))
    if 'kid' in metrics or 'all' in metrics:
        kid = KID().compute_metric(train_feat, None, gen_feat)
        print('KID: {:.5f}'.format(kid))
    if 'precision' in metrics or 'all' in metrics:
        precision = PrecisionRecall(mode="Precision").compute_metric(train_feat, None, gen_feat)
        print('Precision: {:.5f}'.format(precision))
    if 'recall' in metrics or 'all' in metrics:
        recall = PrecisionRecall(mode="Recall").compute_metric(train_feat, None, gen_feat)
        print('Recall: {:.5f}'.format(recall))

if __name__ == '__main__':
    app.run(main)

