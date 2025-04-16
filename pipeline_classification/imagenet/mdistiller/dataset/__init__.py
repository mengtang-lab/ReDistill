from .cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample, get_cifar100_dataloaders_trainval, get_cifar100_dataloaders_val_only, get_cifar100_dataloaders_train_only, get_cifar100_dataloaders_strong
from .imagenet import get_imagenet_dataloaders, get_imagenet_dataloaders_sample, get_imagenet_dataloaders_strong


def get_dataset(cfg, data_path):
    if cfg.DATASET.TYPE == "cifar100":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                mode=cfg.CRD.MODE,
                data_path=data_path,
            )
        else:
            train_loader, val_loader, num_data = get_cifar100_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                data_path=data_path,
            )
        num_classes = 100
    elif cfg.DATASET.TYPE == "imagenet":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                data_path=data_path,
            )
        else:
            train_loader, val_loader, num_data = get_imagenet_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                data_path=data_path,
            )
        num_classes = 1000
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)

    return train_loader, val_loader, num_data, num_classes


def get_dataset_strong(cfg, data_path):
    if cfg.DATASET.TYPE == "cifar100":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                mode=cfg.CRD.MODE,
                data_path=data_path,
            )
        else:
            train_loader, val_loader, num_data = get_cifar100_dataloaders_strong(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                data_path=data_path,
            )
        num_classes = 100
    elif cfg.DATASET.TYPE == "imagenet":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                data_path=data_path,
            )
        else:
            train_loader, val_loader, num_data = get_imagenet_dataloaders_strong(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                data_path=data_path,
            )
        num_classes = 1000
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)

    return train_loader, val_loader, num_data, num_classes


