import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset_strong
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict


from reed import ReED

def main(cfg, data_path, resume, opts, device_ids=[0]):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset_strong(cfg, data_path)

    # distillation
    print(log_msg("Loading teacher model", "INFO"))
    if cfg.DATASET.TYPE == "imagenet":
        model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True, image_size=224, num_classes=num_classes)
        model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False, image_size=224, num_classes=num_classes)
        im_size = 224
    else:
        net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
        assert (
            pretrain_model_path is not None
        ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
        model_teacher = net(image_size=128, num_classes=num_classes)
        model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
        model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
            image_size=128,
            num_classes=num_classes,
        )
        im_size = 128

    print(model_student)
    print(model_teacher)
    distiller = ReED(model_student, [model_teacher], cfg.DISTILLER.DIST_CONFIG, dummy_input=torch.randn(1,3,im_size,im_size))
    distiller = torch.nn.DataParallel(distiller.cuda(), device_ids=device_ids)

    # if cfg.DISTILLER.TYPE != "NONE":
    #     print(
    #         log_msg(
    #             "Extra parameters of {}: {}\033[0m".format(
    #                 cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
    #             ),
    #             "INFO",
    #         )
    #     )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device_ids = list(map(int, args.devices.replace(' ', '').split(',')))
    main(cfg, args.data_path, args.resume, args.opts, device_ids=device_ids)
