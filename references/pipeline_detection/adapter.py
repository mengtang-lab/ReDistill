import torch
import torch.nn as nn

from mmengine.registry import MODELS

from mdistiller.models import model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.engine.cfg import CFG as cfg


@MODELS.register_module()
class BackboneAdapter(nn.Module):
    def __init__(self, dist_cfg, **kwargs):
        super(BackboneAdapter, self).__init__()
        cfg.merge_from_file(dist_cfg)
        cfg.freeze()

        if cfg.DISTILLER.TYPE == "NONE":
            student = model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
            distiller = distiller_dict[cfg.DISTILLER.TYPE](student)
        else:
            teacher = model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            student = model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                student, teacher, cfg
            )
        self.distiller = distiller

    def forward(self, x):
        print(x)
        outs, loss = self.distiller(x)
        print(outs, loss)
        return outs