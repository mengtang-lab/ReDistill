from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, F.adaptive_avg_pool2d(f_t, f_s.shape[-2:]))
        return loss
