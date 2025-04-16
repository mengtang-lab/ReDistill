import os
import yaml
import json
from typing import List, Dict, Tuple

import copy
import torch
import torch.nn as nn

from .modules import build_dist_module
from .hooks import DistillationHook

def read_yaml_to_dict(yaml_path: str, ):

    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value

def parse_config(config):
    print(config, os.path.isfile(config))
    if os.path.isfile(config):
        return read_yaml_to_dict(config)
    raise RuntimeError(f'{config} is not a valid yaml file! Trying to convert string to dict...')

class ReED(nn.Module):
    """
    dist_config = {
        's_layer': {
            't_layer': ['inp_t_layer', 'oup_t_layer'],
            'modules': ['inp_dist_module', 'oup_dist_module'],
            'kwargs': ['inp_dist_module_kwargs', 'oup_dist_module_kwargs'],
        }
    }
    """
    def __init__(self, student: nn.Module, teachers: List[nn.Module] or nn.ModuleList, dist_config: str, dummy_input: torch.Tensor = None):
        super().__init__()
        self.student = student
        self.teachers = nn.ModuleList(teachers)
        for teacher in self.teachers:
            for name, param in teacher.named_parameters(): param.requires_grad = False

        dist_config = parse_config(dist_config) if dist_config is not None else None

        if dist_config is None:
            self.dist_hook = None
            print('Nothing to Distill!')
            # raise Exception('Nothing to Distill!')
        else:
            dist_table, dist_modules = self.config2table(dist_config)
            self.dist_modules = nn.ModuleList(dist_modules)
            self.dist_hook = DistillationHook(self.student, self.teachers, dist_table, dummy_input=dummy_input)

    def config2table(self, dist_config: dict):
        s_module_list = [name for name, _ in self.student.named_modules()]
        dist_modules = []
        dist_table = {}
        for s_layer, config in dist_config.items():
            if s_layer not in s_module_list:
                print(f'S_Layer: {s_layer} is not in the student model!!!')
                continue

            dist_table[s_layer] = {}

            t_layer_inp, t_layer_oup = config['t_layers']
            module_inp, module_oup = config['modules']
            kwargs_inp, kwargs_oup = config['kwargs']

            if None in (t_layer_inp, module_inp): dist_table[s_layer]['input'] = None
            else:
                assert t_layer_inp.startswith('tid')
                t_id, t_layer = t_layer_inp.split('=')
                dist_module = build_dist_module(module_inp, **kwargs_inp)
                dist_table[s_layer]['input'] = {
                    'teacher_id': int(t_id.split('_')[-1]),
                    'teacher_layer': t_layer,
                    'dist_module': dist_module,
                    'dist_loss': 0,
                }
                dist_modules.append(dist_module)

            if None in (t_layer_oup, module_oup): dist_table[s_layer]['output'] = None
            else:
                assert t_layer_oup.startswith('tid')
                t_id, t_layer = t_layer_oup.split('=')
                dist_module = build_dist_module(module_oup, **kwargs_oup)
                dist_table[s_layer]['output'] = {
                    'teacher_id': int(t_id.split('_')[-1]),
                    'teacher_layer': t_layer,
                    'dist_module': dist_module,
                    'dist_loss': 0,
                }
                dist_modules.append(dist_module)

        return dist_table, dist_modules

    def teachers_forward(self, x):
        with torch.no_grad():
            for teacher in self.teachers:
                _ = teacher(x)

    def student_forward(self, x):
        return self.student(x)

    def forward(self, x):
        if self.training:
            self.teachers_forward(x)
            out = self.student_forward(x)
            loss = torch.stack(self.dist_hook.loss()).mean() if self.dist_hook is not None else 0
            return out, loss
        return self.student_forward(x)

    def save(self, save_with_teachers=False):
        pass

    def load(self):
        pass

    def graduate(self):
        self.teachers = None
        self.eval()


class ReED_Diffuser(ReED):
    def __init__(self, student: nn.Module, teachers: List[nn.Module] or nn.ModuleList, dist_config: str, dummy_input: Dict[str, torch.Tensor] = None):
        super().__init__(student, teachers, dist_config, dummy_input)

    # def _zero_init_red_blocks(self):
    #     for module in self.dist_modules.children():
    #         if isinstance(module, nn.Conv2d):
    #             module.weight.data.normal_(mean=0.0, std=1.0)
    #             if module.bias is not None:
    #                 module.bias.data.zero_()

    def teachers_forward(self, x, t, y=None):
        with torch.no_grad():
            for teacher in self.teachers:
                if y is not None:
                    _ = teacher(x, t, y=y)
                else:
                    _ = teacher(x, t)

    def student_forward(self, x, t, y=None):
        if y is not None:
            return self.student(x, t, y=y)
        else:
            return self.student(x, t)

    def forward(self, x, t, y=None):
        if self.training:
            self.teachers_forward(x, t, y)
            out = self.student_forward(x, t, y)
            loss = torch.stack(self.dist_hook.loss()).mean() if self.dist_hook is not None else 0
            # print("RED loss: ", loss)
            return out, loss
        return self.student_forward(x, t, y)


# class ReED_DeepCopy(ReED_Diffuser):
#     def __init__(self, reed_model):
#         print("DeepCopy Model...")
#         super().__init__(None, None, None, None)
#         print("Copy Done")
#         self.student = copy.deepcopy(reed_model.student)
#         self.teachers = reed_model.teachers
#         self.dist_modules = reed_model.dist_modules
#         self.dist_hook = reed_model.dist_hook

    # def forward(self, x, t):
    #     # print('x: ', x.shape, 't: ', t.shape, self.training, self.dist_modules.training)
    #     if self.training:
    #         self.teachers_forward(x, t)
    #         out = self.student_forward(x, t)
    #         loss = torch.stack(self.dist_hook.loss()).mean() if self.dist_hook is not None else 0
    #         return out, loss
    #     return self.student_forward(x, t)


if __name__ == "__main__":
    import sys

    sys.path.append('/home/fangchen/WorkDir/Residual-Encoded-Distillation/')
    from base.network import arch_name_parse, Network
    from torchvision.models import mobilenet_v2

    # teacher = Network(arch_name_parse('mobilenetv2-s1-s1s2s2s2s1s2s1'))
    teacher = mobilenet_v2()
    student = Network(arch_name_parse('resnext18-s8m1-s1s2s1s1'))

    dist_config = {
        'net.pre_layer.0.aggressive_pool': {
            't_layers': [None, 'tid_0=features.4.conv.1.0'],
            'modules': [None, 'RED'],
            'kwargs': [None, {'cs': 64}],
        },
        'net.layer2.0.pooling.aggressive_pool': {
            't_layers': ['tid_1=features.7.conv.1.0', 'tid_1=features.7.conv.1.0'],
            'modules': ['RED', 'RED'],
            'kwargs': [{'cs': 64}, {'cs': 128}]
        }
    }
    model = ReED(student, [teacher, teacher], dist_config, dummy_input=torch.Tensor(torch.randn(1,3,224,224)))

    x = torch.randn(8, 3, 224, 224)
    model.train()
    model(x)
    model.eval()
    model(x)