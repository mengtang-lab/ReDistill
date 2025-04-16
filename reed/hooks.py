import sys

from typing import List, Dict, Tuple

import torch
import torch.nn as nn


class HookHelper:
    verbose = False
    def __init__(self):
        pass

    @staticmethod
    def print_modules_name(model):
        for name, module in model.named_modules():
            print(name)

    @staticmethod
    def get_activation(name, activation_dict: dict):
        def hook(module, input, output):
            if len(input) == 0: input_ = None
            else: input_ = input[0]
            activation_dict[name]['activation'] = {'input': input_, 'output': output}
        return hook

    @staticmethod
    def calibrate(model, dummy_input: torch.Tensor):
        # print(dummy_input.shape, dummy_input.device)
        if isinstance(dummy_input, torch.Tensor):
            _ = model(dummy_input)
        elif isinstance(dummy_input, dict):
            _ = model(**dummy_input)
        else:
            raise NotImplementedError(f"Unsupported input: {dummy_input}!")
        # print(_)


class BaseHook(HookHelper):
    def __init__(self):
        super(BaseHook, self).__init__()
        self.layer_table = {}

    def __print_layer_table(self):
        for key, info in self.layer_table.items():
            print(f"{key}: module={info['module']}, "
                  f"inp_shape={info['activation']['input'].shape}, "
                  f"oup_shape={info['activation']['output'].shape}")

    def set_layer_info(self, model, layer_name_list, dummy_input:torch.Tensor = None):
        module_dict = {}
        for name, module in model.named_modules():
            module_dict[name] = module
        for name in layer_name_list:
            if name not in module_dict.keys(): raise RuntimeError(f'The layer {name} is not a valid module!')
            module = module_dict[name]
            self.layer_table[name] = {'module': module,
                                      'activation': None}
            # print(module)
            module.register_forward_hook(self.get_activation(name, self.layer_table))
        if dummy_input is not None: self.calibrate(model, dummy_input)
        if self.verbose: self.__print_layer_table()


class DistillationHook(HookHelper):
    """
    dist_table = {'s_layer': {
        'input': {
            'teacher_id': 't_id',
            'teacher_layer': 'inp_t_layer',
            'dist_module': 'inp_dist_module',
            'dist_loss': 'inp_dist_loss'
        },
        'output': {
            'teacher_id': 't_id',
            'teacher_layer': 'oup_t_layer',
            'dist_module': 'oup_dist_module',
            'dist_loss': 'oup_dist_loss'
        }
    }}
    """
    def __init__(self, student:nn.Module, teachers:List[nn.Module] or nn.ModuleList, dist_table:dict, dummy_input:torch.Tensor or Dict[str, torch.Tensor] = None):
        super(DistillationHook, self).__init__()
        self.dist_table = dist_table

        t_layer_name_dict = {id:[] for id in range(len(teachers))}
        for key, item in self.dist_table.items():
            inp, oup = item['input'], item['output']
            if inp is not None: t_layer_name_dict[inp['teacher_id']].append(inp['teacher_layer'])
            if oup is not None: t_layer_name_dict[oup['teacher_id']].append(oup['teacher_layer'])

        self.t_hooks = {}
        for i, teacher in enumerate(teachers):
            t_hook = BaseHook()
            t_hook.set_layer_info(teacher, t_layer_name_dict[i], dummy_input=dummy_input)
            self.t_hooks[i] = t_hook
        self.modify_forward_hook(student, dummy_input=dummy_input)

    def __print_dist_table(self):
        for key, item in self.dist_table.items():
            f_str = f"{key}: "
            inp_dict, oup_dict = item['input'], item['output']
            if inp_dict is not None:
                f_str += f"input=[from:teacher_{inp_dict['teacher_id']}-{inp_dict['teacher_layer']}, " \
                         f"dist_module:{inp_dict['dist_module']}, dist_loss:{inp_dict['dist_loss']}]\n"
            if oup_dict is not None:
                f_str += f"output=[from:teacher_{oup_dict['teacher_id']}-{oup_dict['teacher_layer']}, " \
                         f"dist_module:{oup_dict['dist_module']}, dist_loss:{oup_dict['dist_loss']}]\n"
            print(f_str)

    def loss(self):
        losses = []
        for name, item in self.dist_table.items():
            if item['input'] is not None: losses.append(item['input']['dist_loss'])
            if item['output'] is not None: losses.append(item['output']['dist_loss'])
        return losses

    def __forward(self, module, xs, xt):
        if module.training: # distillation module training
            output, loss = module(xs, xt)
            return output, loss
        else: # distillation module inference
            output = module(xs)
            return output, 0

    def __get_t_act(self, query):
        return self.t_hooks[query['teacher_id']].layer_table[query['teacher_layer']]['activation']

    def __modify_inp_forward(self, name):
        def hook(model, input):
            xs = input[0]
            input_dict = self.dist_table[name]['input']
            if input_dict is not None:
                xt = self.__get_t_act(input_dict)['input']
                # print('input:', xs.shape, xt.shape)
                module = input_dict['dist_module']
                xs, loss = self.__forward(module, xs, xt)
                input_dict['dist_loss'] = loss

            if len(input) == 1: return xs
            return tuple([xs] + list(input[1:]))
        return hook

    def __modify_oup_forward(self, name):
        def hook(model, input, output):
            xs = output
            output_dict = self.dist_table[name]['output']
            if output_dict is not None:
                xt = self.__get_t_act(output_dict)['output']
                # print('output:', xs.shape, xt.shape)
                module = output_dict['dist_module']
                xs, loss = self.__forward(module, xs, xt)
                output_dict['dist_loss'] = loss
            return xs.clone()
        return hook

    def modify_forward_hook(self, model, dummy_input:torch.Tensor = None):
        module_dict = {}
        for name, module in model.named_modules():
            module_dict[name] = module
        for name in self.dist_table.keys():
            if name not in module_dict.keys(): raise RuntimeError(f'The layer {name} is not a valid module!')
            module = module_dict[name]
            # print(module)
            module.register_forward_pre_hook(self.__modify_inp_forward(name))
            module.register_forward_hook(self.__modify_oup_forward(name))
        if dummy_input is not None: self.calibrate(model, dummy_input)
        if self.verbose: self.__print_dist_table()