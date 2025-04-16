# def ModelSize(func):
#     def decorator(**kwargs):
#         model = kwargs["model"]
#         dummy_input = kwargs["inputs"][0]
#         param_size = 0
#         for param in model.parameters():
#             param_size += param.nelement() * param.element_size()
#         buffer_size = 0
#         for buffer in model.buffers():
#             buffer_size += buffer.nelement() * buffer.element_size()
#         size_all_mb = (param_size + buffer_size) / 1024 ** 2
#         print('Model Size: {:.3f} MB'.format(size_all_mb))
#         return func(**kwargs)
#     return decorator

import torch
import os

def ModelSize(func):
    def decorator(**kwargs):
        model = kwargs["model"]
        dummy_input = kwargs["inputs"][0]
        torch.save(model.state_dict(), './ms_temp.pth')
        size = os.path.getsize('./ms_temp.pth')
        os.remove('./ms_temp.pth')
        print('Model Size: {:.3f} MB'.format(size / 1024**2))
        return func(**kwargs)
    return decorator