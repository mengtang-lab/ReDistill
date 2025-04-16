import torch
import torch.nn as nn
import pdb


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)


def compute_mac(model, im_size, log_file=None):
	h_in, w_in = im_size

	macs = []
	for name, l in model.named_modules():
		if isinstance(l, nn.Conv2d):
			c_in    = l.in_channels
			k       = l.kernel_size[0]
			h_out   = int((h_in-k+2*l.padding[0])/(l.stride[0])) + 1
			w_out   = int((w_in-k+2*l.padding[0])/(l.stride[0])) + 1
			c_out   = l.out_channels
			mac     = k*k*c_in*h_out*w_out*c_out
			if mac == 0:
				pdb.set_trace()
			macs.append(mac)
			h_in    = h_out
			w_in    = w_out
			print('{}, Mac:{}'.format(name, mac))
		if isinstance(l, nn.Linear):
			mac     = l.in_features * l.out_features
			macs.append(mac)
			print('{}, Mac:{}'.format(name, mac))
		if isinstance(l, nn.AvgPool2d):
			h_in    = h_in//l.kernel_size
			w_in    = w_in//l.kernel_size
	print('Mac: {:e}'.format(sum(macs)))
	exit()
	if log_file is not None:
		log_file.write('\n\n Mac: {}'.format(sum(macs)))


def conver_dataparallel_to_state_dict(state_dict):
	new_state_dict = {}
	for key, value in state_dict.items():
		if key.startswith('module.'):
			new_state_dict[key.split('module.')[1]] = value
	return new_state_dict


def load_pretrained_weight(model, pretrained_file, log_file=None, convert_from_dataparallel=False):
	'''
	state=torch.load(args.pretrained_ann, map_location='cpu')
	cur_dict = model.state_dict()
	for key in state['state_dict'].keys():
		if key in cur_dict:
			if (state['state_dict'][key].shape == cur_dict[key].shape):
				cur_dict[key] = nn.Parameter(state[key].data)
				f.write('\n Success: Loaded {} from {}'.format(key, pretrained_ann))
			else:
				f.write('\n Error: Size mismatch, size of loaded model {}, size of current model {}'.format(state['state_dict'][key].shape, model.state_dict()[key].shape))
		else:
			f.write('\n Error: Loaded weight {} not present in current model'.format(key))

	#model.load_state_dict(cur_dict)
	'''
	state = torch.load(pretrained_file, map_location='cpu')
	if convert_from_dataparallel: state['state_dict'] = conver_dataparallel_to_state_dict(state['state_dict'])

	missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
	print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
	print('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))
	if log_file is not None:
		log_file.write('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
		log_file.write('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))
	return model