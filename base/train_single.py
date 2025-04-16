import warnings
warnings.filterwarnings("ignore")

import argparse

import torch
import torch.nn as nn

import sys
import datetime
import os
import numpy as np
import json

from backbones import Network
from utils import get_dataset, load_pretrained_weight, compute_mac, RunManager


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train Network with Single Mode', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# parser.add_argument('--gpu',                  default=True,               type=bool,      help='use gpu')
	parser.add_argument('-r','--root',              default='./',               type=str,       help='root direction')
	parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
	parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')

	parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['CIFAR10','CIFAR100', 'STL10'])
	parser.add_argument('--im_size',                default=None,               type=int,       help='image size')
	parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')

	parser.add_argument('-a','--architecture',      default='',               type=str,       help='network architecture')
	parser.add_argument('--pretrained',             default='',                 type=str,       help='pretrained model to initialize Network')

	parser.add_argument('--epochs',                 default=300,                type=int,       help='number of training epochs')
	parser.add_argument('-lr','--learning_rate',    default=1e-2,               type=float,     help='initial learning_rate')
	parser.add_argument('--lr_interval',            default='0.45 0.70 0.90',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
	parser.add_argument('--lr_reduce',              default=10,                 type=int,       help='reduction factor for learning rate')
	parser.add_argument('--optimizer',              default='SGD',              type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])

	parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
	parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
	parser.add_argument('--dont_save',              action='store_true',                        help='don\'t save training model during testing')

	args = parser.parse_args()

	# os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

	# Seed random number
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


	root            = args.root
	devices         = args.devices

	dataset_name    = args.dataset
	im_size 		= args.im_size
	batch_size      = args.batch_size

	architecture    = args.architecture
	pretrained      = args.pretrained

	epochs          = args.epochs
	learning_rate   = args.learning_rate
	lr_interval     = args.lr_interval
	lr_reduce       = args.lr_reduce
	optimizer       = args.optimizer

	log             = args.log
	test_only       = args.test_only
	dont_save       = args.dont_save

	# # ================================================ Log File ================================================
	arch_name = architecture.lower()

	arch_name_list = arch_name.split('-')
	backbone_name = arch_name_list[0]

	folder_name = backbone_name
	log_file = os.path.join(root, 'results', 'logs_new', folder_name)
	try:
		os.makedirs(log_file)
	except OSError:
		pass

	identifier = f'{arch_name}_{dataset_name.lower()}_imsize{str(im_size)}_' \
				 f'batchsize{str(batch_size)}_lr{learning_rate}_optimizer{optimizer}'
	log_file = os.path.join(log_file, f'{identifier}.log')

	print(f'log file saves to: {log_file}')

	if log:
		f = open(log_file, 'w', buffering=1)
	else:
		f = sys.stdout

	f.write('\n Run on time: {}'.format(datetime.datetime.now()))

	f.write('\n\n Architecture: {}'.format(arch_name))

	f.write('\n\n Arguments:')
	for arg in vars(args):
		f.write('\n\t {:20} : {}'.format(arg, getattr(args, arg)))

	# # ================================================ Data & Model ================================================
	dataset = get_dataset(dataset_name, im_size=im_size)

	model = Network(arch_name, image_size=im_size, num_classes=dataset.num_classes)

	total_params = sum(p.numel() for p in model.parameters())
	print(f'{total_params:,} total parameters.')
	total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'{total_trainable_params:,} training parameters.')

	device_ids = list(map(int, devices.replace(' ', '').split(',')))
	model = nn.DataParallel(model, device_ids=device_ids)

	for name, param in model.named_parameters():
		if param.requires_grad:
			print('Trainable:', name)

	if pretrained:
		load_pretrained_weight(model, pretrained, f)

	f.write('\n\n Model: {}'.format(model))

	if torch.cuda.is_available():
		model.cuda()

	# # ================================================ Optimizer ================================================
	learning_rate = float(learning_rate)
	lr_interval = list(map(lambda x: int(float(x)*epochs), lr_interval.split()))
	lr_reduce = int(lr_reduce)
	run_manager = RunManager(model, optimizer, learning_rate, lr_interval, lr_reduce, f)

	# compute_mac(model, dataset.im_size, f)
	for epoch in range(1, epochs+1):
		start_time = datetime.datetime.now()
		run_manager.set_start_time(start_time)
		if not test_only:
			run_manager.train(epoch, dataset.train_loader(batch_size))

		if not args.dont_save or not args.test_only:
			save = True
			ann_path = f'{root}/results/trained_models_ann/{folder_name}/{arch_name}/'
			train_identifier = identifier
		else:
			save = False
			ann_path = None
			train_identifier = None
		run_manager.test(epoch, dataset.test_loader(batch_size), save=save, ann_path=ann_path, identifier=train_identifier)

	f.write('\n Highest accuracy: {:.4f}'.format(run_manager.max_accuracy))
