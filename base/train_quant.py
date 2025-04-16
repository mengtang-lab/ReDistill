import warnings
warnings.filterwarnings("ignore")

import argparse

import tracemalloc

import torch
import torch.nn as nn

import sys
import datetime
import time
import os
import numpy as np
import json

from copy import deepcopy
from tqdm import tqdm

from backbones import Network
from utils import get_dataset, load_pretrained_weight, AverageMeter, compute_mac, RunManager


import neural_compressor
import intel_extension_for_pytorch as ipex
from neural_compressor.torch.quantization import prepare, convert
from neural_compressor import quantization, PostTrainingQuantConfig

import inq

from reed import ReED


def measure_latency(model, inputs, device='cpu', latency_repetitions=300):
	'''
        latency_repetitions: number of repetitions for inference
        model: torch model
        inputs: torch tensor
        device: cuda | cpu
    '''
	inputs = tuple([input.to(device) for input in inputs])
	model.to(device)
	model.eval()

	# INIT LOGGERS
	timings = np.zeros((round(latency_repetitions), 1))
	# GPU-WARM-UP
	for _ in range(int(0.1 * latency_repetitions)):
		_ = model(*inputs)
	# MEASURE PERFORMANCE
	with torch.no_grad():
		for rep in range(latency_repetitions):
			start_time = time.perf_counter()
			_ = model(*inputs)
			end_time = time.perf_counter()
			curr_time = (end_time - start_time) * 1000
			timings[rep] = curr_time

	mean_syn = np.sum(timings) / latency_repetitions
	std_syn = np.std(timings)
	print("Latency: {:.4f} +/- {:.4f} ms/{} images".format(mean_syn, std_syn, inputs[0].shape[0]))


def actual_peak_memory(model, inputs, device='cpu', latency_repetitions=300):
	'''
        latency_repetitions: number of repetitions for inference
        model: torch model
        inputs: torch tensor
        device: cuda | cpu
    '''
	inputs = tuple([input.to(device) for input in inputs])
	model.to(device)
	model.eval()

	# INIT LOGGERS
	record = np.zeros((round(latency_repetitions), 1))
	# GPU-WARM-UP
	for _ in range(int(0.1 * latency_repetitions)):
		_ = model(*inputs)
	# MEASURE PERFORMANCE
	with torch.no_grad():
		for rep in range(latency_repetitions):
			tracemalloc.start()

			_ = model(*inputs)

			_, peak = tracemalloc.get_traced_memory()
			tracemalloc.stop()
			record[rep] = peak / (1024 ** 2)

	mean_syn = np.sum(record) / latency_repetitions
	std_syn = np.std(record)
	print("Peak Memory: {:.4f} +/- {:.4f} MB/{} images".format(mean_syn, std_syn, inputs[0].shape[0]))

def measure_modelsize(model, jit_save=True):
	if jit_save:
		model.eval()
		traced_model = torch.jit.trace(model, (torch.randn(1, 3, 128, 128),))
		traced_model = torch.jit.freeze(traced_model)
		torch.jit.save(traced_model, './ms_temp.pt')
	else:
		torch.save(model, 'ms_temp.pt')

	size = os.path.getsize('./ms_temp.pt')
	os.remove('./ms_temp.pt')
	print('Model Size: {:.3f} MB'.format(size / 1024**2))
	return size / 1024**2


class PTQ_Quantizer(object):
	def __init__(self, model_fp, calib_loader, eval_loader, example_inputs):

		self.model_fp = model_fp
		# self.fp_size = measure_modelsize(model_fp)
		self.calib_loader = calib_loader
		self.eval_loader = eval_loader
		self.example_inputs = example_inputs

	def eval_func(self, model, calibrate=True):
		if calibrate: loader = self.calib_loader
		else: loader = self.eval_loader
		top1   = AverageMeter('Acc@1')

		with torch.no_grad():
			model.eval()
			with tqdm(loader, total=len(loader)) as t:
				for batch_idx, (data, target) in enumerate(t):
					if torch.cuda.is_available():
						data, target = data.cuda(), target.cuda()

					output = model(data)

					pred = output.max(1, keepdim=True)[1]
					correct = pred.eq(target.data.view_as(pred)).sum()

					top1.update(correct.item()/data.size(0), data.size(0))

					if batch_idx % 1 == 0:
						t.set_postfix_str("test_acc: {:.4f}".format(top1.avg))

		print('Acc@1 {:.4f}'.format(top1.avg))
		# size = measure_modelsize(model)
		# print(f"Model Size: {size} / FP Size: {self.fp_size}")
		# print('Model Size: {:.3f} MB'.format(size / 1024 ** 2))
		# if abs(size - self.fp_size) < 0.1: return top1.avg + 1
		return top1.avg
		# return size

	def ptq(self, fp_key=[], int_key=[]):
		from neural_compressor.config import TuningCriterion, AccuracyCriterion
		tuning_criterion = TuningCriterion(timeout=100,
										   max_trials=10,
										   strategy="basic")
		# conf = PostTrainingQuantConfig(backend="default",
		# 							   # quant_level=1,
		# 							   # tuning_criterion=tuning_criterion,
		# 							   op_name_dict={
		# 								   "module.net.conv1": {"activation": {"dtype": ["fp32"]},
		# 														"weight": {"dtype": ["fp32"]}},
		# 							   }
		# 							   )
		# op_name_dict = {"net.*":{"activation": {"dtype": ["fp32", "fp16", "int8"]}, "weight": {"dtype": ["fp32", "fp16", "int8"]}}}
		op_name_dict = {}
		for key in fp_key:
			op_name_dict[key] = {"activation": {"dtype": ["fp32"]}, "weight": {"dtype": ["fp32"]}}
		for key in int_key:
			op_name_dict[key] = {"activation": {"dtype": ["int8"]}, "weight": {"dtype": ["int8"]}}
		qconfig = PostTrainingQuantConfig(quant_level=1,
										  tuning_criterion=tuning_criterion,
										  op_name_dict=op_name_dict)
		print(self.model_fp)
		q_model = quantization.fit(self.model_fp,
								   qconfig,
								   calib_dataloader=self.calib_loader,
								   eval_dataloader=self.calib_loader,
								   eval_func=self.eval_func)
		print(q_model, op_name_dict)
		# convert(q_model, )
		return q_model

	def ptq_v2(self):
		self.model_fp.eval()
		import intel_extension_for_pytorch as ipex
		from intel_extension_for_pytorch.quantization import prepare, convert, default_dynamic_qconfig
		# qconfig = ipex.quantization.default_static_qconfig
		from torch.ao.quantization import MinMaxObserver, PlaceholderObserver, PerChannelMinMaxObserver, QConfig
		qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
						  weight=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8))

		prepared_model = ipex.quantization.prepare(self.model_fp, qconfig, example_inputs=self.example_inputs)
		self.eval_func(prepared_model, calibrate=True)
		q_model = ipex.quantization.convert(prepared_model)
		# print(q_model)
		# traced_model = torch.jit.trace(convert_model, self.example_inputs)
		# quantized_model = torch.jit.freeze(traced_model)

		# torch.jit.save(quantized_model, './quantized_model.pt')
		# q_model = torch.jit.load('./quantized_model.pt')
		# q_model = torch.jit.freeze(q_model.eval())

		return q_model




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
	parser.add_argument('-p', '--pretrained', default='', type=str, help='pretrained student model')
	parser.add_argument('-t', '--teacher', default='', type=str, help='teacher network architecture')
	parser.add_argument('--teacher_pretrained', default='', type=str, help='pretrained teacher model')
	parser.add_argument('--dist_config', default='', type=str, help='distillation config')
	parser.add_argument('--dist_pretrained', default='', type=str, help='distillation pretrained model')
	parser.add_argument('--quantize', 				default='ptq', 				type=str, help='quantization mode')


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

	architecture = args.architecture
	pretrained = args.pretrained
	teacher = args.teacher
	teacher_pretrained = args.teacher_pretrained
	dist_config = args.dist_config
	dist_pretrained = args.dist_pretrained

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

	identifier = f'{arch_name}_{args.quantize}_{dataset_name.lower()}_imsize{str(im_size)}_' \
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
	device_ids = list(map(int, devices.replace(' ', '').split(',')))

	student_model = Network(arch_name, image_size=im_size, num_classes=dataset.num_classes)
	load_pretrained_weight(student_model, pretrained, f, convert_from_dataparallel=True)

	teacher_model = Network(teacher.lower(), image_size=im_size, num_classes=dataset.num_classes)
	load_pretrained_weight(teacher_model, teacher_pretrained, f, convert_from_dataparallel=True)

	# print(student_model)
	# print(teacher_model)

	model = ReED(deepcopy(student_model), [deepcopy(teacher_model)], dist_config, dummy_input=torch.randn(1, 3, im_size, im_size))
	# print(model)
	# total_params = sum(p.numel() for p in model.parameters())
	# # print(f'{total_params:,} total parameters.')
	# total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	# # print(f'{total_trainable_params:,} training parameters.')
	#
	# # for name, param in model.named_parameters():
	# # 	if param.requires_grad:
	# # 		print('Trainable:', name)
	#
	# model = nn.DataParallel(model, device_ids=device_ids)
	if dist_pretrained:
		load_pretrained_weight(model, dist_pretrained, f, convert_from_dataparallel=True)

	f.write('\n\n Model: {}'.format(model))

	# print(model)

	if torch.cuda.is_available():
		student_model.cuda()
		teacher_model.cuda()
		model.cuda()

	# # ================================================ Distillation ================================================
	learning_rate = float(learning_rate)
	lr_interval = list(map(lambda x: int(float(x)*epochs), lr_interval.split()))
	lr_reduce = int(lr_reduce)


	if args.quantize == "qat":
		def qat_run(model, distill=False):
			run_manager = RunManager(model, None, learning_rate, lr_interval, lr_reduce, f)
			example_inputs = (torch.randn(1, 3, 128, 128),)
			# measure_modelsize(model, jit_save=False)
			# measure_latency(model, example_inputs, device='cuda:0')

			quantized_parameters = []
			full_precision_parameters = []
			for name, param in model.named_parameters():
				if 'bn' in name or 'bias' in name:
					full_precision_parameters.append(param)
				else:
					quantized_parameters.append(param)
			optimizer = inq.SGD([
				{'params': quantized_parameters},
				{'params': full_precision_parameters, 'weight_bits': None}
			], args.learning_rate, momentum=0.9, weight_decay=0.005, weight_bits=6)

			scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2])
			inq_scheduler = inq.INQScheduler(optimizer, [0.5, 0.75, 0.82, 1.0], strategy="pruning")
			# inq_scheduler = inq.INQScheduler(optimizer, [0.1, 0.5, 1.0], strategy="pruning")

			run_manager.optimizer = optimizer

			for inq_step in range(3):  # Iteration for accumulated quantized weights of 50% 75% and 82%
				inq.reset_lr_scheduler(scheduler)
				inq_scheduler.step()
				for epoch in range(5):
					scheduler.step()

					start_time = datetime.datetime.now()
					run_manager.set_start_time(start_time)

					if not distill:
						run_manager.train(epoch, dataset.train_loader(batch_size))
					else:
						print("Distillation!!!")
						run_manager.distillation(epoch, dataset.train_loader(batch_size))

					save = True
					ann_path = f'{root}/results/trained_models_ann/{folder_name}/{arch_name}/'
					train_identifier = identifier
					run_manager.test(epoch, dataset.test_loader(batch_size), save=save, ann_path=ann_path,
									 identifier=train_identifier)

				# measure_modelsize(run_manager.model, jit_save=False)
				# measure_latency(run_manager.model, example_inputs, device='cuda:0')

			inq_scheduler.step()  # quantize all weights, further training is useless

			# measure_modelsize(run_manager.model, jit_save=False)
			# measure_latency(run_manager.model, example_inputs, device='cuda:0')
			run_manager.test(1, dataset.test_loader(batch_size), save=save, ann_path=ann_path,
							 identifier=train_identifier)

		print("=" * 10 + " Teacher Model " + "=" * 10)
		qat_run(teacher_model)
		print("=" * 10 + " Student Model " + "=" * 10)
		qat_run(student_model)
		print("=" * 10 + " RED Model " + "=" * 10)
		qat_run(model, distill=True)


	if args.quantize == "ptq":
		example_inputs = (torch.randn(1, 3, 128, 128),)
		from torch.utils.data import Subset
		from torch.utils.data import DataLoader

		subset_indices = range(0, 200)  # Define your slice indices
		subset = Subset(dataset.test_dataset, subset_indices)
		calib_dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

		# # teacher model
		print("=" * 10 + " Teacher Model " + "=" * 10)
		measure_modelsize(teacher_model)
		# measure_latency(teacher_model, example_inputs, device='cpu')
		actual_peak_memory(teacher_model, example_inputs, device='cpu')

		# quantizer = PTQ_Quantizer(model, dataset.test_loader(batch_size), example_inputs)
		quantizer = PTQ_Quantizer(model.teachers[0], calib_dataloader, dataset.test_loader(batch_size), example_inputs)
		q_model = quantizer.ptq(fp_key=["net.conv1.*"],
								int_key=["net.layer1.*", "net.layer2.*", "net.layer3.*", "net.layer4.*", "net.fc"])
		# q_model = quantizer.ptq(fp_key=["net.conv1.*", "net.layer1.*"],
		# 						int_key=["net.fc"])
		# q_model = quantizer.ptq_v2()

		measure_modelsize(q_model)
		# measure_latency(q_model, example_inputs, device='cpu')
		actual_peak_memory(q_model, example_inputs, device='cpu')

		quantizer.eval_func(q_model, calibrate=False)


		# # student model
		print("=" * 10 + " Student Model " + "=" * 10)
		measure_modelsize(student_model)
		# measure_latency(student_model, example_inputs, device='cpu')
		actual_peak_memory(student_model, example_inputs, device='cpu')

		# quantizer = PTQ_Quantizer(model, dataset.test_loader(batch_size), example_inputs)
		quantizer = PTQ_Quantizer(student_model, calib_dataloader, dataset.test_loader(batch_size), example_inputs)
		# q_model = quantizer.ptq(fp_key=["net.conv1", "net.layer2.*", "net.layer3.*"], int_key=["net.fc"])
		q_model = quantizer.ptq(fp_key=["net.conv1.*"],
								int_key=["net.layer.*", "net.fc"])
		# q_model = quantizer.ptq_v2()

		measure_modelsize(q_model)
		# measure_latency(q_model, example_inputs, device='cpu')
		actual_peak_memory(q_model, example_inputs, device='cpu')

		quantizer.eval_func(q_model, calibrate=False)


		# # RED model
		print("=" * 10 + " RED Model " + "=" * 10)
		model.graduate()
		measure_modelsize(model)
		# measure_latency(model, example_inputs, device='cpu')
		actual_peak_memory(model, example_inputs, device='cpu')

		# quantizer = PTQ_Quantizer(model, dataset.test_loader(batch_size), example_inputs)
		quantizer = PTQ_Quantizer(model, calib_dataloader, dataset.test_loader(batch_size), example_inputs)
		# q_model = quantizer.ptq(fp_key=["student.net.fc"],
		# 						int_key=["student.net.conv1",
		# 								 "student.net.layer1.*",
		# 								 "student.net.layer2.*",
		# 								 "student.net.layer3.*",
		# 								 "student.net.layer4.*",
		# 								 "dist_modules.*"])
		q_model = quantizer.ptq(fp_key=["student.net.conv1.*"], # "student.net.conv1.*"
								int_key=[])
		# q_model = quantizer.ptq_v2()

		measure_modelsize(q_model)
		# measure_latency(q_model, example_inputs, device='cpu')
		actual_peak_memory(q_model, example_inputs, device='cpu')

		quantizer.eval_func(q_model, calibrate=False)

# # compute_mac(model, dataset.im_size, f)
	# for epoch in range(1, epochs+1):
	# 	start_time = datetime.datetime.now()
	# 	run_manager.set_start_time(start_time)
	# 	if not test_only:
	# 		run_manager.distillation(epoch, dataset.train_loader(batch_size))
	#
	# 	if not args.dont_save or not args.test_only:
	# 		save = True
	# 		ann_path = f'{root}/results/trained_models_ann/{folder_name}/{arch_name}/'
	# 		train_identifier = identifier
	# 	else:
	# 		save = False
	# 		ann_path = None
	# 		train_identifier = None
	# 	run_manager.test(epoch, dataset.test_loader(batch_size), save=save, ann_path=ann_path, identifier=train_identifier)
	#
	# f.write('\n Highest accuracy: {:.4f}'.format(run_manager.max_accuracy))
