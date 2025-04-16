import os
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .tools import AverageMeter, compute_mac


class RunManager(object):
	def __init__(self, model, optimizer, learning_rate, lr_interval, lr_reduce, log_file=None):
		self.model = model
		self.learning_rate = learning_rate
		if optimizer == 'SGD':
			self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate, momentum=0.9, weight_decay=0.000)
		elif optimizer == 'Adam':
			self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate, amsgrad=True, weight_decay=0.000)

		self.lr_interval = lr_interval
		self.lr_reduce = lr_reduce
		self.log_file = log_file
		if self.log_file is not None:
			self.log_file.write('\n\n Optimizer: {}'.format(self.optimizer))
		self.max_accuracy = 0
		self.start_time = None
		self.save_path = None

	def set_start_time(self, time):
		self.start_time = time

	def distillation(self, epoch, loader):
		losses = AverageMeter('Loss')
		top1 = AverageMeter('Acc@1')

		if epoch in self.lr_interval:
			for param_group in self.optimizer.param_groups:
				param_group['lr'] = param_group['lr'] / self.lr_reduce
				self.learning_rate = param_group['lr']

		self.model.train()
		with tqdm(loader, total=len(loader)) as t:
			for batch_idx, (data, target) in enumerate(t):
				if torch.cuda.is_available():
					data, target = data.cuda(), target.cuda()

				output, dist_loss = self.model(data)

				task_loss = F.cross_entropy(output, target)

				alpha = 0.5
				loss = (1 - alpha) * task_loss + alpha * dist_loss

				self.optimizer.zero_grad()
				# loss.backward(inputs = list(self.model.parameters()))
				loss.backward()
				self.optimizer.step()

				pred = output.max(1, keepdim=True)[1]
				truth = target.max(1, keepdim=True)[1]
				correct = pred.eq(truth.data.view_as(pred)).sum()

				losses.update(loss.item(), data.size(0))
				top1.update(correct.item() / data.size(0), data.size(0))

				if batch_idx % 1 == 0:
					# t.set_postfix_str("train_loss: {:.4f}, train_acc: {:.4f}".format(loss.item(), correct.item()/data.size(0)))
					t.set_postfix_str("train_loss: {:.4f}, train_acc: {:.4f}".format(losses.avg, top1.avg))

		self.log_file.write('\n Epoch: {}, lr: {:.1e}, train_loss: {:.4f}, train_acc: {:.4f}'.format(
			epoch,
			self.learning_rate,
			losses.avg,
			top1.avg
		)
		)

	def train(self, epoch, loader):
		losses = AverageMeter('Loss')
		top1   = AverageMeter('Acc@1')

		if epoch in self.lr_interval:
			for param_group in self.optimizer.param_groups:
				param_group['lr'] = param_group['lr'] / self.lr_reduce
				self.learning_rate = param_group['lr']

		self.model.train()
		with tqdm(loader, total=len(loader)) as t:
			for batch_idx, (data, target) in enumerate(t):
				if torch.cuda.is_available():
					data, target = data.cuda(), target.cuda()

				output = self.model(data) # regular model
				task_loss = F.cross_entropy(output, target)
				loss = task_loss

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				pred = output.max(1, keepdim=True)[1]
				truth = target.max(1, keepdim=True)[1]
				correct = pred.eq(truth.data.view_as(pred)).sum()

				losses.update(loss.item(), data.size(0))
				top1.update(correct.item()/data.size(0), data.size(0))

				if batch_idx % 1 == 0:
					# t.set_postfix_str("train_loss: {:.4f}, train_acc: {:.4f}".format(loss.item(), correct.item()/data.size(0)))
					t.set_postfix_str("train_loss: {:.4f}, train_acc: {:.4f}".format(losses.avg, top1.avg))

		self.log_file.write('\n Epoch: {}, lr: {:.1e}, train_loss: {:.4f}, train_acc: {:.4f}'.format(
				epoch,
				self.learning_rate,
				losses.avg,
				top1.avg
				)
			)

	def test(self, epoch, loader, save=False, ann_path=None, identifier=None):
		losses = AverageMeter('Loss')
		top1   = AverageMeter('Acc@1')

		with torch.no_grad():
			self.model.eval()
			total_loss = 0
			correct = 0
			with tqdm(loader, total=len(loader)) as t:
				for batch_idx, (data, target) in enumerate(t):
					if torch.cuda.is_available():
						data, target = data.cuda(), target.cuda()

					output = self.model(data)

					loss = F.cross_entropy(output,target)
					total_loss += loss.item()

					pred = output.max(1, keepdim=True)[1]
					correct = pred.eq(target.data.view_as(pred)).sum()

					losses.update(loss.item(), data.size(0))
					top1.update(correct.item()/data.size(0), data.size(0))

					if batch_idx % 1 == 0:
						t.set_postfix_str("test_loss: {:.4f}, test_acc: {:.4f}".format(losses.avg, top1.avg))

		if top1.avg>self.max_accuracy:
			self.max_accuracy = top1.avg
			state = {
					'accuracy'      : self.max_accuracy,
					'epoch'         : epoch,
					'state_dict'    : self.model.state_dict(),
					'optimizer'     : self.optimizer.state_dict()
			}

			if save==True and ann_path is not None and identifier is not None:
				try:
					os.makedirs(ann_path)
				except OSError:
					pass
				filename = os.path.join(ann_path, f'{identifier}.pth')
				torch.save(state, filename)
				self.save_path = filename

		self.log_file.write(' test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f}, time: {}'.format(
			losses.avg,
			top1.avg,
			self.max_accuracy,
			datetime.timedelta(seconds=(datetime.datetime.now() - self.start_time).seconds)
			)
		)