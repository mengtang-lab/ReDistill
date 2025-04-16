import torch
import torch.nn as nn
import torch.nn.functional as F

def build_dist_module(dist_method, **kwargs):
	dist_module_zoo = {
		'RED': ResidualEncodedModule,
		'STAGE': StageAlign,
		'KD': KnowledgeDistillationModule,
	}
	dist_method =  dist_method.upper()
	if dist_method not in dist_module_zoo.keys(): raise NotImplementedError(f"Unkown Distillation Module: {dist_method}!")
	return dist_module_zoo[dist_method](**kwargs)


class KnowledgeDistillationModule(nn.Module):
	def __init__(self, temperature=2, reduced_dim=-1, alpha=1.):
		super(KnowledgeDistillationModule, self).__init__()
		self.T = temperature
		self.RD = reduced_dim
		self.alpha = alpha
	def forward(self, xs, xt=None):
		if self.training and xt is not None:
			loss = F.kl_div(F.log_softmax(xs/self.T, dim=self.RD), F.softmax(xt/self.T, dim=self.RD), reduction='batchmean') * (self.T**2)
			# print(f"KD loss: {loss}")
			# print(xs.shape, xt.shape)
			return xs, self.alpha * loss
		return xs

class StageAlign(nn.Module):
	def __init__(self, cs, distance = "cosine", alpha = 1.):
		super(StageAlign, self).__init__()

		self.logit = nn.Sequential(
			nn.Conv2d(cs, cs, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(cs),
			nn.Sigmoid()
		)
		self.residual_encoder = nn.Sequential(
			nn.Conv2d(cs, cs, kernel_size=3, stride=1, padding=1), # nn.Conv2d(cs, cs, kernel_size=1, stride=1, padding=0, groups=cs),
			nn.BatchNorm2d(cs),
			nn.ReLU6(inplace=True)
		)

		self.distance = distance
		self.alpha = alpha

	def forward(self, xs, xt=None):
		b, cs, h, w = xs.shape
		# # ablation study
		# output = self.residual_encoder(self.logit(xs))
		# xs_res = self.residual_encoder(xs * self.logit(xs))
		# output = xs + xs_res

		# final version
		xs_res =  self.residual_encoder(xs)
		output = xs * self.logit(xs) + xs_res

		if self.training and xt is not None:
			# print(xs.shape, xt.shape)
			assert xs.shape[:-2] == xt.shape[:-2]

			if self.distance.lower() == "mse" or self.distance.lower() == "l2":
				loss = F.mse_loss(output.mean(dim=(-2,-1)), xt.mean(dim=(-2,-1)), reduction='mean')
			elif self.distance.lower() == "cosine":
				loss = (1 - F.cosine_similarity(output.mean(dim=(-2,-1)).reshape(b,-1), xt.mean(dim=(-2,-1)).reshape(b,-1), dim=1)).mean()
			else:
				raise NotImplementedError(f"Stage Align -- Unkown Distance Measurement (mse/l2, cosine): {self.distance}!")
			# print(f"RED loss: {loss}, Rescale loss: {50 * loss}")
			return output, self.alpha * loss
		return output

class ResidualEncodedModule(nn.Module):
	def __init__(self, cs, distance = "cosine", alpha = 1., logit_ks=1, res_ks=3):
		super(ResidualEncodedModule, self).__init__()

		self.logit = nn.Sequential(
			nn.Conv2d(cs, cs, kernel_size=logit_ks, stride=1, padding=logit_ks//2),
			nn.BatchNorm2d(cs),
			nn.Sigmoid()
		)
		self.residual_encoder = nn.Sequential(
			nn.Conv2d(cs, cs, kernel_size=res_ks, stride=1, padding=res_ks//2), # nn.Conv2d(cs, cs, kernel_size=1, stride=1, padding=0, groups=cs),
			nn.BatchNorm2d(cs),
			nn.ReLU6(inplace=True)
		)

		self.distance = distance
		self.alpha = alpha

	# 	self.apply(self._zero_init_red_block)
	#
	# def _zero_init_red_block(self, module):
	# 	if isinstance(module, nn.Conv2d):
	# 		module.weight.data.zero_()
	# 		if module.bias is not None:
	# 			module.bias.data.zero_()

	def forward(self, xs, xt=None):
		b, cs, h, w = xs.shape
		# # # ablation study
		# output = xs + self.residual_encoder(xs) # w/o LM
		# output = xs * self.logit(xs) # w/o RE
		# output = self.residual_encoder(self.logit(xs)) # w/o Shortcut
		# output = xs # w/o RED block

		# # # final version
		xs_res =  self.residual_encoder(xs)
		output = xs * self.logit(xs) + xs_res

		if self.training and xt is not None:
			# print(xs.shape, xt.shape)
			assert xs.shape[-2:] == xt.shape[-2:]
			if self.distance.lower() == "mse" or self.distance.lower() == "l2":
				loss = F.mse_loss(output.mean(dim=1), xt.mean(dim=1), reduction='mean')
			elif self.distance.lower() == "cosine":
				loss = (1 - F.cosine_similarity(output.mean(dim=1).reshape(b,-1), xt.mean(dim=1).reshape(b,-1), dim=1)).mean()
			else:
				raise NotImplementedError(f"Residual-Encoded-Distillation -- Unkown Distance Measurement (mse/l2, cosine): {self.distance}!")
			# print(f"RED loss: {loss}, Rescale loss: {50 * loss}")
			return output, self.alpha * loss
		return output

