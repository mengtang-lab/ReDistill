# from dd_code.backdoor.benchmarks.pytorch-ddpm.main import self
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
from tqdm import tqdm
import ipdb
from functools import partial

from torchvision.utils import save_image
import string


class GaussianDiffusionSamplerDDIM(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='epsilon', var_type='fixedlarge', omega=2, omega_scheduler='constant', gamma=100, cond=False):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.omega = omega
        self.omega_scheduler_method = omega_scheduler
        self.gamma = gamma
        self.cond = cond
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer(
            'alphas_bar', alphas_bar)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t,
                        method='ddpm',
                        skip=1,
                        eps=None):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        if method == 'ddim':
            assert (eps is not None)
            skip_time = torch.clamp(t - skip, 0, self.T)
            posterior_mean_coef1 = torch.sqrt(extract(self.alphas_bar, t, x_t.shape))
            posterior_mean_coef2 = torch.sqrt(1-extract(self.alphas_bar, t, x_t.shape))
            posterior_mean_coef3 = torch.sqrt(extract(self.alphas_bar, skip_time, x_t.shape))
            posterior_mean_coef4 = torch.sqrt(1-extract(self.alphas_bar, skip_time, x_t.shape))
            posterior_mean = (
                posterior_mean_coef3 / posterior_mean_coef1 * x_t +
                (posterior_mean_coef4 - 
                posterior_mean_coef3 * posterior_mean_coef2 / posterior_mean_coef1) * eps
            )
        else:
            posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)

        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t, y, method, skip):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t, y)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t, y)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            if self.cond:
                eps = self.model(x_t, t, y)
                eps_g = self.model(x_t, t, None)

                eps = eps + (self.omega_t) * (eps - eps_g)
                # print(torch.autograd.grad(eps, y))
                x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps) # score

                model_mean, _ = self.q_mean_variance(x_0, x_t, t, method, skip, eps)
            else:
                #ipdb.set_trace()
                eps = self.model(x_t, t, None)
                x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
                model_mean, _ = self.q_mean_variance(x_0, x_t, t, method, skip, eps)
                #print("un conditional!")
        else:
            raise NotImplementedError(self.mean_type)
        #x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var, x_0

    def omega_scheduler(self, t):
        # self.omega_t \in (-1, omega)
        if self.omega_scheduler_method == 'constant':
            self.omega_t = self.omega
        elif self.omega_scheduler_method == 'linear':
            self.omega_t = (1 - t/self.T) * (self.omega + 1) - 1
        elif self.omega_scheduler_method == 'gamma':
            if self.gamma >= 1:
                self.omega_t = (np.power(self.gamma, 1 - t/self.T) / self.gamma) * (self.omega + 1) - 1
            if self.gamma < 1:
                self.omega_t = (1 - np.power(self.gamma, 1 - t / self.T)) * (self.omega + 1) - 1
        else:
            raise NotImplementedError(self.omega_scheduler_method)
        return self.omega_t

    def forward(self, x_T, y, method='ddim', skip=10, return_intermediate=False):
        """
        Algorithm 2.
            - method: sampling method, default='ddpm'
            - skip: decrease sampling steps from T/skip, default=1
        """
        print(f'Gaussian Diffusion Sampler DDIM, Omega Scheduler: {self.omega_scheduler_method}, Sampling Method: {method}.')
        x_t = x_T
        if return_intermediate:
            xt_list = []
            omega_t_list = []

        for time_step in tqdm(reversed(range(0, self.T, skip)), total=self.T//skip):
            self.omega_t = self.omega_scheduler(time_step)
            # print(f"current guidance rate is {self.omega_t}")
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var, x_0_pred = self.p_mean_variance(x_t=x_t, t=t, y=y, method=method, skip=skip)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0

            if method == 'ddim':
                # ODE for DDIM
                x_t = mean
            else:
                # SDE for DDPM
                x_t = mean + torch.exp(0.5 * log_var) * noise
                # # delete this line
                # x_t_Guided=mean_Guided + torch.exp(0.5 * log_var_Guided) * noise
            if return_intermediate:
                xt_list.append(x_t)
                omega_t_list.append(self.omega_t)

            # update guidance in every step
            #x_t = mean + torch.exp(0.5 * log_var) * noise

        x_0 = x_t

        if return_intermediate:
            return torch.clip(x_0, -1, 1), torch.stack(xt_list).permute(1,0,2,3,4), torch.tensor(omega_t_list).to(x_t.device)
        return torch.clip(x_0, -1, 1)


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def uniform_sampling(n, N, k):
    return np.stack([np.random.randint(int(N/n)*i, int(N/n)*(i+1), k) for i in range(n)])


def dist(X, Y):
    sx = torch.sum(X**2, dim=1, keepdim=True)
    sy = torch.sum(Y**2, dim=1, keepdim=True)
    return torch.sqrt(-2 * torch.mm(X, Y.T) + sx + sy.T)


def topk(y, all_y, K):
    dist_y = dist(y, all_y)
    return torch.topk(-dist_y, K, dim=1)[1]


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self,
                 model, beta_1, beta_T, T, dataset,
                 num_class, cfg
                 ):
        super().__init__()

        self.model = model
        self.T = T
        self.dataset = dataset
        self.num_class = num_class
        self.cfg = cfg

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer(
            'sigma_tsq', 1./alphas_bar-1.)
        self.register_buffer('sigma_t',torch.sqrt(self.sigma_tsq))

    def forward(self, x_0, y_0):
        """
        Algorithm 1.
        """
        # original codes
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0) 
        ini_noise = noise

        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        y_l = y_0

        if self.cfg:
            if torch.rand(1)[0] < 1/10:
                y_l = None
            else:
                y_l = y_0

        eps = self.model(x_t, t, y=y_l)
        loss_ddpm = F.mse_loss(eps, noise, reduction='none')
        return loss_ddpm




class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, num_class, img_size=32, var_type='fixedlarge'):
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.num_class = int(num_class)
        self.img_size = img_size
        self.var_type = var_type
        
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer(
            'alphas_bar', alphas_bar)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps): 
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    ### May change it to cg mode.
    

    def p_mean_variance(self, x_t, t, y=None, omega=0.0, method='cfg'):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped}[self.var_type]

        model_log_var = extract(model_log_var, t, x_t.shape)
        unc_eps = None
        # Mean parameterization
        eps = self.model(x_t, t, y=y)
        if omega > 0 and (method == 'cfg' or method == 'ddpm'):
            unc_eps = self.model(x_t, t, y=None)
            guide = eps - unc_eps
            eps = eps + omega * guide

        x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
        model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T, omega=0.0, method='cfg', visualize=False):
        print(f'Gaussian Diffusion Sampler, Omega: {omega}, Method: {method}.')
        """
        Algorithm 2.
        """
        x_t = x_T.clone()
        y = None

        if method == 'uncond':
            y = None
        else:
            y = torch.randint(0, self.num_class, (len(x_t),)).to(x_t.device)

        with torch.no_grad():
            for time_step in tqdm(reversed(range(0, self.T)), total=self.T):
                t = x_T.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, log_var = self.p_mean_variance(x_t=x_t, t=t, y=y,
                                                     omega=omega, method=method)

                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                
                x_t = mean + torch.exp(0.5 * log_var) * noise

        return torch.clip(x_t, -1, 1), y





