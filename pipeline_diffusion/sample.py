import copy
import json
import os
import warnings
from absl import app, flags

import numpy as np

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from torchvision.utils import make_grid, save_image
from tqdm import trange

from diffusion import GaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score


FLAGS = flags.FLAGS
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
flags.DEFINE_bool('sample', False, help='generate series of images from the same x_t')
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# DATASET
flags.DEFINE_string('dataset', 'cifar10', help='dataset')
# DISTILLATION
flags.DEFINE_string('teacher_pretrain', './ckpt/cifar10/diffuser-unet-1222.pt', help='pretrained UNet teacher')
flags.DEFINE_string('student_pretrain', './ckpt/cifar10/diffuser-unet-2221.pt', help='pretrained UNet student')
flags.DEFINE_string('red_pretrain', './ckpt/cifar10/red-unet-2221.pt', help='pretrained UNet RED student')
flags.DEFINE_string('dist_config', './config/reedx2_dist_config_alpha0.1.yaml', help='red distillation config')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_SAMPLE', help='log directory')
flags.DEFINE_integer('img_size', 32, "image size of sampling")
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_evals', 50000, help='the number of generated images for evaluation')
flags.DEFINE_integer('num_samples', 256, help='the number of generated images')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/fid_stats_celeba64_train_50000_ddim.npz', help='FID cache')

device = torch.device('cuda:0')


class SamplerList(nn.Module):
    def __init__(self, models, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        super().__init__()
        self.T = T
        samplers = []
        for model in models:
            samplers.append(GaussianDiffusionSampler(model, beta_1, beta_T, T, img_size, mean_type, var_type))
        self.samplers = nn.ModuleList(samplers)

    def forward(self, x_T):
        x_t = [x_T] * len(self.samplers)
        for time_step in reversed(range(self.T)):
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t[0])
            else:
                noise = 0
            for i in range(len(self.samplers)):
                sampler_i, xt_i = self.samplers[i], x_t[i]
                t_i = xt_i.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean_i, log_var_i = sampler_i.p_mean_variance(x_t=xt_i, t=t_i)
                x_t[i] = mean_i + torch.exp(0.5 * log_var_i) * noise
        x_0 = x_t
        return [torch.clip(x, -1, 1) for x in x_0]


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_evals, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_evals - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_evals,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images

def eval_model(model, model_type):
    # model setup
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print(f"{model_type} Model: IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, f'{model_type}_evals.png'),
        nrow=16)

def eval():
    teacher_model, student_model, red_model = create_model()
    eval_model(teacher_model, "teacher")
    eval_model(student_model, "student")
    eval_model(red_model, "red")


def generate_images(sampler, x_T):
    with torch.no_grad():
        # print("inference")
        # print(sampler)
        images_list = sampler(x_T.to(device))
        # print("done")
    return [(image.cpu() + 1) / 2 for image in images_list]

def sample():
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    teacher_model, student_model, red_model = create_model()
    sampler = SamplerList(
        [teacher_model.eval(), student_model.eval(), red_model.eval()], FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    teacher_samples, student_samples, red_samples = [], [], []
    desc = "generating images"
    for i in trange(0, FLAGS.num_samples, FLAGS.batch_size, desc=desc):
        batch_size = min(FLAGS.batch_size, FLAGS.num_samples - i)
        x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
        teacher_images, student_images, red_images = generate_images(sampler, x_T)
        teacher_samples.append(teacher_images)
        student_samples.append(student_images)
        red_samples.append(red_images)

    teacher_samples = torch.cat(teacher_samples, dim=0)
    student_samples = torch.cat(student_samples, dim=0)
    red_samples = torch.cat(red_samples, dim=0)

    mini_samples = np.zeros((batch_size, 3, FLAGS.img_size, FLAGS.img_size*3+2))
    for i in range(batch_size):
        mini_samples[i, :, :, :FLAGS.img_size] = teacher_samples[i]
        mini_samples[i, :, :, FLAGS.img_size + 1:FLAGS.img_size * 2 + 1] = student_samples[i]
        mini_samples[i, :, :, FLAGS.img_size*2 + 2:] = red_samples[i]

    save_image(
        torch.tensor(mini_samples),
        os.path.join(FLAGS.logdir, 'mini_samples.png'),
        nrow=1)

    all_samples = torch.cat([teacher_samples, student_samples, red_samples], dim=0).numpy()
    save_image(
        torch.tensor(all_samples),
        os.path.join(FLAGS.logdir, 'all_samples.png'),
        nrow=FLAGS.num_samples)

    save_image(
        torch.tensor(teacher_samples.numpy())[:min(256, FLAGS.num_samples)],
        os.path.join(FLAGS.logdir, 'teacher_samples.png'),
        nrow=16)
    save_image(
        torch.tensor(student_samples.numpy())[:min(256, FLAGS.num_samples)],
        os.path.join(FLAGS.logdir, 'student_samples.png'),
        nrow=16)
    save_image(
        torch.tensor(red_samples.numpy())[:min(256, FLAGS.num_samples)],
        os.path.join(FLAGS.logdir, 'red_samples.png'),
        nrow=16)


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    if FLAGS.sample:
        sample()
    if FLAGS.eval:
        eval()




def create_model():
    # model setup
    teacher_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, pool_strides=[1,2,2,2])
    teacher_ckpt = torch.load(FLAGS.teacher_pretrain)
    teacher_model.load_state_dict(teacher_ckpt['net_model'])
    # teacher_model.load_state_dict(teacher_ckpt['ema_model'])

    student_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, pool_strides=[2,2,2,1])
    student_ckpt = torch.load(FLAGS.student_pretrain)
    student_model.load_state_dict(student_ckpt['net_model'])

    # print(teacher_model, student_model)

    import sys
    sys.path.append('/home/fangchen/WorkDir/Residual-Encoded-Distillation/')
    from reed import ReED_Diffuser
    red_model = ReED_Diffuser(copy.deepcopy(student_model), [copy.deepcopy(teacher_model)], FLAGS.dist_config,
                              dummy_input={'x': torch.randn(1, 3, FLAGS.img_size, FLAGS.img_size),
                                           't': torch.randint(FLAGS.T, size=(1, ))})

    red_ckpt = torch.load(FLAGS.red_pretrain)
    red_model.load_state_dict(red_ckpt['net_model'])

    red_model.graduate()
    return teacher_model, student_model, red_model


if __name__ == '__main__':
    app.run(main)
