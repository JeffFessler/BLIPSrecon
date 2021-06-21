import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
from util.simulation import simu_Affine2D, simu_noise2D
from util.metrics import roll_2
from util.util import ifft2, fft2, fftshift, ifftshift
import time
from util.util import complex_matmul, complex_conj, bspline2_1ndsynth, bilinear_interpolate_torch_gridsample, \
    center_crop
from util.rbf import rbf, rbf_keops
from torchkbnufft import AdjMriSenseNufft, MriSenseNufft, KbNufft, AdjKbNufft, ToepSenseNufft
from .didn import DIDN
import math
import scipy
import scipy.linalg


# from util.cg_block import cg_block
###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(opt, input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'resiso_6blocks':
        netG = ResnetGeneratorISO(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_64':
        netG = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'DIDN':
        netG = DIDN(input_nc, output_nc, global_residual=not opt.no_global_residual)
    elif which_model_netG == 'Mixer':
        netG = Mixer(ngf)
    elif which_model_netG == 'Mixer1':
        netG = Mixer1(ngf)
    elif which_model_netG == 'density':
        netG = DensityLayer(opt.nx, opt.ny)
    elif which_model_netG == 'densitybeta':
        netG = DensityLayer_beta(opt.nx, opt.ny, opt.beta)
    elif which_model_netG == 'sampling':
        netG = SamplingLayer(opt.mask_path, opt.num_shots, opt.stocha_size, opt.tanh_alpha, opt.Trans, opt.noise_level,
                             opt.angle)
    elif which_model_netG == 'sampling3D':
        netG = SamplingLayer3D(opt.mask_path, opt.num_shots, opt.stocha_size, opt.tanh_alpha, opt.Trans,
                               opt.noise_level,
                               opt.angle)

    elif which_model_netG == "samplingBspline":
        netG = SamplingLayerBspline2D(opt.num_shots, opt.nfe, decim=opt.decim_rate, dt=opt.dt,
                                      res=opt.res, init_traj=opt.mask_path,
                                      gpu_ids=gpu_ids, ext=opt.padding)
    elif which_model_netG == "Multinomial":
        netG = Multinomial(opt.nx, opt.ny, init_traj=opt.mask_path, load_traj=opt.load_traj)
    elif which_model_netG == 'VN':
        netG = VN_momentum(opt.num_blocks, opt.num_FOE, opt.num_rbf)

    elif which_model_netG == 'MODL':
        netG = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'resunet_64':
        netG = ResUnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                global_residual=not opt.no_global_residual)
    elif which_model_netG == 'LR':
        netG = para_lambda(opt.num_blocks)

    # elif which_model_netG == 'MODL':
    #     netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'Multi':
        netD = MultiviewDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, use_l1=False, use_wgan=False, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        elif use_l1:
            self.loss = nn.L1Loss()  # With modification
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetGeneratorISO(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGeneratorISO, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias)]
        for i in range(n_blocks):
            model += [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        self.num_downs = num_downs
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def calculate_downsampling_padding2d(self, tensor):
        # calculate pad size
        factor = 2 ** self.num_downs
        imshape = np.array(tensor.shape[-2:])
        paddings = np.ceil(imshape / factor) * factor - imshape
        paddings = paddings.astype(np.int) // 2
        p2d = (paddings[1], paddings[1], paddings[0], paddings[0])
        return p2d

    def pad2d(self, tensor, p2d):
        if np.any(p2d):
            # order of padding is reversed. that's messed up.
            tensor = F.pad(tensor, p2d)
        return tensor

    def unpad2d(self, tensor, shape):
        if tensor.shape == shape:
            return tensor
        else:
            return center_crop(tensor, shape)

    def forward(self, input):
        orig_shape2d = input.shape[-2:]
        p2d = self.calculate_downsampling_padding2d(input)
        input = self.pad2d(input, p2d)
        return self.unpad2d(self.model(input), orig_shape2d)


class ResUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, global_residual=True):
        super(ResUnetGenerator, self).__init__()

        self.residual_global = global_residual
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.residual_global:
            return self.model(input) + input
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class Mixer(nn.Module):
    def __init__(self, ndf=32):
        super(Mixer, self).__init__()
        self.net = [
            nn.Conv2d(2, ndf, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(ndf, 2, kernel_size=1, stride=1, padding=0)]
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class Mixer1(nn.Module):
    def __init__(self, ndf=32):
        super(Mixer1, self).__init__()
        self.net = [
            nn.Conv2d(2, ndf, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(ndf, 2, kernel_size=1, stride=1, padding=0)]
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


# class DensityLayer(nn.Module):
#     def __init__(self, nx, ny):
#         super(DensityLayer, self).__init__()
#         self.net = [nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),
#                     nn.Conv2d(6, 1, kernel_size=3, stride=1, padding=1),
#                     nn.Sigmoid()
#                     ]
#         self.nx = nx
#         self.ny = ny
#         self.net = nn.Sequential(*self.net)
#
#     def forward(self, inputdensity):
#         # intensity: [0,1] size: [1,1,nx,ny]
#         return self.net(inputdensity)


class DensityLayer(nn.Module):
    def __init__(self, nx, ny):
        super(DensityLayer, self).__init__()
        self.weights = nn.Parameter(torch.ones(1, 1, nx, ny))
        # print('self.weights',self.weights.size())
        # print('max',torch.max(self.weights))
        # print('min',torch.min(self.weights))
        self.activation = nn.Sigmoid()
        self.net = [nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(6, 1, kernel_size=3, stride=1, padding=1),
                    nn.Sigmoid()
                    ]
        self.net = nn.Sequential(*self.net)
        # self.weights.register_hook(print)

    def forward(self, x):
        # print('max',torch.max(self.weights))
        # print('min',torch.min(self.weights))  #
        return self.activation(x * self.weights)  # element-wise multiplication


class Multinomial(nn.Module):
    def __init__(self, nx, ny, load_traj=True, init_traj=None):
        super(Multinomial, self).__init__()
        self.weights = torch.ones(nx, ny)
        if load_traj:
            self.weights = torch.tensor(np.load(init_traj))
        self.weights = torch.nn.Parameter(self.weights)
        print(self.weights)
        self.sf = nn.Softmax()

    def forward(self, x):
        return self.sf(torch.flatten(self.weights))


class MultiviewDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        n_layers = 5
        super(MultiviewDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 2):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        sequence_1 = sequence + [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence_1 += [nn.Sigmoid()]

        for n in range(2, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        sequence_2 = sequence + [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence_2 += [nn.Sigmoid()]

        for n in range(3, 4):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence_2 += [nn.Sigmoid()]
        self.model_1 = nn.Sequential(*sequence_1)
        self.model_2 = nn.Sequential(*sequence_2)
        self.model_3 = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model_1(input), self.model_2(input), self.model_3(input)


class VN_momentum(nn.Module):
    def __init__(self, num_blocks, num_kernel, num_rbf):
        super(VN_momentum, self).__init__()
        self.in_channels = 2
        self.num_blocks = num_blocks
        self.num_kernel = num_kernel
        self.num_rbf = num_rbf
        self.lambda_1st = torch.nn.Parameter(0.00001 * torch.ones(num_blocks))  # Weight of gradient
        self.lambda_mo = torch.nn.Parameter(0.0000001 * torch.ones(num_blocks))
        self.lambda_mu = torch.nn.Parameter(0.0000001 * torch.ones(num_blocks))  # Weight of first-order momentum
        self.lambda_R = torch.nn.Parameter(0.000001 * torch.ones(num_blocks))
        self.kernel = torch.nn.Parameter(torch.ones(self.num_blocks, self.num_kernel, self.in_channels, 11, 11))
        self.rbf_weights = torch.nn.Parameter(0.00000001 * torch.ones(self.num_blocks, self.num_kernel, self.num_rbf))

    def forward(self, k, Smap, mask, Ireal):
        # K: undersampled k-space data [Batch, coil, 2, M, N]
        # Smap:
        # Mask:
        # Due to the interpolation effect of RBF, we should make sure that our input image is between [-1,1]
        BchSize, num_coil, _, M, N = Smap.size()
        A = OPA(Smap)
        AT = OPAT(Smap)
        im0 = AT(k, mask)
        im = im0
        im_last = im0
        k_under = k * (mask.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4))
        for ii in range(1, self.num_blocks):
            # print("stage",ii)
            # print('knl',torch.max(self.kernel))
            # print('maxim0', torch.max(im0))
            im_K = F.conv2d(im, self.kernel[ii, :, :, :, :], padding=5)
            # print('maximk', torch.max(im_K))
            # print('stdimk', torch.std(im_K))
            im_r = rbf_keops(self.rbf_weights[ii, :, :], im_K, 100, 150)
            # print('maximr', torch.max(im_r))
            im_R = F.conv_transpose2d(im_r, self.kernel[ii, :, :, :, :], padding=5)
            # print('maximR', torch.max(im_R))
            # print('consisitency', AT(A(im,mask)-k_under,mask).size())
            im_dc = AT(A(im - self.lambda_mu[ii] * im_last, mask) - k_under, mask)
            im_temp = im
            # im = im - self.lambda_R[ii] * im_R - self.lambda_1st[ii]*im_dc - self.lambda_mo[ii]*im_last
            im = im - self.lambda_R[ii] * im_R - self.lambda_1st[ii] * im_dc - self.lambda_mo[ii] * (im - im_last)
            im_last = im_temp
            # print('lambdar', self.lambda_R[ii])
            # print('lambda1', self.lambda_1st[ii])
            # print('lambdamo', self.lambda_mo[ii])
            if ii == 3:
                im_3 = im
            if ii == 6:
                im_6 = im
        return im, im0, im_R, im_3, im_6, im_dc


class OPAT(nn.Module):
    # Initialize: Sensitivity maps: [Batch, Coils, 2, M, N]
    # Input: K: [Batch, coils, 2, M, N]
    #        Mask: [Batch, 2, M, N]
    # Output: Image: [Batch, 2, M, N]
    def __init__(self, Smap):
        super(OPAT, self).__init__()
        self.Smap = Smap

    def forward(self, k, mask):
        BchSize, num_coil, _, M, N = self.Smap.size()
        mask = mask.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        k_under = k * mask
        im_u = ifft2((k_under).permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        im = complex_matmul(im_u, complex_conj(self.Smap)).sum(1)
        return im


class OPA(nn.Module):
    # Initialize: Sensitivity maps: [Batch, Coils, 2, M, N]
    # Input: Image: [Batch, 2, M, N]
    #        Mask: [Batch, 2, M, N]
    # Return: K: [Batch, Coils, 2, M, N]
    def __init__(self, Smap):
        super(OPA, self).__init__()
        self.Smap = Smap

    def forward(self, im, mask):
        BchSize, num_coil, _, M, N = self.Smap.size()
        im = im.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        Image_s = complex_matmul(im, self.Smap)
        k_full = fft2(Image_s.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        mask = mask.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)

        k_under = k_full * mask
        return k_under


class OPATA(nn.Module):
    # Initialize: Sensitivity maps: [Batch, Coils, 2, M, N]
    # Input: Image: [Batch, 2, M, N]
    #        Mask: [Batch, 2, M, N]
    # Return: Image: [Batch, 2, M, N]
    def __init__(self, Smap, lambda1):
        super(OPATA, self).__init__()
        self.Smap = Smap
        self.lambda1 = lambda1

    def forward(self, im, mask):
        BchSize, num_coil, _, M, N = self.Smap.size()
        im_coil = im.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        Image_s = complex_matmul(im_coil, self.Smap)
        k_full = fft2(Image_s.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        mask = mask.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        k_under = k_full * mask
        Im_U = ifft2((k_under).permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        Im_Us = complex_matmul(Im_U, complex_conj(self.Smap)).sum(1)
        return Im_Us + self.lambda1 * im


class CG_A(nn.Module):
    def __init__(self, tol, lambda1):
        super(CG_A, self).__init__()
        self.lambda1 = lambda1  # Could also be learnable
        self.tol = tol

    def forward(self, smap, mask, dn, z_pad, Ireal):
        # Sensitivity map: [Batch, Coils, 2, M, N]
        # Dn: denoised image from CNN, [Batch, 2, M, N]
        # Z_pad: Ifake = alised image, [Batch, 2, M, N]
        ATA = OPATA(smap, self.lambda1)
        b0 = dn * self.lambda1 + z_pad
        x0 = b0
        num_loop = 0
        r0 = b0 - ATA(x0, mask)
        p0 = r0
        rk = r0
        pk = p0
        xk = x0
        while torch.norm(rk).data.cpu().numpy().tolist() > self.tol:
            # for ii in range(4):
            # print('stage:',ii)
            # print('norm of r',torch.norm(rk))
            # print('norm of p',torch.norm(pk))
            # print('pap',torch.pow(torch.norm(pk*ATA(pk,mask)),2))
            # print('tol',self.tol)
            rktrk = torch.pow(torch.norm(rk), 2)
            pktapk = torch.sum(complex_matmul(complex_conj(pk), ATA(pk, mask)))
            alpha = rktrk / pktapk  # Check if it is real!
            # print('alpha',alpha)
            xk1 = xk + alpha * pk
            rk1 = rk - alpha * ATA(pk, mask)
            rk1trk1 = torch.pow(torch.norm(rk1), 2)
            # print('norm of r1',torch.norm(rk1))
            beta = rk1trk1 / rktrk
            pk1 = rk1 + beta * pk
            # print('beta',beta)
            xk = xk1
            rk = rk1
            pk = pk1
            num_loop = num_loop + 1
            print(num_loop, ',error:', torch.norm(ATA(xk, mask) - b0))
            # print('norm of b0', torch.norm(b0))
            # if ii==0:
            #     rk_stage1 = rk
            # if ii==1:
            #     rk_stage2 = rk
            # if ii==2:
            #     rk_stage3 = rk
            # if ii==3:
            #     rk_stage4 = rk
            # if ii==0:
            #     rk_stage1 = xk1
            # if ii==1:
            #     rk_stage2 = xk1
            # if ii==2:
            #     rk_stage3 = xk1
            # if ii==3:
            #     rk_stage4 = xk1
            # print('crit',torch.norm(rk).data.cpu().numpy().tolist()<self.tol)
        # print("total iteration number:", num_loop)
        # return rk_stage1, rk_stage2, rk_stage3, rk_stage4
        return xk


class CG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dn, tol, lambda1, smap, mask, z_pad):
        tol = torch.tensor(tol).to(device=dn.device, dtype=dn.dtype)
        lambda1 = torch.tensor(lambda1).to(device=dn.device, dtype=dn.dtype)
        ctx.save_for_backward(tol, lambda1, smap, mask, z_pad)
        return cg_block(smap, mask, dn * lambda1 + z_pad, z_pad, lambda1, tol, dn=dn)

    @staticmethod
    def backward(ctx, dx):
        tol, lambda1, smap, mask, z_pad = ctx.saved_tensors
        return lambda1 * cg_block(smap, mask, dx, z_pad, lambda1, tol), None, None, None, None, None


def cg_block(smap, mask, b0, z_pad, lambda1, tol, M=None, dn=None):
    # A specified conjugated gradietn block for MR system matrix A
    # Sensitivity map: [Batch, Coils, 2, M, N]
    # Dn: denoised image from CNN, [Batch, 2, M, N]
    # Z_pad: Ifake = alised image, [Batch, 2, M, N]
    ATA = OPATA(smap, lambda1)
    x0 = torch.zeros_like(b0)
    if dn is not None:
        x0 = dn
    num_loop = 0
    r0 = b0 - ATA(x0, mask)
    p0 = r0
    rk = r0
    pk = p0
    xk = x0
    while torch.norm(rk).data.cpu().numpy().tolist() > tol:
        rktrk = torch.pow(torch.norm(rk), 2)
        pktapk = torch.sum(complex_matmul(complex_conj(pk), ATA(pk, mask)))
        # print('real')
        # print(torch.norm(complex_matmul(complex_conj(pk),ATA(pk,mask))[0,0,:,:]))
        # print('imag')
        # print(torch.norm(complex_matmul(complex_conj(pk),ATA(pk,mask))[0,1,:,:]))
        alpha = rktrk / pktapk
        xk1 = xk + alpha * pk
        rk1 = rk - alpha * ATA(pk, mask)
        rk1trk1 = torch.pow(torch.norm(rk1), 2)
        beta = rk1trk1 / rktrk
        pk1 = rk1 + beta * pk
        xk = xk1
        rk = rk1
        pk = pk1
        num_loop = num_loop + 1
        # print(torch.norm(rk).data.cpu().numpy().tolist())
    # print(num_loop)
    return xk



