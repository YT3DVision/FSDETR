import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, RepC3
__all__ = ['CFSB', 'SNI', 'SHAB', 'SPDConv']

######################################## SPD-Conv start ########################################

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x

######################################## SPD-Conv end ########################################


######################################## FreqSpatial start ########################################

class ScharrConv(nn.Module):
	def __init__(self, channel):
		super(ScharrConv, self).__init__()

		# 定义Scharr算子的水平和垂直卷积核
		scharr_kernel_x = np.array([[3, 0, -3],
									[10, 0, -10],
									[3, 0, -3]], dtype=np.float32)

		scharr_kernel_y = np.array([[3, 10, 3],
									[0, 0, 0],
									[-3, -10, -3]], dtype=np.float32)

		# 将Scharr核转换为PyTorch张量并扩展为通道数
		scharr_kernel_x = torch.tensor(scharr_kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
		scharr_kernel_y = torch.tensor(scharr_kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

		# 扩展为多通道
		self.scharr_kernel_x = scharr_kernel_x.expand(channel, 1, 3, 3)  # (channel, 1, 3, 3)
		self.scharr_kernel_y = scharr_kernel_y.expand(channel, 1, 3, 3)  # (channel, 1, 3, 3)

		# 定义卷积层，但不学习卷积核，直接使用Scharr核
		self.scharr_kernel_x_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
		self.scharr_kernel_y_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

		# 将卷积核的权重设置为Scharr算子的核
		self.scharr_kernel_x_conv.weight.data = self.scharr_kernel_x.clone()
		self.scharr_kernel_y_conv.weight.data = self.scharr_kernel_y.clone()

		# 禁用梯度更新
		self.scharr_kernel_x_conv.requires_grad = False
		self.scharr_kernel_y_conv.requires_grad = False

	def forward(self, x):
		# 对输入的特征图进行Scharr卷积（水平和垂直方向）
		grad_x = self.scharr_kernel_x_conv(x)
		grad_y = self.scharr_kernel_y_conv(x)

		# 计算梯度幅值
		edge_magnitude = grad_x * 0.5 + grad_y * 0.5

		return edge_magnitude


class FreqSpatial(nn.Module):
	def __init__(self, in_channels):
		super(FreqSpatial, self).__init__()

		self.sed = ScharrConv(in_channels)

		# 时域卷积部分
		self.spatial_conv1 = Conv(in_channels, in_channels)
		self.spatial_conv2 = Conv(in_channels, in_channels)

		# 频域卷积部分
		self.fft_conv = Conv(in_channels * 2, in_channels * 2, 3)
		self.fft_conv2 = Conv(in_channels, in_channels, 3)

		self.final_conv = Conv(in_channels, in_channels, 1)

	def forward(self, x):
		batch, c, h, w = x.size()
		# 时域提取
		spatial_feat = self.sed(x)
		spatial_feat = self.spatial_conv1(spatial_feat)
		spatial_feat = self.spatial_conv2(spatial_feat + x)

		# 频域卷积
		# 1. 先转换到频域
		fft_feat = torch.fft.rfft2(x, norm='ortho')
		x_fft_real = torch.unsqueeze(torch.real(fft_feat), dim=-1)
		x_fft_imag = torch.unsqueeze(torch.imag(fft_feat), dim=-1)
		fft_feat = torch.cat((x_fft_real, x_fft_imag), dim=-1)
		fft_feat = rearrange(fft_feat, 'b c h w d -> b (c d) h w').contiguous()

		# 2. 频域卷积处理
		fft_feat = self.fft_conv(fft_feat)

		# 3. 还原回时域
		fft_feat = rearrange(fft_feat, 'b (c d) h w -> b c h w d', d=2).contiguous()
		fft_feat = torch.view_as_complex(fft_feat)
		fft_feat = torch.fft.irfft2(fft_feat, s=(h, w), norm='ortho')

		fft_feat = self.fft_conv2(fft_feat)

		# 合并时域和频域特征
		out = spatial_feat + fft_feat
		return self.final_conv(out)


class CFSB(C2f):
	def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
		super().__init__(c1, c2, n, shortcut, g, e)
		self.m = nn.ModuleList(FreqSpatial(self.c) for _ in range(n))


######################################## FreqSpatial end ########################################


class SNI(nn.Module):
    '''
    https://github.com/AlanLi1997/rethinking-fpn
    soft nearest neighbor interpolation for up-sampling
    secondary features aligned
    '''
    def __init__(self, up_f=2):
        super(SNI, self).__init__()
        self.us = nn.Upsample(None, up_f, 'nearest')
        self.alpha = 1/(up_f**2)

    def forward(self, x):
        return self.alpha*self.us(x)


######################################## SHViT CVPR2024 start ########################################

class Conv2d_BN(torch.nn.Sequential):
	def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
				 groups=1, bn_weight_init=1, resolution=-10000):
		super().__init__()
		self.add_module('c', torch.nn.Conv2d(
			a, b, ks, stride, pad, dilation, groups, bias=False))
		self.add_module('bn', torch.nn.BatchNorm2d(b))
		torch.nn.init.constant_(self.bn.weight, bn_weight_init)
		torch.nn.init.constant_(self.bn.bias, 0)

	@torch.no_grad()
	def fuse_self(self):
		c, bn = self._modules.values()
		w = bn.weight / (bn.running_var + bn.eps) ** 0.5
		w = c.weight * w[:, None, None, None]
		b = bn.bias - bn.running_mean * bn.weight / \
			(bn.running_var + bn.eps) ** 0.5
		m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
			0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
							groups=self.c.groups,
							device=c.weight.device)
		m.weight.data.copy_(w)
		m.bias.data.copy_(b)
		return m


class Residual(nn.Module):
	def __init__(self, fn):
		super(Residual, self).__init__()
		self.fn = fn

	def forward(self, x):
		return self.fn(x) + x


class SHSA_GroupNorm(torch.nn.GroupNorm):
	"""
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

	def __init__(self, num_channels, **kwargs):
		super().__init__(1, num_channels, **kwargs)


class SHSABlock_FFN(torch.nn.Module):
	def __init__(self, ed, h):
		super().__init__()
		self.pw1 = Conv2d_BN(ed, h)
		self.act = torch.nn.SiLU()
		self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

	def forward(self, x):
		x = self.pw2(self.act(self.pw1(x)))
		return x


class SHSA(torch.nn.Module):
	"""Single-Head Self-Attention"""

	def __init__(self, dim, qk_dim, pdim):
		super().__init__()
		self.scale = qk_dim ** -0.5
		self.qk_dim = qk_dim
		self.dim = dim
		self.pdim = pdim

		self.pre_norm = SHSA_GroupNorm(pdim)

		self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
		self.proj = torch.nn.Sequential(torch.nn.SiLU(), Conv2d_BN(
			dim, dim, bn_weight_init=0))

	def forward(self, x):
		B, C, H, W = x.shape
		x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
		x1 = self.pre_norm(x1)
		qkv = self.qkv(x1)
		q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
		q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

		attn = (q.transpose(-2, -1) @ k) * self.scale
		attn = attn.softmax(dim=-1)
		x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
		x = self.proj(torch.cat([x1, x2], dim=1))

		return x


class SHSABlock(torch.nn.Module):
	def __init__(self, dim, qk_dim=16, pdim=64):
		super().__init__()
		self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
		self.mixer = Residual(SHSA(dim, qk_dim, pdim))
		self.ffn = Residual(SHSABlock_FFN(dim, int(dim * 2)))

	def forward(self, x):
		return self.ffn(self.mixer(self.conv(x)))


class SHAB(C2f):
	def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
		super().__init__(c1, c2, n, shortcut, g, e)
		self.m = nn.ModuleList(SHSABlock(self.c) for _ in range(n))

######################################## SHViT CVPR2024 end ########################################