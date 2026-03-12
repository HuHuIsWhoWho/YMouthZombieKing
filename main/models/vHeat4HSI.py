import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous()


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1).contiguous()
    
    
def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')
    
      
class HSI_DWS_Stem(nn.Module):
    r""" Stem layer of input HSImage
    Args:
        in_chans (int): number of input channels
        mid_chans (int): number of middle channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self, 
                 in_c=200,
                 mid_c=100,
                 out_c=160, 
                 act_layer='GELU', 
                 norm_layer='BN'):
        super().__init__()
        # 空间维度的深度可分离卷积
        self.dw = nn.Conv2d(in_c, 
                            in_c, 
                            kernel_size=3, 
                            padding=1, 
                            groups=in_c)
        self.norm1 = build_norm_layer(in_c, norm_layer, 
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        # 光谱维度的瓶颈结构特征提取
        self.pw1 = nn.Conv2d(in_c, mid_c, 1)  # 先压缩
        self.norm2 = build_norm_layer(mid_c, norm_layer, 
                                      'channels_first', 'channels_first')
        self.pw2 = nn.Conv2d(mid_c, out_c, 1)  # 再扩展
        # self.norm3 = build_norm_layer(out_c, norm_layer, 
        #                               'channels_first', 'channels_first')

    
    def forward(self, x):
        x = self.dw(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.pw1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.pw2(x)
        # x = self.norm3(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        # Linear = partial(nn.Conv3d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 如果使用 Conv3d（channels_first），则期望输入为 (B, C, H, W)
        # 我们临时在 depth 维度插入长度为1的维度，变为 (B, C, 1, H, W)，保证兼容已有接口
        if isinstance(self.fc1, nn.Conv3d):
            x = x.unsqueeze(2)  # (B, C, 1, H, W)
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            x = x.squeeze(2)  # (B, C', H, W)
            return x
        # 默认行为：使用线性变换（channels_last）
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Heat2D(nn.Module):
    """
    du/dt -k(d2u/dx2 + d2u/dy2) = 0;
    du/dx_{x=0, x=a} = 0
    du/dy_{y=0, y=b} = 0
    =>
    A_{n, m} = C(a, b, n==0, m==0) * sum_{0}^{a}{ sum_{0}^{b}{\phi(x, y)cos(n\pi/ax)cos(m\pi/by)dxdy }}
    core = cos(n\pi/ax)cos(m\pi/by)exp(-[(n\pi/a)^2 + (m\pi/b)^2]kt)
    u_{x, y, t} = sum_{0}^{\infinite}{ sum_{0}^{\infinite}{ core } }
    
    assume a = N, b = M; x in [0, N], y in [0, M]; n in [0, N], m in [0, M]; with some slight change
    => 
    (\phi(x, y) = linear(dwconv(input(x, y))))
    A(n, m) = DCT2D(\phi(x, y))
    u(x, y, t) = IDCT2D(A(n, m) * exp(-[(n\pi/a)^2 + (m\pi/b)^2])**kt)
    """    
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
    
    def infer_init_heat2d(self, freq):
        weight_exp = self.get_decay_map((self.res, self.res), device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, None], self.to_k(freq)), requires_grad=False)
        del self.to_k

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        # cos((x + 0.5) / N * n * \pi) which is also the form of DCT and IDCT
        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * f(x) )
        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * F(n) )
        # returns: (Res_n, Res_x)
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        # exp(-[(n\pi/a)^2 + (m\pi/b)^2])
        # returns: (Res_h, Res_w)
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, H, W = x.shape
        
        x = self.dwconv(x)
        
        x = self.linear(x.permute(0, 2, 3, 1).contiguous()) # B, H, W, 2C
        x, z = x.chunk(chunks=2, dim=-1) # B, H, W, C

        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
            assert weight_cosn is not None
            assert weight_cosm is not None
            assert weight_exp is not None
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]
        
        x = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1))
        x = F.conv1d(x.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)
        
        if self.infer_mode:
            x = torch.einsum("bnmc,nmc->bnmc", x, self.k_exp)
        else:
            weight_exp = torch.pow(weight_exp[:, :, None], self.to_k(freq_embed))
            x = torch.einsum("bnmc,nmc -> bnmc", x, weight_exp) # exp decay
        
        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1))
        x = F.conv1d(x.contiguous().view(-1, M, C), weight_cosm.t().contiguous().view(W, M, 1)).contiguous().view(B, H, W, -1)

        x = self.out_norm(x)
        
        x = x * nn.functional.silu(z)
        x = self.out_linear(x)

        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class Heat3D(nn.Module):
    """
    硬扩展 Heat2D 到三维热传导，包括通道维度（光谱维度）。
    但光谱维度不具备自然图像bias（相邻像素相关性更强），需要调整。
    du/dt - k(d2u/dx2 + d2u/dy2) - l(d2u/dz2) = 0;
    其中 z 是通道维度（光谱维度）。

    光谱维度的扩散：
    波段强度（点） +  可学习波段嵌入→语义距离（边权重） →  基于图拉普拉斯算子的热扩散
    """
    
    def __init__(self, infer_mode=False, res=11, hidden_dim=160, embed_dim=16, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)

        self.hidden_dim = hidden_dim  # 光谱通道数
        self.embed_dim = embed_dim  # 波段嵌入维度
        self.band_embed = nn.Embedding(hidden_dim, embed_dim)  # 可学习的波段嵌入：形状 (C, embed_dim)？？？？人家正常word_embedding是咋学的？至少要输入词汇表吧？？？
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 空间扩散系数
        self.beta  = nn.Parameter(torch.tensor(0.1))  # 光谱扩散系数

        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
    
    def infer_init_heat2d(self, freq):
        weight_exp = self.get_decay_map((self.res, self.res), device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, None], self.to_k(freq)), requires_grad=False)
        del self.to_k
    
    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        # cos((x + 0.5) / N * n * \pi) which is also the form of DCT and IDCT
        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * f(x) )
        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * F(n) )
        # returns: (Res_n, Res_x)
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        # exp(-[(n\pi/a)^2 + (m\pi/b)^2])
        # returns: (Res_h, Res_w)
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight
   
    def forward(self, x: torch.Tensor, freq_embed=None):
        """
        输入 x: (B, C, H, W)
        输出: (B, C, H, W)
        
        步骤：
        1. 光谱维度特征变换
        1. 3D卷积提取局部特征
        2. 线性变换并分拆为主支和门控支
        3. 在三个维度上分别进行DCT变换：H, W, C
        4. 应用三维热扩散衰减
        5. 三个维度上的IDCT变换
        6. 门控机制和输出变换
        """
        B, C, H, W = x.shape
        x = self.dwconv(x)  # (B, hidden_dim, C, H, W)

        # 2. 线性变换并分拆
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())  # (B, H, W, 2C)
        x, z = x.chunk(chunks=2, dim=-1)  # 各为 (B, H, W, C)
        
        # 3. 在三个维度上进行衰减和扩撒
        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
            assert weight_cosn is not None
            assert weight_cosm is not None
            assert weight_exp is not None
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)        
        
        N, M = weight_cosn.shape[0], weight_cosm.shape[0]
        
        x = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1))
        x = F.conv1d(x.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)
        
        # 波段嵌入初始化（仿照 weight_cosn 的缓存方式）
        if not hasattr(self, 'band_embed') or self.band_embed.num_embeddings != C:
            # 创建新的嵌入表，并移动到当前设备
            self.band_embed = nn.Embedding(C, self.embed_dim).to(x.device)
            # 可选：使用与线性层相同的初始化方法
            nn.init.normal_(self.band_embed.weight, std=0.02)
            # 由于 band_embed 是 nn.Module，赋值给 self 即完成注册
        
        # 计算图拉普拉斯特征分解
        E = self.band_embed.weight                     # (C, embed_dim)
        E_norm = F.normalize(E, dim=1)
        sim = E_norm @ E_norm.T
        W_sim = F.softmax(sim / self.temperature, dim=1)
        W_sim = (W_sim + W_sim.T) / 2
        D_mat = torch.diag(W_sim.sum(dim=1))
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_mat.diag() + 1e-8))
        L_sym = torch.eye(C, device=x.device) - D_inv_sqrt @ W_sim @ D_inv_sqrt
        eigvals, eigvecs = torch.linalg.eigh(L_sym)   # (C,), (C, C)
        # 图傅里叶变换：沿光谱维做变换
        x = torch.einsum('bhwc,ck->bhwk', x, eigvecs)  # (B, H, W, C)

        # 5. 计算联合衰减因子
        weight_exp = weight_exp[:, :, None] * torch.exp(self.beta * eigvals)[None, None, :]   # (H, W, C)

        x = torch.einsum("bnml,nml->bnml", x, weight_exp)
        
        # 5. 在三个维度上进行逆变换
        x = torch.einsum('bhwk,kc->bhwc', x, eigvecs.T)  # (B, H, W, C)
        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1))
        x = F.conv1d(x.contiguous().view(-1, M, C), weight_cosm.t().contiguous().view(W, M, 1)).contiguous().view(B, H, W, C)

        x = self.out_norm(x)

        # 6. 门控机制
        x = x * nn.functional.silu(z)
        x = self.out_linear(x)
        
        x = x.permute(0, 3, 1, 2).contiguous()
        
        return x


# class Heat3D(nn.Module):
#     """
#     扩展 Heat2D 到三维热传导，包括通道维度（光谱维度）。
#     但光谱维度不具备自然图像bias（相邻像素相关性更强），需要先调整。
#     du/dt - k(d2u/dx2 + d2u/dy2 + d2u/dz2) = 0;
#     其中 z 是通道维度（光谱维度）。
#     边界条件：Neumann边界条件，即梯度为0。
    
#     实现方式：
#     - 在空间维度 (H, W) 和通道维度 C 上分别进行 DCT/IDCT。
#     - 在光谱维度先进行特征变化以使其适合热扩散处理。
#     - 在频域中应用热扩散衰减，包括通道维度。
#     """
    
#     def __init__(self, infer_mode=False, res=11, hidden_dim=16, **kwargs):
#         super().__init__()
#         self.res = res
#         self.dwconv = nn.Conv3d(1, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=True)
#         self.hidden_dim = hidden_dim
#         self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
#         self.out_norm = nn.LayerNorm(hidden_dim)
#         self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
#         self.infer_mode = infer_mode
#         self.to_k = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim, bias=True),
#             nn.ReLU(),
#         )
    
#     def infer_init_heat3d(self, freq):
#         """
#         推理模式初始化：预计算衰减映射。
#         包括通道维度的衰减。
#         """
#         weight_exp = self.get_decay_map((self.res, self.res, self.hidden_dim), device=freq.device)
#         self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, :], self.to_k(freq)), requires_grad=False)
#         del self.to_k

#     @staticmethod
#     def get_cos_map(N=200, device=torch.device("cpu"), dtype=torch.float):
#         """
#         生成DCT/IDCT所需的余弦权重矩阵。
#         返回形状为 (N, N) 的矩阵，用于一维DCT变换。
#         """
#         weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
#         weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
#         weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
#         weight[0, :] = weight[0, :] / math.sqrt(2)
#         return weight

#     @staticmethod
#     def get_decay_map(resolution=(11, 11, 200), device=torch.device("cpu"), dtype=torch.float):
#         """
#         生成三维衰减映射：exp(-[(nπ/a)^2 + (mπ/b)^2 + (lπ/c)^2])
#         resolution: (H, W, C) 表示三个维度的分辨率
#         返回形状: (H, W, C)
#         """
#         res_h, res_w, res_c = resolution
#         # 生成频率网格
#         weight_n = torch.linspace(0, torch.pi, res_h + 1, device=device, dtype=dtype)[:res_h].view(-1, 1, 1)
#         weight_m = torch.linspace(0, torch.pi, res_w + 1, device=device, dtype=dtype)[:res_w].view(1, -1, 1)
#         weight_l = torch.linspace(0, torch.pi, res_c + 1, device=device, dtype=dtype)[:res_c].view(1, 1, -1)
#         # 计算衰减项
#         weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2) + torch.pow(weight_l, 2)
#         weight = torch.exp(-weight)
#         return weight
   
#     def forward(self, x: torch.Tensor, freq_embed=None):
#         """
#         输入 x: (B, C, H, W)
#         输出: (B, C, H, W)
        
#         步骤：
#         1. 光谱维度特征变换
#         1. 3D卷积提取局部特征
#         2. 线性变换并分拆为主支和门控支
#         3. 在三个维度上分别进行DCT变换：H, W, C
#         4. 应用三维热扩散衰减
#         5. 三个维度上的IDCT变换
#         6. 门控机制和输出变换
#         """
#         B, C, H, W = x.shape
#         # 将 x 从 (B, C, H, W) 重塑为 (B, 1, C, H, W) 以适配 Conv3d
#         x = x.unsqueeze(1)  # (B, 1, C, H, W)
#         x = self.dwconv(x)  # (B, hidden_dim, C, H, W)
#         # 将多余维度聚合（用均值聚合，保持接口不变）
#         x = x.mean(dim=1)  # (B, C, H, W)

#         # 2. 线性变换并分拆
#         x = self.linear(x.permute(0, 2, 3, 1).contiguous())  # (B, H, W, 2C)
#         x, z = x.chunk(chunks=2, dim=-1)  # 各为 (B, H, W, C)
        
#         # 3. 在三个维度上进行DCT变换
#         if ((H, W, C) == getattr(self, "__RES__", (0, 0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
#             weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
#             weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
#             weight_cosl = getattr(self, "__WEIGHT_COSL__", None)
#             weight_exp = getattr(self, "__WEIGHT_EXP__", None)
#             assert weight_cosn is not None
#             assert weight_cosm is not None
#             assert weight_cosl is not None
#             assert weight_exp is not None
#         else:
#             weight_cosn = self.get_cos_map(H, device=x.device).detach_()
#             weight_cosm = self.get_cos_map(W, device=x.device).detach_()
#             weight_cosl = self.get_cos_map(C, device=x.device).detach_()
#             weight_exp = self.get_decay_map((H, W, C), device=x.device).detach_()
#             setattr(self, "__RES__", (H, W, C))
#             setattr(self, "__WEIGHT_COSN__", weight_cosn)
#             setattr(self, "__WEIGHT_COSM__", weight_cosm)
#             setattr(self, "__WEIGHT_COSL__", weight_cosl)
#             setattr(self, "__WEIGHT_EXP__", weight_exp)
        
#         N, M, L = weight_cosn.shape[0], weight_cosm.shape[0], weight_cosl.shape[0]
        
#         x = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1))
#         x = F.conv1d(x.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)
#         x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, N, M)
#         x = F.conv1d(x.contiguous().view(B, C, -1), weight_cosl.contiguous().view(L, C, 1)).contiguous().view(B, L, N, M)
#         x = x.permute(0, 2, 3, 1).contiguous()  # (B, N, M, L)

#         # 4. 应用三维热扩散衰减
#         if self.infer_mode:
#             x = torch.einsum("bnml,nml->bnml", x, self.k_exp)
#         else:
#             weight_exp = torch.pow(weight_exp[:, :, :], self.to_k(freq_embed))
#             x = torch.einsum("bnml,nml->bnml", x, weight_exp)
        
#         # 5. 在三个维度上进行IDCT（逆变换）
#         x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1))
#         x = F.conv1d(x.contiguous().view(-1, M, L), weight_cosm.t().contiguous().view(W, M, 1)).contiguous().view(B, H, W, L)
#         x = x.permute(0, 3, 1, 2).contiguous()  # (B, L, H, W)
#         x = F.conv1d(x.contiguous().view(B, L, -1), weight_cosl.t().contiguous().view(C, L, 1)).contiguous().view(B, C, H, W)
#         x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

#         x = self.out_norm(x)

#         # 6. 门控机制
#         x = x * nn.functional.silu(z)
#         x = self.out_linear(x)
        
#         x = x.permute(0, 3, 1, 2).contiguous()
        
#         return x


class HeatBlock(nn.Module):
    def __init__(
        self,
        res: int = 14,
        infer_mode = False,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint: bool = False,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        mlp_ratio: float = 4.0,
        post_norm = True,
        layer_scale = None,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        # self.op = Heat2D(res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode)
        self.op = Heat3D(hidden_dim=hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=True)
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None
        
        self.infer_mode = infer_mode
        
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)

    def _forward(self, x: torch.Tensor, freq_embed):
        if not self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.op(x, freq_embed)))
                if self.mlp_branch:
                    x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.op(self.norm1(x), freq_embed))
                if self.mlp_branch:
                    x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
            return x
        if self.post_norm:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.norm1(self.op(x, freq_embed)))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.norm2(self.mlp(x))) # FFN
        else:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.op(self.norm1(x), freq_embed))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.mlp(self.norm2(x))) # FFN
        return x
    
    def forward(self, input: torch.Tensor, freq_embed=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, freq_embed)
        else:
            return self._forward(input, freq_embed)


class AdditionalInputSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self[:-1]:
            if isinstance(module, nn.Module):
                x = module(x, *args, **kwargs)
            else:
                x = module(x)
        x = self[-1](x)
        return x


class vHeat4HSI(nn.Module):
    def __init__(self, hsi_patch_size=7, hsi_band_size=200, num_classes=16, depths=[1, 2, 2, 1], 
                 dims=[160, 80, 40, 20], drop_path_rate=0.2, patch_norm=True, post_norm=True,
                 layer_scale=None, use_checkpoint=False, mlp_ratio=4.0,
                 act_layer='GELU', infer_mode=False, **kwargs):
        super().__init__()

        # 高光谱新增参数
        self.hsi_band_size = hsi_band_size
        self.hsi_patch_size = hsi_patch_size

        self.num_classes = num_classes
        self.num_layers = len(depths)

        # 在光谱通道下采样
        # if isinstance(dims, int):
        #     dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        if isinstance(dims, int):
            dims = [int(dims // 2 ** i_layer) for i_layer in range(self.num_layers)]
        if isinstance(dims, float):
            dims = self.hsi_band_size * dims
            dims = [int(dims), int(dims//2), int(dims//4), int(dims//8)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        
        self.depths = depths

        self.patch_embed = HSI_DWS_Stem(in_c=self.hsi_band_size, mid_c=self.hsi_band_size // 2, out_c=self.embed_dim)


        # 高光谱时计算各层分辨率
        # res0 = img_size/patch_size
        # self.res = [int(res0), int(res0//2), int(res0//4), int(res0//8)]
        res0 = hsi_patch_size
        # 不在空间通道下采样
        self.res = [int(res0), int(res0), int(res0), int(res0)]
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        self.infer_mode = infer_mode
        
        self.freq_embed = nn.ParameterList()
        for i in range(self.num_layers):
            self.freq_embed.append(nn.Parameter(torch.zeros(self.res[i], self.res[i], self.dims[i]), requires_grad=True))
            trunc_normal_(self.freq_embed[i], std=.02)
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(self.make_layer(
                res = self.res[i_layer],
                dim = self.dims[i_layer],
                depth = depths[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=LayerNorm2d,
                post_norm=post_norm,
                layer_scale=layer_scale,
                downsample=self.make_hsi_downsample(  # <1>：原make_downsample改为make_hsi_downsample，只在光谱维度下采样
                    self.dims[i_layer], 
                    self.dims[i_layer + 1], 
                    norm_layer=LayerNorm2d,
                ) if (i_layer < self.num_layers - 1) else nn.Identity(),
                mlp_ratio=mlp_ratio,
                infer_mode=infer_mode,
            ))
            
        self.classifier = nn.Sequential(
            LayerNorm2d(self.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.num_features, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def make_downsample(dim=160, out_dim=80, norm_layer=LayerNorm2d):
        return nn.Sequential(
            #norm_layer(dim),
            #nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_dim)
        )
    
    @staticmethod
    def make_hsi_downsample(dim=96, out_dim=192, norm_layer=LayerNorm2d):
        """直接替换vHeat中的make_downsample"""
        return nn.Sequential(
            # 光谱下采样（不是空间下采样！）
            nn.Conv2d(dim, out_dim, kernel_size=1, stride=1),  # stride=1保持空间尺寸
            norm_layer(out_dim),
        )

    @staticmethod
    def make_layer(
        res=11,
        dim=160, 
        depth=2,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=LayerNorm2d,
        post_norm=True,
        layer_scale=None,
        downsample=nn.Identity(), 
        mlp_ratio=4.0,
        infer_mode=False,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(HeatBlock(
                res=res,
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                post_norm=post_norm,
                layer_scale=layer_scale,
                infer_mode=infer_mode,
            ))
        
        return AdditionalInputSequential(
            *blocks, 
            downsample,
        )
 
    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def infer_init(self):
        for i, layer in enumerate(self.layers):
            for block in layer[:-1]:
                block.op.infer_init_heat2d(self.freq_embed[i])
        del self.freq_embed
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.infer_mode:
            for layer in self.layers:
                x = layer(x)
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x, self.freq_embed[i]) # (B, C, H, W)
        return x

    def forward(self, x):
        # x = self.hsi_mapper(x)
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    from fvcore.nn import flop_count_table, flop_count_str, FlopCountAnalysis
    model = vHeat4HSI().cuda()
    input = torch.randn((1, 200, 7, 7), device=torch.device('cuda'))
    analyze = FlopCountAnalysis(model, (input,))
    print(flop_count_str(analyze))
