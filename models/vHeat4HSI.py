import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.layers import DropPath, to_2tuple, trunc_normal_

# -----------------------------------------------------------------------------
# 基础组件
# -----------------------------------------------------------------------------

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

def build_norm_layer(dim, norm_layer, in_format='channels_last', out_format='channels_last', eps=1e-6):
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
        raise NotImplementedError(f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)

def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()
    raise NotImplementedError(f'build_act_layer does not support {act_layer}')

# -----------------------------------------------------------------------------
# 高光谱适配组件
# -----------------------------------------------------------------------------

class HSI_DWS_Stem(nn.Module):
    """
    针对高光谱图像设计的 Stem 层。
    适配输入通道数，并结合空间深度卷积与光谱点卷积。
    """
    def __init__(self, in_c=200, mid_c=100, out_c=160, act_layer='GELU', norm_layer='BN'):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c)
        self.norm1 = build_norm_layer(in_c, norm_layer, 'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.pw1 = nn.Conv2d(in_c, mid_c, 1)
        self.norm2 = build_norm_layer(mid_c, norm_layer, 'channels_first', 'channels_first')
        self.pw2 = nn.Conv2d(mid_c, out_c, 1)

    def forward(self, x):
        x = self.dw(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.pw1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.pw2(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# -----------------------------------------------------------------------------
# 核心扩散算子
# -----------------------------------------------------------------------------

class SpectralGraphDiffusion(nn.Module):
    """
    基于图拉普拉斯算子的光谱维度扩散。
    利用波段嵌入学习光谱维度的非局部相关性。
    """
    def __init__(self, channels, embed_dim=16, temperature=0.1):
        super().__init__()
        self.channels = channels
        self.temperature = temperature
        self.band_embed = nn.Parameter(torch.randn(channels, embed_dim))
        self.beta = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.band_embed, std=0.02)

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        device = x.device
        
        # 1. 计算相似度矩阵 W
        E_norm = F.normalize(self.band_embed, dim=1)
        sim = torch.matmul(E_norm, E_norm.t())
        W = F.softmax(sim / self.temperature, dim=1)
        W = (W + W.t()) / 2
        
        # 2. 构建对称归一化图拉普拉斯算子 L_sym
        d = W.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(d + 1e-8))
        L_sym = torch.eye(C, device=device) - torch.matmul(torch.matmul(D_inv_sqrt, W), D_inv_sqrt)
        
        # 3. 特征分解
        eigvals, eigvecs = torch.linalg.eigh(L_sym)
        
        # 4. 图傅里叶变换 (GFT)
        x_freq = torch.matmul(x, eigvecs)
        
        # 5. 频域扩散衰减
        decay = torch.exp(-torch.abs(self.beta) * eigvals)
        x_freq_decayed = x_freq * decay.view(1, 1, 1, C)
        
        # 6. 逆图傅里叶变换 (IGFT)
        x_out = torch.matmul(x_freq_decayed, eigvecs.t())
        
        return x_out

class Heat3D(nn.Module):
    """
    改进的 Heat3D 算子：
    - 空间维度：使用 DCT/IDCT 实现热传导扩散。
    - 光谱维度：使用图拉普拉斯算子实现语义感知扩散。
    """
    def __init__(self, infer_mode=False, res=14, hidden_dim=160, embed_dim=16, temperature=0.1, **kwargs):
        super().__init__()
        self.res = res
        self.hidden_dim = hidden_dim
        self.infer_mode = infer_mode
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        # 空间局部特征提取
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        
        # 线性变换与门控
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        
        # 光谱图扩散
        self.spectral_diff = SpectralGraphDiffusion(self.hidden_dim, self.embed_dim, self.temperature)
        
        # 空间扩散系数 k (由频率嵌入动态生成或固定)
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
        
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)

    @staticmethod
    def get_cos_map(N, device=torch.device("cpu"), dtype=torch.float):
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution, device=torch.device("cpu"), dtype=torch.float):
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, H, W = x.shape
        
        # 1. 空间局部特征
        x = self.dwconv(x)
        
        # 2. 线性变换并分拆
        x = self.linear(x.permute(0, 2, 3, 1).contiguous()) # (B, H, W, 2C)
        x, z = x.chunk(chunks=2, dim=-1) # (B, H, W, C)

        # 3. 空间维度扩散 (DCT)
        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]
        
        # 空间 DCT 变换
        x_spatial = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1))
        x_spatial = F.conv1d(x_spatial.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)
        
        # 空间频域衰减
        if freq_embed is not None:
            spatial_decay = torch.pow(weight_exp[:, :, None], self.to_k(freq_embed))
            x_spatial = torch.einsum("bnmc,nmc -> bnmc", x_spatial, spatial_decay)
        else:
            x_spatial = x_spatial * weight_exp[:, :, None]
        
        # 空间 IDCT 变换
        x = F.conv1d(x_spatial.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1))
        x = F.conv1d(x.contiguous().view(-1, M, C), weight_cosm.t().contiguous().view(W, M, 1)).contiguous().view(B, H, W, C)

        # 4. 光谱维度扩散 (图拉普拉斯)
        x = self.spectral_diff(x)

        # 5. 门控与输出
        x = self.out_norm(x)
        x = x * F.silu(z)
        x = self.out_linear(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

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
        self.op = Heat3D(hidden_dim=hidden_dim, embed_dim=int(math.log2(hidden_dim))+1, infer_mode=infer_mode, res=res, temperature=0.1)
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
