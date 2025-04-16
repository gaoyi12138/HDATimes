from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_layers import *
from layers.RevIN import RevIN
from layers.box_coder1D import *
import math


# generate_pairs(n): 生成给定数量 n 的所有可能的索引对 (i, j)，排除 i 等于 j 的情况。
def generate_pairs(n):
    pairs = []

    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append([i, j])

    return np.array(pairs)


# cal_PSI(x, r): 计算补丁相似度指数 (PSI)，通过计算张量 x 中补丁对之间的绝对差异。返回最大绝对差异小于阈值 r 的对数均值。
def cal_PSI(x, r):
    # [bs x nvars x patch_len x patch_num]
    x = x.permute(0, 1, 3, 2)
    batch, n_vars, patch_num, patch_len = x.shape
    x = x.reshape(batch * n_vars, patch_num, patch_len)
    # Generate all possible pairs of patch_num indices within each batch
    pairs = generate_pairs(patch_num)
    # Calculate absolute differences between pairs of sequences
    abs_diffs = torch.abs(x[:, pairs[:, 0], :] - x[:, pairs[:, 1], :])
    # Find the maximum absolute difference for each pair of sequences
    max_abs_diffs = torch.max(abs_diffs, dim=-1).values
    max_abs_diffs = max_abs_diffs.reshape(-1, patch_num, patch_num - 1)
    # Count the number of pairs with max absolute difference less than r
    c = torch.log(1 + torch.mean((max_abs_diffs < r).float(), dim=-1))
    psi = torch.mean(c, dim=-1)
    return psi


# cal_PaEn(lfp, lep, r, lambda_): 通过比较两个张量（lfp 和 lep）的 PSI 来计算 PaEn 损失。包含一个由 lambda_ 控制的正则化项。
def cal_PaEn(lfp, lep, r, lambda_):
    psi_lfp = cal_PSI(lfp, r)
    psi_lep = cal_PSI(lep, r)
    psi_diff = psi_lfp - psi_lep
    lep = lep.permute(0, 1, 3, 2)
    batch, n_vars, patch_num, patch_len = lep.shape
    lep = lep.reshape(batch * n_vars, patch_num, patch_len)
    sum_x = torch.sum(lep, dim=[-2, -1])
    PaEN_loss = torch.mean(sum_x * psi_diff) * lambda_  # update parameters with REINFORCE
    return PaEN_loss


# Model 类
# 目的: 作为模型的主要接口，接受输入数据，通过主干模型处理数据，并输出预测结果和损失。
class Model(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = False,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):
        super().__init__()

        # load parameters
        c_in = configs.enc_in
        self.seq_len = context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        # model
        self.decomposition = decomposition
        self.model = HDMixer_backbone(configs, c_in=c_in, context_window=context_window, target_window=target_window,
                                      patch_len=patch_len, stride=stride,
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                      attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                      padding_var=padding_var,
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                      store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                                      padding_patch=padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                      revin=revin, affine=affine,
                                      subtract_last=subtract_last, verbose=verbose, **kwargs)

    def forward(self, x):  # x: [Batch, Input length, Channel]

        x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        x, PaEN_Loss = self.model(x)
        x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x, PaEN_Loss


# HDMixer_backbone 类
# 目的: 实现模型的核心部分，处理输入的归一化、补丁提取，并通过编码器处理数据
class HDMixer_backbone(nn.Module):
    def __init__(self, configs, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int,
                 max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True,
                 subtract_last=False,
                 verbose: bool = False, **kwargs):

        super().__init__()

        # RevIn层: 归一化输入数据，是否使用取决于配置。
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.deform_patch = configs.deform_patch
        if self.deform_patch:
            self.patch_len = patch_len
            self.stride = stride
            self.patch_num = patch_num = context_window // self.stride
            self.patch_shift_linear = nn.Linear(context_window, self.patch_num * 3)
            self.box_coder = pointwhCoder(input_size=context_window, patch_count=self.patch_num, weights=(1., 1., 1.),
                                          pts=self.patch_len, tanh=True, wh_bias=torch.tensor(5. / 3.).sqrt().log(),
                                          deform_range=configs.deform_range)
            self.lambda_ = configs.lambda_
            self.r = configs.r
        else:
            # Patching：将输入序列划分为更小的补丁以便处理
            self.patch_len = patch_len
            self.stride = stride
            self.padding_patch = padding_patch
            patch_num = int((context_window - patch_len) / stride + 1)  # patch的数量是(336-16)/8 + 1 向下取整
            if padding_patch == 'end':  # can be modified to general case
                self.padding_patch_layer = nn.ReplicationPad1d(
                    (0, stride))  # 使用了 PyTorch 中的 nn.ReplicationPad1d 模块，用于对一维张量进行复制填充操作
                patch_num += 1
        # Backbone ：使用 Encoder 类处理补丁。
        self.backbone = Encoder(configs, c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.head_type = head_type
        self.individual = individual

        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        batch_size = z.shape[0]
        seq_len = z.shape[-1]
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        x_lfp = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
        x_lfp = x_lfp.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]
        if self.deform_patch:
            anchor_shift = self.patch_shift_linear(z).view(batch_size * self.n_vars, self.patch_num, 3)
            sampling_location_1d = self.box_coder(anchor_shift)  # B*C, self.patch_num,self.patch_len, 1
            add1d = torch.ones(size=(batch_size * self.n_vars, self.patch_num, self.patch_len, 1)).float().to(
                sampling_location_1d.device)
            sampling_location_2d = torch.cat([sampling_location_1d, add1d], dim=-1)
            z = z.reshape(batch_size * self.n_vars, 1, 1, seq_len)
            patch = F.grid_sample(z, sampling_location_2d, mode='bilinear', padding_mode='border',
                                  align_corners=False).squeeze(1)  # B*C, self.patch_num,self.patch_len
            x_lep = patch.reshape(batch_size, self.n_vars, self.patch_num, self.patch_len).permute(0, 1, 3,
                                                                                                   2)  # [bs x nvars x patch_len x patch_num]
            PaEN_Loss = cal_PaEn(x_lfp, x_lep, self.r, self.lambda_)
        else:
            if self.padding_patch == 'end':
                z = self.padding_patch_layer(z)
            z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
            z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]
            patch = z
        # model
        z = self.backbone(x_lep)  # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)  # z: [bs x nvars x target_window]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z, PaEN_Loss


# 将编码器的输出映射到目标输出形状。可以单独或整体处理多个变量的输出
class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


# Encoder 类
# 目的: 将输入补丁编码成适合后续层处理的表示。结合了补丁编码和位置编码。
# 组件:线性投影: 将补丁映射到更高维空间。
#     位置编码: 编码每个补丁的位置，以保持序列信息。
class Encoder(nn.Module):  # i means channel-independent
    def __init__(self, configs, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len  #

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = HDMixer(configs, q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                               attn_dropout=attn_dropout, dropout=dropout,
                               pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                               store_attn=store_attn)

    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(x)  # z: [bs x nvars x patch_num x d_model]
        return z


# HDMixer 类：包含多个处理层，跨时间、变量和通道混合信息。
class HDMixer(nn.Module):
    def __init__(self, configs, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([HDMixerLayer(configs, q_len,
                                                  d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                  attn_dropout=attn_dropout, dropout=dropout,
                                                  activation=activation, res_attention=res_attention,
                                                  pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return output


# 应用层归一化，保持训练过程的稳定性并改善收敛性。
class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(768)
    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        x=x.permute(0,2,1)
        #print(f"x.shape before norm:{x.shape}")
        x = self.norm(x)
        return x


# 定义每一层的架构，包括跨时间、变量和补丁的混合操作。
class HDMixerLayer(nn.Module):
    def __init__(self, configs, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False):
        super().__init__()

        c_in = configs.enc_in
        # Add & Norm
        # [bs x nvars x patch_num x d_model]
        # Position-wise Feed-Forward
        self.mix_time = configs.mix_time
        self.mix_variable = configs.mix_variable
        self.mix_channel = configs.mix_channel
        self.patch_mixer = nn.Sequential(
            LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model, bias=bias),
            nn.Dropout(dropout),
        )
        self.time_mixer = nn.Sequential(
            Transpose(2, 3), LayerNorm(q_len),
            nn.Linear(q_len, q_len * 2, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(q_len * 2, q_len, bias=bias),
            nn.Dropout(dropout),
            Transpose(2, 3)
        )
        # [bs x nvars  x d_model  x patch_num] ->  [bs x nvars x patch_num x d_model]

        # [bs x nvars x patch_num x d_model]
        self.variable_mixer = nn.Sequential(
            Transpose(1, 3), LayerNorm(c_in),
            nn.Linear(c_in, c_in * 2, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(c_in * 2, c_in, bias=bias),
            nn.Dropout(dropout),
            Transpose(1, 3)
        )

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        # [bs x nvars x patch_num x d_model]

        #print(f"src shape: {src.shape}")
        #src = src.permute(0, 2, 1)
        # 如果 `src` 需要被展平
        #batch_size, seq_len, feature_dim = src.shape  # 假设 src 是 (batch_size, seq_len, feature_dim)

        # 确保展平后的形状匹配 `patch_mixer` 的输入
        #src = src.reshape(-1, feature_dim)  # (batch_size * seq_len, feature_dim)

        #print(f"Flattened src shape: {src.shape}")
        #print(f"patch_mixer(src).shape: {self.patch_mixer(src).shape}")
        g=self.patch_mixer(src).permute(0,2,1)
        if self.mix_channel:
            u = g + src
        else:
            u = src
        if self.mix_time:
            v = g + src
        else:
            v = u
        if self.mix_variable:
            w = g + src
        else:
            w = v
        out = w
        return out