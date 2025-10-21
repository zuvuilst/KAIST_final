# models_diffusion.py
# -*- coding: utf-8 -*-
"""
RADA diff (diff_RATD) / CSDI (diff_CSDI) 모듈 래핑
- 외부 의존(deps)이 없으면 ImportError를 던지고, 상위(generator)에서 폴백 처리.
- forward shape 규약:
  x:          (B, inputdim, K, L)    # inputdim=2 (value, mask) 권장
  cond_info:  (B, side_dim, K, L)    # 공간/거시/이웃통계 등 시간축 브로드캐스트
  reference:  (B, K, L, ref_dim)     # RADA 전용 (CSDI는 None)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- optional deps guard ----------
try:
    from einops import repeat, rearrange
except Exception as e:
    repeat = rearrange = None

def _requires(pkg: str):
    raise ImportError(f"[models_diffusion] '{pkg}' 가 필요합니다. "
                      f"pip install einops diffusers linear-attention-transformer")

# ========= shared small utils =========
def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=False
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

# =============== CSDI ===============
class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x); x = F.silu(x)
        x = self.projection2(x); x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / max(1, (dim - 1)) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class ResidualBlockCSDI(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads=4, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.is_linear = is_linear
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def _forward_time(self, y, base_shape):
        B, C, K, L = base_shape
        if L == 1: return y
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B * K, C, L)   # (B*K,C,L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K * L)
        return y

    def _forward_feature(self, y, base_shape):
        B, C, K, L = base_shape
        if K == 1: return y
        y = y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B * L, C, K)   # (B*L,C,K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, C, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, C, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,C,1)
        y = x + diffusion_emb

        y = self._forward_time(y, base_shape)
        y = self._forward_feature(y, base_shape)
        y = self.mid_projection(y)  # (B,2C,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2C,K*L)
        y = y + cond_info

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)  # (B,C,K*L)
        y = self.output_projection(y)               # (B,2C,K*L)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip

class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)
        self.residual_layers = nn.ModuleList([
            ResidualBlockCSDI(
                side_dim=config["side_dim"],
                channels=self.channels,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                nheads=config.get("nheads", 4),
                is_linear=config.get("is_linear", False),
            ) for _ in range(config["layers"])
        ])

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x); x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, sk = layer(x, cond_info, diffusion_emb); skip.append(sk)
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x); x = F.relu(x)
        x = self.output_projection2(x)
        x = x.reshape(B, K, L)
        return x

# =============== RADA diff (RATD) ===============
# 부분적으로 diffusers / einops 의존
class ReferenceModulatedCrossAttention(nn.Module):
    def __init__(self, *, dim, heads=8, dim_head=64, context_dim=None, dropout=0., talking_heads=False, prenorm=False):
        super().__init__()
        if repeat is None or rearrange is None:
            _requires("einops")
        context_dim = context_dim or dim
        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)
        self.y_to_q = nn.Linear(dim, inner_dim, bias=False)
        self.cond_to_k = nn.Linear(2*dim + context_dim, inner_dim, bias=False)
        self.ref_to_v  = nn.Linear(dim + context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)
        self.talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()

    def forward(self, x, cond_info, reference, return_attn=False):
        if repeat is None or rearrange is None:
            _requires("einops")
        B, C, K, L, h, device = x.shape[0], x.shape[1], x.shape[2], x.shape[-1], self.heads, x.device
        x = self.norm(x); reference = self.norm(reference); cond_info = self.context_norm(cond_info)
        reference = repeat(reference, 'b n c -> (b f) n c', f=C)       # (B*C, K, L?)
        q_y = self.y_to_q(x.reshape(B*C, K, L))
        cond = self.cond_to_k(torch.cat((x.reshape(B*C, K, L), cond_info.reshape(B*C, K, L), reference), dim=-1))
        ref  = self.ref_to_v(torch.cat((x.reshape(B*C, K, L), reference), dim=-1))
        q_y, cond, ref = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q_y, cond, ref))
        sim = torch.einsum('b h i d, b h j d -> b h i j', cond, ref) * self.scale
        attn = sim.softmax(dim=-1); context_attn = sim.softmax(dim=-2)
        attn = self.dropout(attn); context_attn = self.context_dropout(context_attn)
        attn = self.talking_heads(attn); context_attn = self.context_talking_heads(context_attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, ref)
        context_out = torch.einsum('b h j i, b h j d -> b h i d', context_attn, cond)
        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))
        out = self.to_out(out)
        if return_attn: return out, context_out, attn, context_attn
        return out

class ResidualBlockRATD(nn.Module):
    def __init__(self, side_dim, ref_size, h_size, channels, diffusion_embedding_dim, nheads=4, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, channels, 1)
        self.mid_projection  = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.is_linear = is_linear
        self.time_layer    = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

        dim_heads = 8
        self.q_dim = nheads*dim_heads
        # RMA
        self.RMA = ReferenceModulatedCrossAttention(dim=ref_size + h_size, context_dim=ref_size*3)
        self.fusion_type = 1

    def _forward_time(self, y, base_shape):
        B, C, K, L = base_shape
        if L == 1: return y
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B * K, C, L)
        y = self.time_layer(y.permute(2,0,1)).permute(1,2,0)
        y = y.reshape(B, K, C, L).permute(0,2,1,3).reshape(B, C, K*L)
        return y

    def _forward_feature(self, y, base_shape):
        B, C, K, L = base_shape
        if K == 1: return y
        y = y.reshape(B, C, K, L).permute(0,3,1,2).reshape(B*L, C, K)
        y = self.feature_layer(y.permute(2,0,1)).permute(1,2,0)
        y = y.reshape(B, L, C, K).permute(0,2,3,1).reshape(B, C, K*L)
        return y

    def forward(self, x, cond_info, diffusion_emb, reference):
        B, C, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, C, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x + diffusion_emb

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)

        if reference is not None and self.fusion_type == 1:
            # RMA expects shapes as in ReferenceModulatedCrossAttention.forward
            # here we simply pass through; generator에서 reference shape 준비
            cond_info = self.RMA(y.reshape(B, C, K, L), cond_info.reshape(B, C, K, L), reference)

        y = y + cond_info.reshape(B, C, K*L)
        y = self._forward_time(y, base_shape)
        y = self._forward_feature(y, base_shape)
        y = self.mid_projection(y)

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip

class diff_RATD(nn.Module):
    def __init__(self, config, inputdim=2, use_ref=True):
        super().__init__()
        self.channels = config["channels"]
        self.use_ref = use_ref
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)
        self.residual_layers = nn.ModuleList([
            ResidualBlockRATD(
                side_dim=config["side_dim"],
                ref_size=config["ref_size"],
                h_size=config["h_size"],
                channels=self.channels,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                nheads=config.get("nheads", 4),
                is_linear=config.get("is_linear", False),
            ) for _ in range(config["layers"])
        ])

    def forward(self, x, cond_info, diffusion_step, reference=None):
        if self.use_ref and reference is None:
            raise ValueError("diff_RATD: reference is required when use_ref=True")
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x); x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, sk = layer(x, cond_info, diffusion_emb, reference)
            skip.append(sk)
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x); x = F.relu(x)
        x = self.output_projection2(x)
        x = x.reshape(B, K, L)
        return x
