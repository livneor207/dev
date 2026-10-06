import math

import torch
from torch import nn
from torch.nn import functional as F


def group_norm_groups(channels, max_groups=32):
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def create_spatial_projection(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.GroupNorm(group_norm_groups(out_channels), out_channels),
    )


def sinusoidal_timestep_embedding(timestep, embed_dim):
    """timestep: [B] long/int -> [B, embed_dim] float."""
    half_dim = embed_dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half_dim, device=timestep.device, dtype=torch.float32) / max(half_dim - 1, 1)
    )
    args = timestep.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
    if embed_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.GroupNorm(group_norm_groups(in_channels), in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, int(2 * out_channels)),
            nn.RMSNorm(normalized_shape=int(2 * out_channels)),
        )
        self.norm_2 = nn.GroupNorm(group_norm_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.conv_2.weight)
        nn.init.zeros_(self.conv_2.bias)

    def forward(self, x, time_embed=None):
        h = self.conv_1(F.silu(self.norm_1(x)))
        scale, shift = self.time_proj(time_embed).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm_2(h) * (1 + scale) + shift
        h = self.conv_2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        if channels % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(group_norm_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, int(3 * channels), kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, time_embed=None):
        b, channels, height, width = x.shape
        head_dim = channels // self.num_heads
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b, 3, self.num_heads, head_dim, height * width)
        q = qkv[:, 0].permute(0, 1, 3, 2)
        k = qkv[:, 1].permute(0, 1, 3, 2)
        v = qkv[:, 2].permute(0, 1, 3, 2)
        # q, k, v: [B, heads, H*W, head_dim]
        attended = F.scaled_dot_product_attention(q, k, v)
        out = attended.permute(0, 1, 3, 2).reshape(b, channels, height, width)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, time_embed=None):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, time_embed=None):
        h = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(h)
