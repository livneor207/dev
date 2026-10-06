import torch
from torch import nn
from torch.nn import functional as F

from defusion_model.diffusion.blocks import (
    AttentionBlock,
    Downsample,
    ResidualBlock,
    Upsample,
    create_spatial_projection,
    group_norm_groups,
    sinusoidal_timestep_embedding,
)


# Sentinel label for the unconditional branch of classifier-free guidance.
NULL_CLASS = -1


class UNet(nn.Module):
    def __init__(self, in_channels=3, model_channels=64, out_channels=3, channel_multipliers=(1, 2, 4),
                 num_res_blocks=2, attention_resolutions=(16,), dropout=0.1, num_classes=0,
                 image_size=32, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.channel_multipliers = tuple(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = tuple(attention_resolutions)
        self.dropout = dropout
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_heads = num_heads
        self.is_multi_task = False
        time_embed_dim = int(model_channels * 4)
        self.time_embed_dim = time_embed_dim
        # -----------
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.RMSNorm(normalized_shape=time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.RMSNorm(normalized_shape=time_embed_dim),
        )
        if num_classes > 0:
            self.class_embed = nn.Sequential(
                nn.Embedding(num_classes, time_embed_dim),
                nn.RMSNorm(normalized_shape=time_embed_dim),
            )
        else:
            self.class_embed = nn.Identity()
        # -----------
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        self.down_ops = []
        ch = model_channels
        resolution = image_size
        skip_channels = [model_channels]
        for level, multiplier in enumerate(self.channel_multipliers):
            out_ch = int(model_channels * multiplier)
            for block_idx in range(num_res_blocks):
                name = f'down_{level}_res_{block_idx}'
                setattr(self, name, ResidualBlock(ch, out_ch, time_embed_dim, dropout))
                use_attn = resolution in self.attention_resolutions
                self.down_ops.append({'name': name, 'kind': 'res', 'store_skip': not use_attn})
                ch = out_ch
                if use_attn:
                    attn_name = f'down_{level}_attn_{block_idx}'
                    setattr(self, attn_name, AttentionBlock(ch, num_heads=num_heads))
                    self.down_ops.append({'name': attn_name, 'kind': 'attn', 'store_skip': True})
                skip_channels.append(ch)
            if level != len(self.channel_multipliers) - 1:
                down_name = f'down_{level}_downsample'
                setattr(self, down_name, Downsample(ch))
                self.down_ops.append({'name': down_name, 'kind': 'down', 'store_skip': True})
                skip_channels.append(ch)
                resolution //= 2
        # -----------
        self.mid_res_1 = ResidualBlock(ch, ch, time_embed_dim, dropout)
        self.mid_attn = AttentionBlock(ch, num_heads=num_heads)
        self.mid_res_2 = ResidualBlock(ch, ch, time_embed_dim, dropout)
        # -----------
        self.up_ops = []
        for level, multiplier in reversed(list(enumerate(self.channel_multipliers))):
            out_ch = int(model_channels * multiplier)
            for block_idx in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                fusion_dim = int(ch + skip_ch)
                fuse_name = f'up_{level}_fuse_{block_idx}'
                res_name = f'up_{level}_res_{block_idx}'
                setattr(self, fuse_name, create_spatial_projection(fusion_dim, out_ch))
                setattr(self, res_name, ResidualBlock(out_ch, out_ch, time_embed_dim, dropout))
                use_attn = resolution in self.attention_resolutions
                self.up_ops.append({
                    'name': res_name,
                    'fuse': fuse_name,
                    'kind': 'res',
                    'out_channels': out_ch,
                })
                ch = out_ch
                if use_attn:
                    attn_name = f'up_{level}_attn_{block_idx}'
                    setattr(self, attn_name, AttentionBlock(ch, num_heads=num_heads))
                    self.up_ops.append({'name': attn_name, 'kind': 'attn', 'fuse': None})
            if level != 0:
                up_name = f'up_{level}_upsample'
                setattr(self, up_name, Upsample(ch))
                self.up_ops.append({'name': up_name, 'kind': 'up', 'fuse': None})
                resolution *= 2
        if skip_channels:
            raise RuntimeError(f'UNet skip channel mismatch, leftover={skip_channels}')
        # -----------
        self.output_norm = nn.GroupNorm(group_norm_groups(ch), ch)
        self.output_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def config_dict(self):
        return {
            'in_channels': self.in_channels,
            'model_channels': self.model_channels,
            'out_channels': self.out_channels,
            'channel_multipliers': self.channel_multipliers,
            'num_res_blocks': self.num_res_blocks,
            'attention_resolutions': self.attention_resolutions,
            'dropout': self.dropout,
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'num_heads': self.num_heads,
        }

    def forward(self, noisy_image, timestep, class_label=None):
        """Returns predicted noise with the same layout as noisy_image, [B, C, H, W].

        ``class_label`` entries of ``NULL_CLASS`` (-1) are the unconditional token
        used by classifier-free guidance: their class embedding is masked to zero
        so one model serves both the conditional and unconditional score.
        """
        time_features = self.time_embed(sinusoidal_timestep_embedding(timestep, self.model_channels))
        if self.num_classes > 0 and class_label is not None:
            keep = (class_label >= 0).to(time_features.dtype).unsqueeze(-1)
            class_features = self.class_embed(class_label.clamp(min=0))
            time_features = time_features + class_features * keep
        h = self.input_conv(noisy_image)
        skips = [h]
        for op in self.down_ops:
            h = getattr(self, op['name'])(h, time_features)
            if op['store_skip']:
                skips.append(h)
        h = self.mid_res_1(h, time_features)
        h = self.mid_attn(h, time_features)
        h = self.mid_res_2(h, time_features)
        for op in self.up_ops:
            if op['kind'] == 'res':
                skip = skips.pop()
                fused = torch.concat((h, skip), dim=1)
                h = getattr(self, op['fuse'])(fused)
                h = getattr(self, op['name'])(h, time_features)
            else:
                h = getattr(self, op['name'])(h, time_features)
        if skips:
            raise RuntimeError(f'UNet skip stack not empty, leftover={len(skips)}')
        return self.output_conv(F.silu(self.output_norm(h)))
