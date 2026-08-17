"""Conditional U-Net backbones used by the diffusion trainer."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels, preferred=32):
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ProfileEncoder(nn.Module):
    """Encode a profile of shape (batch, 3, 128) into a conditioning vector."""

    def __init__(self, out_dim, mode="cnn"):
        super().__init__()
        if mode == "mlp":
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * 128, out_dim),
                nn.SiLU(),
            )
        elif mode == "cnn":
            self.net = nn.Sequential(
                nn.Conv1d(3, 64, 5, padding=2),
                nn.SiLU(),
                nn.Conv1d(64, 128, 5, stride=2, padding=2),
                nn.SiLU(),
                nn.Conv1d(128, 256, 5, stride=2, padding=2),
                nn.SiLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(256, out_dim),
            )
        else:
            raise ValueError(f"unknown profile encoder {mode!r}")

    def forward(self, profile):
        if profile.ndim != 3 or tuple(profile.shape[1:]) != (3, 128):
            raise ValueError(
                f"profile must have shape (batch, 3, 128), got {tuple(profile.shape)}"
            )
        return self.net(profile)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, value):
        value = value.float().reshape(-1)
        half = self.dim // 2
        frequency = torch.exp(
            -math.log(10000) * torch.arange(half, device=value.device) / max(half - 1, 1)
        )
        angle = value[:, None] * frequency[None, :]
        embedding = torch.cat((angle.sin(), angle.cos()), dim=1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.cond = nn.Linear(cond_dim, out_channels * 2)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond(condition).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return (h + self.skip(x)) / math.sqrt(2.0)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(batch, 3, channels, height * width).unbind(1)
        attention = torch.softmax(torch.einsum("bcn,bcm->bnm", q, k) / math.sqrt(channels), dim=-1)
        h = torch.einsum("bnm,bcm->bcn", attention, v).reshape(batch, channels, height, width)
        return x + self.proj(h)


class ConditionalUNet(nn.Module):
    """Compact conditional U-Net with a compact or ADM-style configuration."""

    def __init__(
        self,
        image_channels=1,
        profile_encoder="cnn",
        backbone="compact",
        base_channels=None,
        dropout=0.0,
    ):
        super().__init__()
        if backbone not in {"compact", "adm"}:
            raise ValueError("backbone must be 'compact' or 'adm'")

        if base_channels is None:
            base_channels = 64 if backbone == "compact" else 96
        multipliers = (1, 2, 4, 4) if backbone == "compact" else (1, 2, 3, 4)
        channels = [base_channels * multiplier for multiplier in multipliers]
        cond_dim = channels[-1]
        embed_dim = channels[0] * 4

        self.time_embedding = nn.Sequential(
            SinusoidalEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, cond_dim),
        )
        self.profile_encoder = ProfileEncoder(cond_dim, profile_encoder)
        self.input = nn.Conv2d(image_channels, channels[0], 3, padding=1)

        attention_resolutions = {16, 32} if backbone == "adm" else {16}
        self.downs = nn.ModuleList()
        current = channels[0]
        resolution = 128
        skip_channels = []
        for index, out_channels in enumerate(channels):
            use_attention = resolution in attention_resolutions
            blocks = nn.ModuleList([
                ResBlock(current, out_channels, cond_dim, dropout),
                ResBlock(out_channels, out_channels, cond_dim, dropout),
            ])
            attention = AttentionBlock(out_channels) if use_attention else nn.Identity()
            down = (
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
                if index < len(channels) - 1
                else nn.Identity()
            )
            self.downs.append(nn.ModuleDict({"blocks": blocks, "attention": attention, "down": down}))
            skip_channels.append(out_channels)
            current = out_channels
            resolution //= 2

        self.middle = nn.ModuleList([
            ResBlock(current, current, cond_dim, dropout),
            AttentionBlock(current) if backbone == "adm" else nn.Identity(),
            ResBlock(current, current, cond_dim, dropout),
        ])

        self.ups = nn.ModuleList()
        for index, skip_channels_value in enumerate(reversed(skip_channels)):
            out_channels = skip_channels_value
            resolution *= 2
            up = (
                nn.ConvTranspose2d(current, current, 4, stride=2, padding=1)
                if index > 0
                else nn.Identity()
            )
            blocks = nn.ModuleList([
                ResBlock(current + skip_channels_value, out_channels, cond_dim, dropout),
                ResBlock(out_channels, out_channels, cond_dim, dropout),
            ])
            attention = AttentionBlock(out_channels) if resolution in attention_resolutions else nn.Identity()
            self.ups.append(nn.ModuleDict({"up": up, "blocks": blocks, "attention": attention}))
            current = out_channels

        self.output = nn.Sequential(
            nn.GroupNorm(_group_count(current), current),
            nn.SiLU(),
            nn.Conv2d(current, image_channels, 3, padding=1),
        )

    def forward(self, image, noise_level, profile):
        condition = self.time_embedding(noise_level) + self.profile_encoder(profile)
        h = self.input(image)
        skips = []
        for level in self.downs:
            for block in level["blocks"]:
                h = block(h, condition)
            h = level["attention"](h)
            skips.append(h)
            h = level["down"](h)

        h = self.middle[0](h, condition)
        h = self.middle[1](h)
        h = self.middle[2](h, condition)

        for level in self.ups:
            h = level["up"](h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat((h, skip), dim=1)
            for block in level["blocks"]:
                h = block(h, condition)
            h = level["attention"](h)

        return self.output(h)
