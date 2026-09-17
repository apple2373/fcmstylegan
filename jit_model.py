"""Profile-conditioned adapter around the local JiT implementation."""
from pathlib import Path
import sys
import torch
import torch.nn as nn

_JIT_ROOT = Path(__file__).resolve().parent / "JiT"
if str(_JIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_JIT_ROOT))
from model_jit import JiT_models  # noqa: E402

class ProfileMLP(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 128, out_dim), nn.SiLU())

    def forward(self, profile):
        if profile.ndim != 3 or tuple(profile.shape[1:]) != (3, 128):
            raise ValueError(f"profile must have shape (batch, 3, 128), got {tuple(profile.shape)}")
        return self.net(profile)

class ConditionalJiT(nn.Module):
    def __init__(self, model_name="JiT-B/16", img_size=128, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.net = JiT_models[model_name](input_size=img_size, in_channels=1, num_classes=1,
                                          attn_drop=attn_dropout, proj_drop=proj_dropout)
        # Reuse JiT's AdaLN and in-context conditioning path with profile embeddings.
        self.net.y_embedder = ProfileMLP(self.net.hidden_size)

    def forward(self, image, timestep, profile):
        return self.net(image, timestep, profile)


class OriginalProfileMLP(nn.Module):
    """Profile projection with JiT-style learned null conditioning."""

    def __init__(self, out_dim, drop_prob=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 128, out_dim), nn.SiLU())
        self.null_embedding = nn.Parameter(torch.zeros(out_dim))
        self.drop_prob = drop_prob
        self.force_unconditional = False

    def forward(self, profile):
        if profile.ndim != 3 or tuple(profile.shape[1:]) != (3, 128):
            raise ValueError(f"profile must have shape (batch, 3, 128), got {tuple(profile.shape)}")
        embedding = self.net(profile)
        if self.force_unconditional:
            return self.null_embedding.expand(profile.shape[0], -1)
        if self.training and self.drop_prob > 0:
            dropped = torch.rand(profile.shape[0], device=profile.device) < self.drop_prob
            embedding = torch.where(dropped[:, None], self.null_embedding[None, :], embedding)
        return embedding


class ConditionalJiTOriginal(ConditionalJiT):
    """JiT adapter using the paper-style clean-image parameterization."""

    def __init__(self, model_name="JiT-B/16", img_size=128, attn_dropout=0.0,
                 proj_dropout=0.0, profile_drop_prob=0.1):
        super().__init__(model_name, img_size, attn_dropout, proj_dropout)
        self.net.y_embedder = OriginalProfileMLP(self.net.hidden_size, profile_drop_prob)

    def forward(self, image, timestep, profile, unconditional=False):
        embedder = self.net.y_embedder
        previous = embedder.force_unconditional
        embedder.force_unconditional = unconditional
        try:
            return self.net(image, timestep, profile)
        finally:
            embedder.force_unconditional = previous
